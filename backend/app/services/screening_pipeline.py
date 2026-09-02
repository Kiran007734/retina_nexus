"""End-to-end screening orchestration for the current synchronous worker.

The runner is deliberately isolated from HTTP. It updates durable run state
between stages, so the same execute method can later be called by a queue
worker without changing the public contract.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.safe_errors import safe_error_message
from app.database.session import SessionLocal
from app.models.audit_log import AuditLog
from app.models.fundus_image import FundusImage, QualityDecision
from app.models.screening import ScreeningSession, ScreeningStatus
from app.models.screening_run import ScreeningRun
from app.ml.evidence.service import RetinalEvidenceAnalysis, RetinalEvidenceService
from app.ml.explainability.service import ExplainabilityAnalysis, ExplainabilityService
from app.ml.inference.classifier import ClassifierNotConfiguredError, DRPrediction, TorchDRClassificationService
from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService, TrustGateDecision, TrustGateOutcome
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs, RetinaGuardResult, derive_lesion_evidence_strength, derive_vessel_evidence_status
from app.services.screening_persistence import persist_evidence_analysis, persist_explainability

logger = logging.getLogger(__name__)

RUN_STAGES = (
    "image_validation", "quality_assessment", "dr_classification", "retinal_structure_analysis",
    "lesion_detection", "grad_cam", "attention_lesion_agreement", "uncertainty",
    "model_disagreement", "retinaguard", "triage",
)

OPTIONAL_EVIDENCE_STAGES = (
    "retinal_structure_analysis", "lesion_detection", "grad_cam", "attention_lesion_agreement",
)


class OptionalStageTimeout(Exception):
    """Signals a bounded optional operation without cancelling its worker thread."""

    def __init__(self, stage: str, budget_seconds: int, task: asyncio.Task[Any]) -> None:
        super().__init__(f"{stage} exceeded its {budget_seconds}s runtime budget")
        self.stage = stage
        self.budget_seconds = budget_seconds
        self.task = task


@dataclass
class ScreeningPipelineOutput:
    screening_id: UUID
    status: str
    quality: dict[str, Any] | None
    classification: dict[str, Any] | None
    lesions: dict[str, Any] | None
    explainability: dict[str, Any] | None
    retinaguard: dict[str, Any] | None
    triage: dict[str, Any] | None
    model_versions: dict[str, Any]
    stage_status: dict[str, str]
    stage_metrics: dict[str, Any]
    stage_errors: dict[str, Any]
    error: dict[str, Any] | None
    prediction: DRPrediction | None = None
    evidence_analysis: RetinalEvidenceAnalysis | None = None
    explanation_analysis: ExplainabilityAnalysis | None = None
    retinaguard_result: RetinaGuardResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "screening_id": self.screening_id,
            "status": self.status,
            "quality": self.quality,
            "classification": self.classification,
            "lesions": self.lesions,
            "explainability": self.explainability,
            "retinaguard": self.retinaguard,
            "triage": self.triage,
            "model_versions": self.model_versions,
            "stage_status": self.stage_status,
            "stage_metrics": self.stage_metrics,
            "stage_errors": self.stage_errors,
            "error": self.error,
        }


class ScreeningPipelineService:
    """Run every current AI stage with explicit state and audit boundaries."""

    def __init__(self, quality_service: ImageTrustGateService, classifier: TorchDRClassificationService, evidence_service: RetinalEvidenceService, explainability_service: ExplainabilityService, retinaguard: RetinaGuardEngine, storage: Any, max_concurrent_screenings: int = 1, timeout_seconds: int = 900, primary_timeout_seconds: int = 60, optional_evidence_timeout_seconds: int = 240, optional_explainability_timeout_seconds: int = 30):
        self.quality_service = quality_service
        self.classifier = classifier
        self.evidence_service = evidence_service
        self.explainability_service = explainability_service
        self.retinaguard = retinaguard
        self.storage = storage
        concurrency = max(1, min(8, int(max_concurrent_screenings)))
        self._screening_slots = asyncio.Semaphore(concurrency)
        self._optional_slots = asyncio.Semaphore(concurrency)
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.timeout_seconds = max(30, int(timeout_seconds))
        # These defaults are based on observed local CPU timings: classification
        # was ~0.6-3.3s, Grad-CAM/agreement ~0.9-8.4s, while combined vessel and
        # lesion evidence ranged from ~1.4s to ~310s and once exceeded 900s.
        # The budgets bound optional work without changing model behaviour.
        self.primary_timeout_seconds = max(10, int(primary_timeout_seconds))
        self.optional_evidence_timeout_seconds = max(30, int(optional_evidence_timeout_seconds))
        self.optional_explainability_timeout_seconds = max(10, int(optional_explainability_timeout_seconds))

    async def execute(self, db: AsyncSession, run: ScreeningRun, session: ScreeningSession, image: FundusImage, actor_id: UUID | None, run_stability: bool | None = None, run_counterfactual: bool | None = None, model_predictions: list[dict[str, Any]] | None = None, defer_optional: bool = False) -> ScreeningPipelineOutput:
        acquired = False
        try:
            await asyncio.wait_for(self._screening_slots.acquire(), timeout=min(60, self.timeout_seconds))
            acquired = True
            return await asyncio.wait_for(
                self._execute_inner(db, run, session, image, actor_id, run_stability, run_counterfactual, model_predictions, defer_optional),
                timeout=self.primary_timeout_seconds if defer_optional else self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            run.status = "FAILED"
            run.error = {"stage": "pipeline", "type": "TimeoutError", "message": "Screening exceeded the configured processing limit; no complete prediction was returned."}
            run.stage_errors = {**(run.stage_errors or {}), "pipeline": run.error}
            run.stage_status = {**(run.stage_status or {}), "pipeline": "FAILED"}
            run.completed_at = datetime.now(timezone.utc)
            session.status = ScreeningStatus.FAILED
            session.completed_at = run.completed_at
            await db.commit()
            logger.error("screening.pipeline.timeout", extra={"event": "screening.pipeline.timeout", "stage": "pipeline", "screening_id": str(run.id)})
            return self._output(run)
        finally:
            if acquired:
                self._screening_slots.release()

    async def execute_primary(self, db: AsyncSession, run: ScreeningRun, session: ScreeningSession, image: FundusImage, actor_id: UUID | None, run_stability: bool | None = None, run_counterfactual: bool | None = None, model_predictions: list[dict[str, Any]] | None = None) -> ScreeningPipelineOutput:
        """Return the mandatory screening result without waiting for enrichment."""
        return await self.execute(
            db, run, session, image, actor_id,
            model_predictions=model_predictions,
            defer_optional=True,
        )

    def start_optional_processing(self, run_id: UUID, image_id: UUID, actor_id: UUID | None, run_stability: bool | None = None, run_counterfactual: bool | None = None) -> None:
        """Schedule evidence work on the current local worker.

        This is intentionally a small in-process background mechanism for the
        prototype. The durable run record exposes progress and survives the
        HTTP response; a queue can replace this method later.
        """
        task = asyncio.create_task(self._optional_worker(run_id, image_id, actor_id, run_stability, run_counterfactual))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _optional_worker(self, run_id: UUID, image_id: UUID, actor_id: UUID | None, run_stability: bool | None, run_counterfactual: bool | None) -> None:
        try:
            async with self._optional_slots:
                async with SessionLocal() as db:
                    run = await db.get(ScreeningRun, run_id)
                    image = await db.get(FundusImage, image_id)
                    # ScreeningRun uses the session primary key as its own id;
                    # it has no separate session_id column.
                    session = await db.get(ScreeningSession, run.id) if run else None
                    if not run or not image or not session or not run.classification:
                        logger.warning("screening.optional.not_started", extra={"event": "screening.optional.not_started", "screening_id": str(run_id)})
                        return
                    content_path = image.enhanced_storage_path or image.storage_path
                    prepared = await self.storage.get(content_path)
                    await self._execute_optional(db, run, session, image, actor_id, prepared, run_stability, run_counterfactual)
        except asyncio.CancelledError:
            logger.warning("screening.optional.cancelled", extra={"event": "screening.optional.cancelled", "screening_id": str(run_id)})
            raise
        except Exception:
            logger.exception("screening.optional.worker_failed", extra={"event": "screening.optional.worker_failed", "screening_id": str(run_id)})

    async def _execute_inner(self, db: AsyncSession, run: ScreeningRun, session: ScreeningSession, image: FundusImage, actor_id: UUID | None, run_stability: bool | None = None, run_counterfactual: bool | None = None, model_predictions: list[dict[str, Any]] | None = None, defer_optional: bool = False) -> ScreeningPipelineOutput:
        run.status = "PROCESSING"
        run.started_at = datetime.now(timezone.utc)
        run.model_versions = {"preprocessing": "image-trust-gate-v1"}
        session.status = ScreeningStatus.PROCESSING
        await self._audit(db, actor_id, "screening.run.processing", run.id, {"stage_count": len(RUN_STAGES)})
        await db.commit()
        current_stage = "image_validation"
        try:
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            content = await self.storage.get(image.storage_path)
            metadata = self.quality_service.validate_input(content)
            run.quality = {"input_metadata": asdict(metadata)}
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"format": metadata.format, "width": metadata.width, "height": metadata.height})

            current_stage = "quality_assessment"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            outcome, prepared = await self._assess_quality(db, image, content)
            run.quality = {"initial": outcome.initial.to_dict(), "final": outcome.final.to_dict(), "enhancement_applied": outcome.enhancement_applied, "enhancement_passes": outcome.enhancement_passes}
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"decision": outcome.final.quality_decision, "enhancement_passes": outcome.enhancement_passes})

            if outcome.final.quality_decision != TrustGateDecision.GRADABLE:
                return await self._finish_quality_block(db, run, session, image, actor_id, outcome)

            current_stage = "dr_classification"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            prediction = await self.classifier.classify(prepared)
            classification = self._classification_payload(prediction)
            run.classification = classification
            run.model_versions = {"preprocessing": "image-trust-gate-v1", "dr_classifier": prediction.model_version, "dr_backbone": prediction.backbone}
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"model_version": prediction.model_version})

            if defer_optional:
                return await self._finish_primary(
                    db, run, session, image, actor_id, prepared, prediction,
                    model_predictions or [],
                )

            evidence = None
            current_stage = "retinal_structure_analysis"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            current_stage = "lesion_detection"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            evidence = await self.evidence_service.analyze(prepared, str(image.id), str(session.id), image.eye.value)
            lesions = self._lesion_payload(evidence)
            run.lesions = lesions
            evidence_versions = {
                name: {
                    "implementation": module.get("implementation"),
                    "model_version": (module.get("metadata") or {}).get("model_version"),
                    "status": module.get("status"),
                }
                for name, module in evidence.modules.items()
            }
            run.model_versions = {**(run.model_versions or {}), "retinal_evidence": evidence_versions}
            await self._set_stage(db, run, actor_id, "retinal_structure_analysis", "COMPLETED", {"module_count": len(evidence.modules)})
            await self._set_stage(db, run, actor_id, "lesion_detection", "COMPLETED", {"supported_module_count": sum(1 for module in evidence.modules.values() if module.get("supported"))})
            for timing_name, duration_ms in (evidence.stage_timings_ms or {}).items():
                stage_name = {"vessel_inference_ms": "lesion_detection", "structure_analysis_ms": "retinal_structure_analysis", "lesion_inference_ms": "lesion_detection"}.get(timing_name)
                if stage_name:
                    run.stage_metrics = {**(run.stage_metrics or {}), stage_name: {**((run.stage_metrics or {}).get(stage_name) or {}), timing_name: duration_ms}}

            current_stage = "grad_cam"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            current_stage = "attention_lesion_agreement"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            current_stage = "uncertainty"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            explanation = await self.explainability_service.analyze(prepared, str(image.id), str(session.id), evidence, run_stability=run_stability, run_counterfactual=run_counterfactual)
            run.explainability = explanation.to_dict()
            run.model_versions = {**(run.model_versions or {}), "explainability": explanation.model_version}
            await self._set_stage(db, run, actor_id, "grad_cam", "COMPLETED", {"target_class": explanation.predicted_class, "model_version": explanation.model_version})
            await self._set_stage(db, run, actor_id, "attention_lesion_agreement", "COMPLETED", {"status": explanation.attention_lesion_agreement.get("status"), "score": explanation.attention_lesion_agreement.get("score")})
            await self._set_stage(db, run, actor_id, "uncertainty", "COMPLETED", {"source": "classifier_probability_distribution"})

            current_stage = "model_disagreement"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            additional = model_predictions or []
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"additional_model_count": len(additional)})

            current_stage = "retinaguard"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            final_quality = run.quality["final"]
            inputs = RetinaGuardInputs(
                quality_score=final_quality.get("quality_score"), raw_confidence=prediction.raw_confidence,
                probabilities=prediction.probabilities, classifier_logits=prediction.severity_logits,
                model_predictions=additional, lesion_evidence_strength=derive_lesion_evidence_strength(evidence), vessel_evidence_status=derive_vessel_evidence_status(evidence),
                attention_lesion_agreement=explanation.attention_lesion_agreement,
                explanation_stability=explanation.explanation_stability,
                quality_feature_vector={key: float(value) for key, value in final_quality.get("feature_vector", {}).items() if isinstance(value, (int, float))},
                predicted_grade=prediction.predicted_grade, predicted_grade_label=prediction.predicted_grade_label,
                referable_dr=prediction.referable_dr, model_version=prediction.model_version,
            )
            guard = await self.retinaguard.evaluate_async(inputs, prepared, self.classifier)
            run.retinaguard = guard.to_dict()
            run.model_versions = {**(run.model_versions or {}), "retinaguard": guard.configuration.get("version"), "confidence_calibration": guard.configuration.get("calibration_version")}
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"category": guard.trust_category, "score": guard.trust_score, "version": guard.configuration.get("version")})

            current_stage = "triage"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            triage = self._triage_payload(prediction, guard)
            run.triage = triage
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"recommendation": triage["recommendation"], "priority": triage["priority"]})
            run.status = "COMPLETED"
            run.completed_at = datetime.now(timezone.utc)
            session.status = ScreeningStatus.COMPLETE if guard.trust_category == "TRUSTED" else ScreeningStatus.NEEDS_REVIEW
            session.completed_at = session.completed_at or run.completed_at
            await self._audit(db, actor_id, "screening.run.completed", run.id, {
                "final_decision": guard.trust_category,
                "trust_category": guard.trust_category,
                "trust_score": guard.trust_score,
                "triage": triage["recommendation"],
                "model_versions": run.model_versions,
                "preprocessing_version": run.model_versions.get("preprocessing"),
                "trust_engine_version": run.model_versions.get("retinaguard"),
            })
            await db.commit()
            return self._output(run, prediction=prediction, evidence=evidence, explanation=explanation, guard=guard)
        except Exception as exc:
            logger.exception("screening.run.failed", extra={"screening_id": str(run.id), "stage": current_stage})
            run.status = "FAILED"
            run.error = {
                "stage": current_stage,
                "type": type(exc).__name__,
                "message": safe_error_message(exc, f"The {current_stage.replace('_', ' ')} stage failed; no result was produced."),
            }
            run.stage_errors = {**(run.stage_errors or {}), current_stage: run.error}
            run.stage_status = {**(run.stage_status or {}), current_stage: "FAILED"}
            self._record_stage_metric(run, current_stage, "FAILED")
            run.completed_at = datetime.now(timezone.utc)
            session.status = ScreeningStatus.FAILED
            session.completed_at = run.completed_at
            await self._audit(db, actor_id, "screening.run.failed", run.id, {
                **run.error,
                "model_versions": run.model_versions or {},
                "preprocessing_version": (run.model_versions or {}).get("preprocessing"),
                "trust_engine_version": (run.model_versions or {}).get("retinaguard"),
            })
            await db.commit()
            return self._output(run)

    async def _finish_primary(
        self,
        db: AsyncSession,
        run: ScreeningRun,
        session: ScreeningSession,
        image: FundusImage,
        actor_id: UUID | None,
        prepared: bytes,
        prediction: DRPrediction,
        model_predictions: list[dict[str, Any]],
    ) -> ScreeningPipelineOutput:
        """Complete the mandatory result and queue enrichment separately."""
        current_stage = "uncertainty"
        try:
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"source": "classifier_probability_distribution"})

            current_stage = "model_disagreement"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {"additional_model_count": len(model_predictions)})

            current_stage = "retinaguard"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            final_quality = run.quality["final"]
            inputs = RetinaGuardInputs(
                quality_score=final_quality.get("quality_score"),
                raw_confidence=prediction.raw_confidence,
                probabilities=prediction.probabilities,
                classifier_logits=prediction.severity_logits,
                model_predictions=model_predictions,
                # Missing optional evidence is represented as None. RetinaGuard
                # applies its configured missing-signal policy and records this
                # limitation; it is never converted into negative evidence.
                lesion_evidence_strength=None,
                vessel_evidence_status=None,
                attention_lesion_agreement=None,
                explanation_stability=None,
                quality_feature_vector={
                    key: float(value)
                    for key, value in final_quality.get("feature_vector", {}).items()
                    if isinstance(value, (int, float))
                },
                predicted_grade=prediction.predicted_grade,
                predicted_grade_label=prediction.predicted_grade_label,
                referable_dr=prediction.referable_dr,
                model_version=prediction.model_version,
            )
            guard = await self.retinaguard.evaluate_async(inputs, prepared, self.classifier)
            run.retinaguard = guard.to_dict()
            run.model_versions = {
                **(run.model_versions or {}),
                "retinaguard": guard.configuration.get("version"),
                "confidence_calibration": guard.configuration.get("calibration_version"),
                "retinal_evidence": {"status": "QUEUED", "runtime_budget_seconds": self.optional_evidence_timeout_seconds},
                "explainability": {"status": "QUEUED", "runtime_budget_seconds": self.optional_explainability_timeout_seconds},
            }
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {
                "category": guard.trust_category,
                "score": guard.trust_score,
                "version": guard.configuration.get("version"),
                "optional_evidence_pending": True,
            })

            current_stage = "triage"
            await self._set_stage(db, run, actor_id, current_stage, "PROCESSING")
            triage = self._triage_payload(prediction, guard)
            run.triage = triage
            await self._set_stage(db, run, actor_id, current_stage, "COMPLETED", {
                "recommendation": triage["recommendation"],
                "priority": triage["priority"],
            })

            for stage in ("retinal_structure_analysis", "lesion_detection"):
                await self._set_stage(db, run, actor_id, stage, "QUEUED", {
                    "optional": True,
                    "runtime_budget_seconds": self.optional_evidence_timeout_seconds,
                    "reason": "Primary screening completed; optional evidence runs in the background.",
                })
            for stage in ("grad_cam", "attention_lesion_agreement"):
                await self._set_stage(db, run, actor_id, stage, "QUEUED", {
                    "optional": True,
                    "runtime_budget_seconds": self.optional_explainability_timeout_seconds,
                    "reason": "Primary screening completed; optional explainability runs in the background.",
                })

            run.status = "COMPLETED"
            run.completed_at = datetime.now(timezone.utc)
            session.status = ScreeningStatus.COMPLETE if guard.trust_category == "TRUSTED" else ScreeningStatus.NEEDS_REVIEW
            session.completed_at = session.completed_at or run.completed_at
            run.stage_metrics = {
                **(run.stage_metrics or {}),
                "primary_screening": {
                    "status": "COMPLETED",
                    "completed_at": run.completed_at.isoformat(),
                    "duration_ms": self._elapsed_ms(run.started_at, run.completed_at),
                },
            }
            await self._audit(db, actor_id, "screening.primary.completed", run.id, {
                "final_decision": guard.trust_category,
                "trust_score": guard.trust_score,
                "triage": triage["recommendation"],
                "model_versions": run.model_versions,
                "preprocessing_version": run.model_versions.get("preprocessing"),
                "trust_engine_version": run.model_versions.get("retinaguard"),
                "optional_evidence_status": "QUEUED",
            })
            await db.commit()
            return self._output(run, prediction=prediction, guard=guard)
        except Exception as exc:
            logger.exception("screening.primary.failed", extra={"screening_id": str(run.id), "stage": current_stage})
            run.status = "FAILED"
            run.error = {
                "stage": current_stage,
                "type": type(exc).__name__,
                "message": safe_error_message(exc, f"The {current_stage.replace('_', ' ')} stage failed; no result was produced."),
            }
            run.stage_errors = {**(run.stage_errors or {}), current_stage: run.error}
            run.stage_status = {**(run.stage_status or {}), current_stage: "FAILED"}
            self._record_stage_metric(run, current_stage, "FAILED")
            run.completed_at = datetime.now(timezone.utc)
            session.status = ScreeningStatus.FAILED
            session.completed_at = run.completed_at
            await self._audit(db, actor_id, "screening.primary.failed", run.id, {
                **run.error,
                "model_versions": run.model_versions or {},
            })
            await db.commit()
            return self._output(run)

    async def _execute_optional(
        self,
        db: AsyncSession,
        run: ScreeningRun,
        session: ScreeningSession,
        image: FundusImage,
        actor_id: UUID | None,
        prepared: bytes,
        run_stability: bool | None,
        run_counterfactual: bool | None,
    ) -> None:
        """Run evidence and explainability with honest, independent budgets."""
        evidence: RetinalEvidenceAnalysis | None = None
        evidence_task: asyncio.Task[Any] | None = None
        try:
            for stage in ("retinal_structure_analysis", "lesion_detection"):
                await self._set_stage(db, run, actor_id, stage, "PROCESSING", {
                    "optional": True,
                    "runtime_budget_seconds": self.optional_evidence_timeout_seconds,
                })
            evidence_task = asyncio.create_task(self.evidence_service.analyze(
                prepared, str(image.id), str(session.id), image.eye.value,
            ))
            evidence = await self._await_optional(
                "retinal_evidence", evidence_task, self.optional_evidence_timeout_seconds,
            )
        except OptionalStageTimeout as timeout:
            await self._mark_optional_timeout(db, run, actor_id, ("retinal_structure_analysis", "lesion_detection"), timeout)
            await self._drain_optional_task(timeout.task, run.id, timeout.stage)
            return
        except Exception as exc:
            await self._mark_optional_failure(db, run, actor_id, ("retinal_structure_analysis", "lesion_detection"), exc)
            return

        assert evidence is not None
        run.lesions = self._lesion_payload(evidence)
        evidence_versions = {
            name: {
                "implementation": module.get("implementation"),
                "model_version": (module.get("metadata") or {}).get("model_version"),
                "status": module.get("status"),
            }
            for name, module in evidence.modules.items()
        }
        run.model_versions = {**(run.model_versions or {}), "retinal_evidence": {"status": "AVAILABLE", "modules": evidence_versions}}
        for timing_name, duration_ms in (evidence.stage_timings_ms or {}).items():
            stage_name = {
                "vessel_inference_ms": "retinal_structure_analysis",
                "structure_analysis_ms": "retinal_structure_analysis",
                "lesion_inference_ms": "lesion_detection",
            }.get(timing_name)
            if stage_name:
                run.stage_metrics = {
                    **(run.stage_metrics or {}),
                    stage_name: {**((run.stage_metrics or {}).get(stage_name) or {}), timing_name: duration_ms},
                }
        await self._set_stage(db, run, actor_id, "retinal_structure_analysis", "COMPLETED", {"module_count": len(evidence.modules)})
        await self._set_stage(db, run, actor_id, "lesion_detection", "COMPLETED", {
            "supported_module_count": sum(1 for module in evidence.modules.values() if module.get("supported")),
        })
        await persist_evidence_analysis(run.lesions, image, session, db, update_session_status=False)

        explanation_task: asyncio.Task[Any] | None = None
        try:
            for stage in ("grad_cam", "attention_lesion_agreement"):
                await self._set_stage(db, run, actor_id, stage, "PROCESSING", {
                    "optional": True,
                    "runtime_budget_seconds": self.optional_explainability_timeout_seconds,
                })
            explanation_task = asyncio.create_task(self.explainability_service.analyze(
                prepared,
                str(image.id),
                str(session.id),
                evidence,
                run_stability=run_stability,
                run_counterfactual=run_counterfactual,
            ))
            explanation = await self._await_optional(
                "explainability", explanation_task, self.optional_explainability_timeout_seconds,
            )
        except OptionalStageTimeout as timeout:
            run.explainability = self._optional_unavailable_payload("TIMED_OUT", timeout)
            run.model_versions = {**(run.model_versions or {}), "explainability": {"status": "TIMED_OUT", "runtime_budget_seconds": timeout.budget_seconds}}
            await self._mark_optional_timeout(db, run, actor_id, ("grad_cam", "attention_lesion_agreement"), timeout)
            await self._drain_optional_task(timeout.task, run.id, timeout.stage)
            return
        except Exception as exc:
            run.explainability = self._optional_failure_payload("UNAVAILABLE", "explainability", exc)
            run.model_versions = {**(run.model_versions or {}), "explainability": {"status": "UNAVAILABLE"}}
            await self._mark_optional_failure(db, run, actor_id, ("grad_cam", "attention_lesion_agreement"), exc)
            return

        run.explainability = explanation.to_dict()
        run.model_versions = {**(run.model_versions or {}), "explainability": {"status": "AVAILABLE", "model_version": explanation.model_version}}
        await self._set_stage(db, run, actor_id, "grad_cam", "COMPLETED", {
            "target_class": explanation.predicted_class,
            "model_version": explanation.model_version,
        })
        await self._set_stage(db, run, actor_id, "attention_lesion_agreement", "COMPLETED", {
            "status": explanation.attention_lesion_agreement.get("status"),
            "score": explanation.attention_lesion_agreement.get("score"),
        })
        await persist_explainability(run.explainability, image, session, db, update_session_status=False)
        await self._audit(db, actor_id, "screening.optional.completed", run.id, {
            "evidence_status": "AVAILABLE",
            "explainability_status": "AVAILABLE",
            "model_versions": run.model_versions,
        })
        await db.commit()

    async def _await_optional(self, stage: str, task: asyncio.Task[Any], budget_seconds: int) -> Any:
        try:
            # Shield the task so a timeout does not cancel a CPU worker spawned
            # by asyncio.to_thread. The durable status is updated immediately;
            # the worker is drained in the background for clean resource use.
            return await asyncio.wait_for(asyncio.shield(task), timeout=budget_seconds)
        except asyncio.TimeoutError as exc:
            raise OptionalStageTimeout(stage, budget_seconds, task) from exc

    async def _drain_optional_task(self, task: asyncio.Task[Any], screening_id: UUID, stage: str) -> None:
        try:
            await task
        except asyncio.CancelledError:
            logger.warning("screening.optional.task_cancelled", extra={"event": "screening.optional.task_cancelled", "screening_id": str(screening_id), "stage": stage})
        except Exception:
            logger.exception("screening.optional.task_failed_after_timeout", extra={"event": "screening.optional.task_failed_after_timeout", "screening_id": str(screening_id), "stage": stage})

    async def _mark_optional_timeout(self, db: AsyncSession, run: ScreeningRun, actor_id: UUID | None, stages: tuple[str, ...], timeout: OptionalStageTimeout) -> None:
        message = f"Optional stage exceeded the configured {timeout.budget_seconds}-second runtime budget; no evidence output was returned."
        details = {
            "status": "TIMED_OUT",
            "type": "OptionalStageTimeout",
            "message": message,
            "runtime_budget_seconds": timeout.budget_seconds,
            "stage": timeout.stage,
            "evidence_is_not_negative": True,
        }
        for stage in stages:
            await self._set_stage(db, run, actor_id, stage, "TIMED_OUT", details)
        run.stage_errors = {**(run.stage_errors or {}), timeout.stage: details}
        if timeout.stage == "retinal_evidence":
            run.lesions = self._optional_unavailable_payload("TIMED_OUT", timeout)
            run.model_versions = {**(run.model_versions or {}), "retinal_evidence": {"status": "TIMED_OUT", "runtime_budget_seconds": timeout.budget_seconds}}
        await self._audit(db, actor_id, "screening.optional.timed_out", run.id, details)
        await db.commit()

    async def _mark_optional_failure(self, db: AsyncSession, run: ScreeningRun, actor_id: UUID | None, stages: tuple[str, ...], exc: Exception) -> None:
        message = safe_error_message(exc, "Optional evidence was unavailable; no evidence output was substituted.")
        details = {"status": "UNAVAILABLE", "type": type(exc).__name__, "message": message, "evidence_is_not_negative": True}
        for stage in stages:
            await self._set_stage(db, run, actor_id, stage, "UNAVAILABLE", details)
        run.stage_errors = {**(run.stage_errors or {}), "optional_evidence": details}
        if any(stage in {"retinal_structure_analysis", "lesion_detection"} for stage in stages):
            run.lesions = {"status": "UNAVAILABLE", "modules": {}, "evidence_map_data_uri": None, "note": message, "provenance": details}
            run.model_versions = {**(run.model_versions or {}), "retinal_evidence": {"status": "UNAVAILABLE"}}
        await self._audit(db, actor_id, "screening.optional.unavailable", run.id, details)
        await db.commit()

    @staticmethod
    def _optional_unavailable_payload(status: str, timeout: OptionalStageTimeout) -> dict[str, Any]:
        return {
            "status": status,
            "grad_cam": {},
            "attention_lesion_agreement": {"status": "UNAVAILABLE", "score": None, "metrics": {}, "note": "No explainability result was returned within the optional runtime budget."},
            "explanation_stability": {"status": "NOT_RUN", "reason": "Optional explainability did not complete."},
            "counterfactual": {"status": "NOT_RUN", "reason": "Optional explainability did not complete."},
            "note": "Optional explainability timed out; this is not evidence against the classification.",
            "provenance": {"status": status, "stage": timeout.stage, "runtime_budget_seconds": timeout.budget_seconds},
        }

    @staticmethod
    def _optional_failure_payload(status: str, stage: str, exc: Exception) -> dict[str, Any]:
        return {
            "status": status,
            "grad_cam": {},
            "attention_lesion_agreement": {"status": "UNAVAILABLE", "score": None, "metrics": {}},
            "explanation_stability": {"status": "NOT_RUN", "reason": "Optional explainability did not complete."},
            "counterfactual": {"status": "NOT_RUN", "reason": "Optional explainability did not complete."},
            "note": safe_error_message(exc, "Optional explainability was unavailable; no output was substituted."),
            "provenance": {"status": status, "stage": stage},
        }

    async def _assess_quality(self, db: AsyncSession, image: FundusImage, content: bytes) -> tuple[TrustGateOutcome, bytes]:
        initial = await self.quality_service.assess(content)
        final = initial
        prepared = content
        enhanced = False
        passes = 0
        if initial.quality_decision == TrustGateDecision.BORDERLINE:
            prepared = self.quality_service.enhance(content)
            final = await self.quality_service.assess(prepared)
            enhanced = True
            passes = 1
            if final.quality_decision == TrustGateDecision.BORDERLINE:
                final.recommended_action = "One controlled enhancement pass was used; recapture is recommended."
                final.next_action = "RECAPTURE_IMAGE"
        if enhanced:
            key = f"fundus/{image.patient_id}/{image.id}/enhanced-pass-{passes}.png"
            image.enhanced_storage_path = await self.storage.save(key, prepared, "image/png")
        image.quality_score = final.quality_score
        image.quality_decision = QualityDecision(final.quality_decision.lower())
        image.enhancement_passes = passes
        image.quality_checked_at = datetime.now(timezone.utc)
        image.quality_assessment = {"initial": initial.to_dict(), "final": final.to_dict(), "enhancement_applied": enhanced, "enhancement_passes": passes}
        await db.commit()
        return TrustGateOutcome(initial, final, enhanced, passes), prepared

    async def _finish_quality_block(self, db: AsyncSession, run: ScreeningRun, session: ScreeningSession, image: FundusImage, actor_id: UUID | None, outcome: TrustGateOutcome) -> ScreeningPipelineOutput:
        run.triage = {"status": "blocked_before_clinical_ai", "recommendation": "RECAPTURE_IMAGE", "priority": "high", "reasons": [issue.message for issue in outcome.final.issues] or [outcome.final.recommended_action], "recommended_action": outcome.final.recommended_action, "clinical_ai_started": False, "note": "The Image Trust Gate blocked downstream clinical AI for this run."}
        run.model_versions = {"preprocessing": "image-trust-gate-v1"}
        for stage in RUN_STAGES[2:-1]:
            await self._set_stage(db, run, actor_id, stage, "SKIPPED", {"reason": "Image Trust Gate did not mark the image GRADABLE."})
        await self._set_stage(db, run, actor_id, "triage", "COMPLETED", {"recommendation": "RECAPTURE_IMAGE", "priority": "high"})
        run.status = "COMPLETED"
        run.completed_at = datetime.now(timezone.utc)
        session.status = ScreeningStatus.NEEDS_REVIEW
        session.completed_at = run.completed_at
        await self._audit(db, actor_id, "screening.run.completed_quality_block", run.id, {
            "decision": outcome.final.quality_decision,
            "final_decision": "UNRELIABLE",
            "recommendation": "RECAPTURE_IMAGE",
            "model_versions": run.model_versions,
            "preprocessing_version": run.model_versions.get("preprocessing"),
            "trust_engine_version": None,
        })
        await db.commit()
        return self._output(run)

    async def _set_stage(self, db: AsyncSession, run: ScreeningRun, actor_id: UUID | None, stage: str, status: str, details: dict[str, Any] | None = None) -> None:
        run.stage_status = {**(run.stage_status or {}), stage: status}
        self._record_stage_metric(run, stage, status)
        if status in {"FAILED", "TIMED_OUT", "UNAVAILABLE"}:
            run.stage_errors = {**(run.stage_errors or {}), stage: details or {"message": "Stage failed"}}
        if details and details.get("optional"):
            run.stage_metrics = {
                **(run.stage_metrics or {}),
                stage: {
                    **((run.stage_metrics or {}).get(stage) or {}),
                    "optional": True,
                    "runtime_budget_seconds": details.get("runtime_budget_seconds"),
                },
            }
        if status in {"COMPLETED", "FAILED", "SKIPPED", "TIMED_OUT", "UNAVAILABLE"}:
            logger.info("screening.stage", extra={
                "event": "screening.stage",
                "screening_id": str(run.id),
                "stage": stage,
                "status": status,
                "status_code": 504 if status == "TIMED_OUT" else (500 if status in {"FAILED", "UNAVAILABLE"} else 200),
                "duration_ms": (run.stage_metrics or {}).get(stage, {}).get("duration_ms"),
            })
        await self._audit(db, actor_id, f"screening.stage.{status.lower()}", run.id, {"stage": stage, **(details or {})})
        await db.commit()

    @staticmethod
    def _record_stage_metric(run: ScreeningRun, stage: str, status: str) -> None:
        now = datetime.now(timezone.utc)
        metrics = {**(run.stage_metrics or {})}
        entry = {**(metrics.get(stage) or {})}
        if status == "PROCESSING":
            entry["started_at"] = now.isoformat()
        elif status in {"COMPLETED", "FAILED", "SKIPPED", "TIMED_OUT", "UNAVAILABLE"}:
            entry["completed_at"] = now.isoformat()
            started_at = entry.get("started_at")
            if started_at:
                try:
                    entry["duration_ms"] = round(max(0.0, (now - datetime.fromisoformat(started_at)).total_seconds() * 1000), 3)
                except ValueError:
                    entry["duration_ms"] = None
            elif status == "SKIPPED":
                entry["duration_ms"] = 0.0
        metrics[stage] = entry
        run.stage_metrics = metrics

    @staticmethod
    async def _audit(db: AsyncSession, actor_id: UUID | None, action: str, resource_id: UUID, details: dict[str, Any]) -> None:
        db.add(AuditLog(actor_id=actor_id, action=action, resource_type="screening_run", resource_id=str(resource_id), details={"timestamp": datetime.now(timezone.utc).isoformat(), **details}))

    @staticmethod
    def _classification_payload(prediction: DRPrediction) -> dict[str, Any]:
        return {"predicted_grade": prediction.predicted_grade, "predicted_grade_label": prediction.predicted_grade_label, "probabilities": prediction.probabilities, "referable_dr": prediction.referable_dr, "referable_probability": prediction.referable_probability, "raw_confidence": prediction.raw_confidence, "model_name": prediction.model_name, "model_version": prediction.model_version, "backbone": prediction.backbone, "referable_mapping": prediction.referable_mapping, "hierarchical_probabilities": prediction.hierarchical_probabilities, "ordinal_mode": prediction.ordinal_mode}

    @staticmethod
    def _lesion_payload(evidence: RetinalEvidenceAnalysis) -> dict[str, Any]:
        modules = {name: module for name, module in evidence.modules.items() if module.get("category") == "lesion_detection" or name in {"exudate_segmentation", "vessel_segmentation"}}
        return {
            "status": evidence.status,
            "modules": modules,
            "evidence_map_data_uri": evidence.evidence_map_data_uri,
            "coarse_to_fine": evidence.coarse_to_fine,
            "dataset_support": evidence.dataset_support,
            "stage_timings_ms": evidence.stage_timings_ms,
            "note": "Vessel and lesion modules are supporting evidence and do not replace the DR classifier.",
        }

    @staticmethod
    def _triage_payload(prediction: DRPrediction, guard: RetinaGuardResult) -> dict[str, Any]:
        if guard.trust_category in {"UNRELIABLE", "INSUFFICIENT_EVIDENCE"}:
            recommendation = "RECAPTURE_OR_SPECIALIST_REVIEW"
            priority = "high"
        elif guard.trust_category in {"REVIEW_RECOMMENDED", "UNCERTAIN"}:
            recommendation = "HUMAN_REVIEW_REQUIRED"
            priority = "high"
        elif prediction.referable_dr:
            recommendation = "SPECIALIST_REVIEW_RECOMMENDED"
            priority = "high"
        else:
            recommendation = "AI_TRIAGE_MAY_PROCEED"
            priority = "routine"
        return {"status": "completed", "recommendation": recommendation, "priority": priority, "reasons": guard.reason_summary, "referable_dr_signal": prediction.referable_dr, "note": "Workflow triage recommendation only; not a diagnosis or final clinical decision."}

    @staticmethod
    def _output(run: ScreeningRun, prediction: DRPrediction | None = None, evidence: RetinalEvidenceAnalysis | None = None, explanation: ExplainabilityAnalysis | None = None, guard: RetinaGuardResult | None = None) -> ScreeningPipelineOutput:
        return ScreeningPipelineOutput(run.id, run.status, run.quality, run.classification, run.lesions, run.explainability, run.retinaguard, run.triage, run.model_versions or {}, run.stage_status or {}, run.stage_metrics or {}, run.stage_errors or {}, run.error, prediction, evidence, explanation, guard)

    @staticmethod
    def _elapsed_ms(started_at: datetime | None, completed_at: datetime | None) -> float | None:
        if not started_at or not completed_at:
            return None
        return round(max(0.0, (completed_at - started_at).total_seconds() * 1000), 3)
