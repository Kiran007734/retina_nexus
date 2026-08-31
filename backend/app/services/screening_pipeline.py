"""End-to-end screening orchestration for the current synchronous worker.

The runner is deliberately isolated from HTTP. It updates durable run state
between stages, so the same execute method can later be called by a queue
worker without changing the public contract.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.models.fundus_image import FundusImage, QualityDecision
from app.models.screening import ScreeningSession, ScreeningStatus
from app.models.screening_run import ScreeningRun
from app.ml.evidence.service import RetinalEvidenceAnalysis, RetinalEvidenceService
from app.ml.explainability.service import ExplainabilityAnalysis, ExplainabilityService
from app.ml.inference.classifier import ClassifierNotConfiguredError, DRPrediction, TorchDRClassificationService
from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService, TrustGateDecision, TrustGateOutcome
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs, RetinaGuardResult, derive_lesion_evidence_strength, derive_vessel_evidence_status

logger = logging.getLogger(__name__)

RUN_STAGES = (
    "image_validation", "quality_assessment", "dr_classification", "retinal_structure_analysis",
    "lesion_detection", "grad_cam", "attention_lesion_agreement", "uncertainty",
    "model_disagreement", "retinaguard", "triage",
)


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

    def __init__(self, quality_service: ImageTrustGateService, classifier: TorchDRClassificationService, evidence_service: RetinalEvidenceService, explainability_service: ExplainabilityService, retinaguard: RetinaGuardEngine, storage: Any):
        self.quality_service = quality_service
        self.classifier = classifier
        self.evidence_service = evidence_service
        self.explainability_service = explainability_service
        self.retinaguard = retinaguard
        self.storage = storage

    async def execute(self, db: AsyncSession, run: ScreeningRun, session: ScreeningSession, image: FundusImage, actor_id: UUID | None, run_stability: bool | None = None, run_counterfactual: bool | None = None, model_predictions: list[dict[str, Any]] | None = None) -> ScreeningPipelineOutput:
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
            run.error = {"stage": current_stage, "type": type(exc).__name__, "message": str(exc)}
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
        if status == "FAILED":
            run.stage_errors = {**(run.stage_errors or {}), stage: details or {"message": "Stage failed"}}
        await self._audit(db, actor_id, f"screening.stage.{status.lower()}", run.id, {"stage": stage, **(details or {})})
        await db.commit()

    @staticmethod
    def _record_stage_metric(run: ScreeningRun, stage: str, status: str) -> None:
        now = datetime.now(timezone.utc)
        metrics = {**(run.stage_metrics or {})}
        entry = {**(metrics.get(stage) or {})}
        if status == "PROCESSING":
            entry["started_at"] = now.isoformat()
        elif status in {"COMPLETED", "FAILED", "SKIPPED"}:
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
        return {"status": evidence.status, "modules": modules, "evidence_map_data_uri": evidence.evidence_map_data_uri, "note": "Vessel and lesion modules are supporting evidence and do not replace the DR classifier."}

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
