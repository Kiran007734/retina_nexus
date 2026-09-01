"""Measure real local screening-stage latency without writing clinical records."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.routes.reports import _pdf_bytes  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.ml.evidence.lesion_model import MODEL_CLASS_TO_MODULE, PretrainedRetinalLesionAdapter  # noqa: E402
from app.ml.evidence.service import RetinalEvidenceService  # noqa: E402
from app.ml.evidence.vessel_model import PretrainedRetinalVesselAdapter  # noqa: E402
from app.ml.explainability.service import ExplainabilityService  # noqa: E402
from app.ml.inference.classifier import TorchDRClassificationService  # noqa: E402
from app.ml.models.classifier import ReferableDRMapping  # noqa: E402
from app.ml.quality.trust_gate import ImageTrustGateService, TrustGateDecision  # noqa: E402
from app.ml.trust.calibration import TemperatureScaler  # noqa: E402
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs, derive_lesion_evidence_strength, derive_vessel_evidence_status  # noqa: E402
from app.ml.trust.ood import FeatureDistributionMonitor  # noqa: E402
from app.ml.trust.uncertainty import UncertaintyEstimator  # noqa: E402
from app.schemas.reports import ReportPayload  # noqa: E402
from app.services.runtime import resolve_path  # noqa: E402


def _settings() -> Settings:
    root_env = ROOT / ".env"
    backend_env = ROOT / "backend" / ".env"
    env_file = backend_env if backend_env.is_file() else root_env if root_env.is_file() else None
    return Settings(_env_file=str(env_file)) if env_file else Settings(_env_file=None)


def _image_path(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError("The requested benchmark image was not found")
        return candidate.resolve()
    candidates = sorted((ROOT / "ml" / "datasets" / "raw").rglob("*.png")) + sorted((ROOT / "ml" / "datasets" / "raw").rglob("*.jpeg")) + sorted((ROOT / "ml" / "datasets" / "raw").rglob("*.jpg"))
    if not candidates:
        raise FileNotFoundError("No authorized local JPEG/PNG image was found under ml/datasets/raw")
    return candidates[0].resolve()


def _services(settings: Settings):
    classifier = TorchDRClassificationService(
        model_path=str(resolve_path(settings.classifier_model_path)) if settings.classifier_model_path else None, backbone=settings.classifier_backbone,
        model_version=settings.classifier_model_version, device=settings.classifier_device,
        referable_mapping=ReferableDRMapping(name=f"grade_{settings.referable_min_grade}_or_worse", referable_grades=tuple(range(settings.referable_min_grade, 5))),
    )
    lesion = PretrainedRetinalLesionAdapter(model_path=str(resolve_path(settings.lesion_model_path)) if settings.lesion_model_path else None, device=settings.lesion_model_device, threshold=settings.lesion_model_threshold, version=settings.lesion_model_version)
    vessel = PretrainedRetinalVesselAdapter(model_path=str(resolve_path(settings.vessel_model_path)) if settings.vessel_model_path else None, device=settings.vessel_model_device, threshold=settings.vessel_model_threshold, version=settings.vessel_model_version)
    evidence = RetinalEvidenceService(max_dimension=settings.evidence_max_dimension, enable_heuristics=settings.evidence_enable_heuristics, model_adapters={"vessel_segmentation": vessel, **({module: lesion for module in MODEL_CLASS_TO_MODULE.values()} if lesion.is_configured else {})}, enable_vessel_baseline=settings.evidence_enable_vessel_baseline)
    explanation = ExplainabilityService(classifier=classifier, stability_enabled=False, counterfactual_enabled=False)
    guard = RetinaGuardEngine(
        version=settings.retinaguard_config_version,
        calibrator=TemperatureScaler(temperature=settings.retinaguard_temperature, version=settings.retinaguard_calibration_version, fitted=settings.retinaguard_calibration_fitted),
        uncertainty_estimator=UncertaintyEstimator(),
        ood_monitor=FeatureDistributionMonitor(settings.retinaguard_ood_reference_path, settings.retinaguard_ood_threshold),
        weights={"quality": settings.retinaguard_weight_quality, "calibrated_confidence": settings.retinaguard_weight_calibrated_confidence, "uncertainty": settings.retinaguard_weight_uncertainty, "model_agreement": settings.retinaguard_weight_model_agreement, "lesion_evidence": settings.retinaguard_weight_lesion_evidence, "attention_lesion_agreement": settings.retinaguard_weight_attention_lesion_agreement, "explanation_stability": settings.retinaguard_weight_explanation_stability, "ood": settings.retinaguard_weight_ood},
        missing_signal_score=settings.retinaguard_missing_signal_score, trusted_threshold=settings.retinaguard_trusted_threshold, unreliable_threshold=settings.retinaguard_unreliable_threshold,
    )
    return classifier, lesion, vessel, evidence, explanation, guard


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"sample_count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None, "stddev_ms": None}
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {"sample_count": len(values), "mean_ms": round(statistics.mean(values), 3), "median_ms": round(statistics.median(values), 3), "p95_ms": round(p95, 3), "min_ms": round(min(values), 3), "max_ms": round(max(values), 3), "stddev_ms": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0}


def _timed(function):
    started = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - started) * 1000


async def _run_once(content: bytes, image_name: str, services) -> dict:
    classifier, lesion, vessel, evidence_service, explainability, guard = services
    quality_service = ImageTrustGateService()
    row: dict = {"status": "FAILED", "stage_timings_ms": {}, "errors": {}}
    try:
        metadata, duration = _timed(lambda: quality_service.validate_input(content))
        row["stage_timings_ms"]["image_validation"] = duration
        initial, duration = await _async_timed(quality_service.assess(content))
        row["stage_timings_ms"]["quality_assessment"] = duration
        final, prepared = initial, content
        if initial.quality_decision == TrustGateDecision.BORDERLINE:
            enhanced, enhance_ms = _timed(lambda: quality_service.enhance(content))
            recheck, recheck_ms = await _async_timed(quality_service.assess(enhanced))
            prepared, final = enhanced, recheck
            row["stage_timings_ms"]["quality_enhancement"] = enhance_ms + recheck_ms
        row["quality"] = {"initial_decision": initial.quality_decision, "final_decision": final.quality_decision, "initial_score": initial.quality_score, "final_score": final.quality_score, "enhancement_applied": prepared is not content, "dimensions": [metadata.width, metadata.height]}
        if final.quality_decision != TrustGateDecision.GRADABLE:
            row["status"] = "QUALITY_BLOCKED"
            return row

        prediction, duration = await _async_timed(classifier.classify(prepared))
        row["stage_timings_ms"]["classification"] = duration
        evidence, duration = await _async_timed(evidence_service.analyze(prepared, "benchmark-image", "benchmark-session", "right"))
        row["stage_timings_ms"]["evidence_total"] = duration
        row["stage_timings_ms"].update(evidence.stage_timings_ms or {})
        explanation, duration = await _async_timed(explainability.analyze(prepared, "benchmark-image", "benchmark-session", evidence))
        row["stage_timings_ms"]["grad_cam_and_agreement"] = duration
        final_quality = final
        feature_vector = final_quality.feature_vector
        inputs = RetinaGuardInputs(
            quality_score=final_quality.quality_score, raw_confidence=prediction.raw_confidence, probabilities=prediction.probabilities, classifier_logits=prediction.severity_logits,
            lesion_evidence_strength=derive_lesion_evidence_strength(evidence), vessel_evidence_status=derive_vessel_evidence_status(evidence), attention_lesion_agreement=explanation.attention_lesion_agreement, explanation_stability=explanation.explanation_stability,
            quality_feature_vector=feature_vector, predicted_grade=prediction.predicted_grade, predicted_grade_label=prediction.predicted_grade_label, referable_dr=prediction.referable_dr, model_version=prediction.model_version,
        )
        guard_result, duration = await _async_timed(guard.evaluate_async(inputs, prepared, classifier))
        row["stage_timings_ms"]["retinaguard"] = duration
        report = ReportPayload(
            screening_id=uuid4(), session_id=uuid4(), eye="right", generated_at=datetime.now(timezone.utc),
            image_quality={"decision": final.quality_decision, "score": final.quality_score},
            ai_assessment={"predicted_grade": prediction.predicted_grade, "predicted_grade_label": prediction.predicted_grade_label, "referable_dr": prediction.referable_dr, "confidence": prediction.raw_confidence},
            clinical_evidence={"summary": []}, explainability={}, retinaguard=guard_result.to_dict(), recommended_action="benchmark", disclaimer="Engineering benchmark only; not a clinical report.",
        )
        _, duration = _timed(lambda: _pdf_bytes(report))
        row["stage_timings_ms"]["pdf_generation"] = duration
        row["status"] = "COMPLETED"
        row["prediction"] = {"grade": prediction.predicted_grade, "referable_dr": prediction.referable_dr, "raw_confidence": prediction.raw_confidence, "model_version": prediction.model_version}
        row["retinaguard"] = {"trust_category": guard_result.trust_category, "trust_score": guard_result.trust_score}
        row["evidence_status"] = {name: module.get("status") for name, module in evidence.modules.items()}
    except Exception as exc:
        row["errors"]["pipeline"] = {"type": type(exc).__name__, "message": "A benchmark stage failed; no result was substituted."}
    return row


async def _async_timed(awaitable):
    started = time.perf_counter()
    result = await awaitable
    return result, (time.perf_counter() - started) * 1000


async def _main(args) -> int:
    settings = _settings()
    image = _image_path(args.image)
    content = image.read_bytes()
    services = _services(settings)
    classifier, lesion, vessel, evidence_service, _, _ = services
    load_timings: dict[str, float] = {}
    for name, loader in (("classifier", classifier.verify_loadable), ("lesion_segmentation", lesion.verify_loadable), ("vessel_segmentation", vessel.verify_loadable)):
        started = time.perf_counter()
        try:
            loader()
            load_timings[name] = (time.perf_counter() - started) * 1000
        except Exception:
            load_timings[name] = None

    rows = []
    for _ in range(max(1, args.runs)):
        rows.append(await _run_once(content, image.name, services))
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    stage_values: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.get("stage_timings_ms", {}).items():
            if isinstance(value, (int, float)):
                stage_values.setdefault(name, []).append(float(value))
    full_pipeline_values = [
        sum(float(value) for name, value in row.get("stage_timings_ms", {}).items() if name in {"image_validation", "quality_assessment", "quality_enhancement", "classification", "evidence_total", "grad_cam_and_agreement", "retinaguard", "pdf_generation"})
        for row in completed
    ]
    stage_latency = {name: _stats(values) for name, values in stage_values.items()}
    stage_latency["classification_only"] = _stats(stage_values.get("classification", []))
    stage_latency["full_pipeline"] = _stats(full_pipeline_values)
    evidence_modules = completed[-1].get("evidence_status", {}) if completed else {}
    output = {
        "benchmark_version": "deployment-benchmark-v1",
        "status": "COMPLETE" if rows and all(row["status"] in {"COMPLETED", "QUALITY_BLOCKED"} for row in rows) else "PARTIAL_FAILURE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image": {"name": image.name, "source": "authorized local image under ml/datasets/raw", "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)},
        "hardware": {"os": platform.platform(), "python": platform.python_version(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "torch": _version("torch"), "torchvision": _version("torchvision"), "cuda_available": _cuda_available()},
        "configuration": {"runs": args.runs, "warmup_policy": "Model load timings are recorded separately; stage samples are warm after the first load.", "classifier_device": settings.classifier_device, "evidence_max_dimension": settings.evidence_max_dimension, "clinical_validation_claim": False},
        "model_load_timings_ms": load_timings,
        "stage_latency": stage_latency,
        "completed_run_count": len(completed),
        "run_statuses": [row["status"] for row in rows],
        "evidence_module_status": evidence_modules,
        "runs": rows,
        "limitations": ["Local engineering latency only; no throughput or clinical performance claim.", "P95 is descriptive and based on the configured repetition count.", "Unavailable optional evidence is reported as unavailable and is not replaced by a heuristic model."],
    }
    output_path = ROOT / "ml" / "evaluation" / "deployment" / "performance_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    report_lines = [
        "# RETINA-NEXUS deployment performance benchmark", "", f"Generated: `{output['generated_at_utc']}`", "",
        "This is a local engineering benchmark using an authorized image. It is not a production-scale throughput claim, clinical validation, or safety guarantee.", "",
        "## Environment", "", f"- OS: `{output['hardware']['os']}`", f"- Python: `{output['hardware']['python']}`", f"- Processor: `{output['hardware']['processor'] or 'not reported'}`", f"- CPU count: `{output['hardware']['cpu_count']}`", f"- Torch: `{output['hardware']['torch']}`", f"- Torchvision: `{output['hardware']['torchvision']}`", f"- CUDA available: `{output['hardware']['cuda_available']}`", "",
        "## Method", "", f"- Warm repetitions: `{output['configuration']['runs']}`", "- Model loads are measured separately; stage samples run warm after the first load.", "- No database records, model artifacts, datasets, or evaluation reports were modified.", "",
        "## Stage latency", "", "| Stage | n | Mean ms | Median ms | P95 ms | Stddev ms |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, values in output["stage_latency"].items():
        report_lines.append(f"| {stage} | {values['sample_count']} | {values['mean_ms']} | {values['median_ms']} | {values['p95_ms']} | {values['stddev_ms']} |")
    report_lines.extend(["", "## Artifact/model status", "", f"- Run statuses: `{', '.join(output['run_statuses'])}`", f"- Completed runs: `{output['completed_run_count']}`", f"- Evidence module statuses: `{json.dumps(output['evidence_module_status'], sort_keys=True)}`", "", "## Limitations", "", *[f"- {item}" for item in output["limitations"]]])
    (output_path.parent / "performance_benchmark_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(output_path.relative_to(ROOT)), "completed_run_count": len(completed), "run_statuses": output["run_statuses"], "stage_latency": output["stage_latency"]}, indent=2))
    return 0 if output["status"] == "COMPLETE" and completed else 2


def _version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def _cuda_available() -> bool | None:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Authorized local JPEG/PNG; defaults to the first image under ml/datasets/raw")
    parser.add_argument("--runs", type=int, default=3, help="Warm stage repetitions (default: 3)")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
