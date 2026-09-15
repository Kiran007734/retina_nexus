"""Read-only forensic diagnostic for one real retinal image.

This command deliberately does not call the HTTP API, write to the database,
write model artifacts, or persist screening results. It loads the same
configured services used by the backend and emits a JSON trace of the image
through preprocessing, classifier inference, evidence, explainability, and
RetinaGuard.

Usage from the repository root:

    python scripts/diagnose_inference.py path/to/image.png

Use ``--skip-evidence`` or ``--skip-explainability`` only when isolating a
slow optional stage. The default runs the full local diagnostic path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"


def _json_value(value: Any) -> Any:
    """Convert tensors, numpy scalars, and dataclasses to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _transform_metadata(transform: Any) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for item in getattr(transform, "transforms", []):
        record: dict[str, Any] = {"name": item.__class__.__name__}
        for attribute in (
            "size", "degrees", "translate", "scale", "brightness", "contrast",
            "saturation", "hue", "mean", "std", "p", "interpolation", "antialias",
        ):
            if hasattr(item, attribute):
                record[attribute] = _json_value(getattr(item, attribute))
        values.append(record)
    return {"name": transform.__class__.__name__, "transforms": values, "repr": repr(transform)}


def _image_metadata(image: Any, raw_bytes: bytes, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "byte_count": len(raw_bytes),
        "sha256": _sha256(raw_bytes),
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "bands": list(image.getbands()),
        "channels": len(image.getbands()),
        "exif_keys": [str(key) for key in image.getexif().keys()],
    }


def _tensor_metadata(tensor: Any, stage: str) -> dict[str, Any]:
    values = tensor.detach().cpu()
    return {
        "stage": stage,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "device": str(values.device),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "per_channel_mean": [float(item) for item in values[0].mean(dim=(1, 2)).tolist()],
        "per_channel_std": [float(item) for item in values[0].std(dim=(1, 2), unbiased=False).tolist()],
    }


def _module_summary(evidence: Any | None) -> dict[str, Any]:
    if evidence is None:
        return {"status": "NOT_RUN", "modules": {}}
    modules: dict[str, Any] = {}
    for name, module in evidence.modules.items():
        metadata = module.metadata if hasattr(module, "metadata") else module.get("metadata", {})
        modules[name] = {
            "status": module.status if hasattr(module, "status") else module.get("status"),
            "supported": module.supported if hasattr(module, "supported") else module.get("supported"),
            "implementation": module.implementation if hasattr(module, "implementation") else module.get("implementation"),
            "confidence": module.confidence if hasattr(module, "confidence") else module.get("confidence"),
            "count": module.count if hasattr(module, "count") else module.get("count"),
            "issues": module.issues if hasattr(module, "issues") else module.get("issues", []),
            "model_version": metadata.get("model_version"),
            "adapter_name": metadata.get("adapter_name"),
            "adapter_version": metadata.get("adapter_version"),
        }
    return {
        "status": evidence.status,
        "image_metadata": evidence.image_metadata,
        "module_count": len(modules),
        "modules": modules,
        "dataset_support": evidence.dataset_support,
        "stage_timings_ms": evidence.stage_timings_ms,
        "evidence_map_available": bool(evidence.evidence_map_data_uri),
    }


def _explainability_summary(explanation: Any | None) -> dict[str, Any]:
    if explanation is None:
        return {"status": "NOT_RUN"}
    grad_cam = explanation.grad_cam
    return {
        "status": "AVAILABLE",
        "predicted_class": explanation.predicted_class,
        "predicted_class_label": explanation.predicted_class_label,
        "model_version": explanation.model_version,
        "grad_cam": {
            "target_class": grad_cam.get("target_class"),
            "target_layer": grad_cam.get("target_layer"),
            "map_width": grad_cam.get("map_width"),
            "map_height": grad_cam.get("map_height"),
            "heatmap_available": bool(grad_cam.get("heatmap_data_uri")),
            "overlay_available": bool(grad_cam.get("overlay_data_uri")),
            "normalized_attention_map_available": bool(grad_cam.get("normalized_attention_map_data_uri")),
        },
        "attention_lesion_agreement": explanation.attention_lesion_agreement,
        "explanation_stability": explanation.explanation_stability,
        "counterfactual": explanation.counterfactual,
    }


def _prediction_payload(prediction: Any) -> dict[str, Any]:
    return {
        "predicted_grade": prediction.predicted_grade,
        "predicted_grade_label": prediction.predicted_grade_label,
        "probabilities": prediction.probabilities,
        "referable_dr": prediction.referable_dr,
        "referable_probability": prediction.referable_probability,
        "raw_confidence": prediction.raw_confidence,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
        "backbone": prediction.backbone,
        "referable_mapping": prediction.referable_mapping,
        "hierarchical_probabilities": prediction.hierarchical_probabilities,
        "ordinal_mode": prediction.ordinal_mode,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    # Resolve the image before changing cwd so relative paths work from root.
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BACKEND_ROOT))
    os.chdir(BACKEND_ROOT)

    from PIL import Image
    import torch
    from torchvision import transforms

    from app.core.config import get_settings
    from app.ml.evidence.service import RetinalEvidenceService
    from app.ml.explainability.service import ExplainabilityService
    from app.ml.inference.classifier import GRADE_LABELS, TorchDRClassificationService
    from app.ml.models.classifier import ReferableDRMapping, build_classifier, severity_probabilities
    from app.ml.quality.trust_gate import ImageTrustGateService, TrustGateDecision
    from app.ml.trust.guard import RetinaGuardInputs, derive_lesion_evidence_strength, derive_vessel_evidence_status
    from app.services.container import get_evidence_service, get_retinaguard_service
    from app.services.runtime import resolve_path
    from app.services.screening_pipeline import ScreeningPipelineService
    from ml.training.retinal_preprocessing import RetinalFieldCrop, build_inference_transform
    from scripts.train_classifier import make_transforms

    settings = get_settings()
    raw_bytes = image_path.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as source:
        source_image = source.copy()
        original_metadata = _image_metadata(source, raw_bytes, "uploaded_file")

    quality_service = ImageTrustGateService(max_image_pixels=settings.max_image_pixels)
    quality_initial = await quality_service.assess(raw_bytes)
    prepared_bytes = raw_bytes
    quality_final = quality_initial
    enhancement_applied = False
    enhancement_passes = 0
    if quality_initial.quality_decision == TrustGateDecision.BORDERLINE:
        prepared_bytes = quality_service.enhance(raw_bytes)
        quality_final = await quality_service.assess(prepared_bytes)
        enhancement_applied = True
        enhancement_passes = 1
    with Image.open(io.BytesIO(prepared_bytes)) as prepared_image:
        prepared_metadata = _image_metadata(prepared_image, prepared_bytes, "classifier_input_after_quality_gate")
        prepared_copy = prepared_image.copy()

    classifier = TorchDRClassificationService(
        model_path=settings.classifier_model_path,
        backbone=settings.classifier_backbone,
        model_version=settings.classifier_model_version,
        device=settings.classifier_device,
        referable_mapping=ReferableDRMapping(
            name=f"grade_{settings.referable_min_grade}_or_worse",
            referable_grades=tuple(range(settings.referable_min_grade, 5)),
        ),
    )
    configured_path = resolve_path(settings.classifier_model_path)
    checkpoint_sha = _sha256(configured_path.read_bytes()) if configured_path and configured_path.is_file() else None
    checkpoint_manifest: dict[str, Any] = {}
    checkpoint_payload: dict[str, Any] = {}
    if configured_path and configured_path.is_file():
        checkpoint_payload = torch.load(configured_path, map_location="cpu", weights_only=False)
        manifest_path = configured_path.parent / "model_manifest.json"
        if manifest_path.is_file():
            checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    classifier._load()  # Same lazy loader used by production inference.
    model = classifier._model
    transform = classifier._transform
    input_size = int(classifier._artifact_config.get("input_size", 224))
    train_transform, validation_transform = make_transforms(input_size)

    # Independently load the same checkpoint into a fresh in-memory model so
    # missing/unexpected keys are visible in the forensic record.
    model_config = checkpoint_payload.get("model_config", {})
    verification_model = build_classifier(
        backbone=model_config.get("backbone", settings.classifier_backbone),
        num_classes=5,
        pretrained=False,
        ordinal_mode=bool(model_config.get("ordinal_mode", False)),
    )
    load_result = verification_model.load_state_dict(checkpoint_payload["state_dict"], strict=False)
    verification_model.eval()

    artifact_uses_crop = bool(classifier._artifact_config.get("retinal_field_crop", False))
    pre_normalize_steps = [RetinalFieldCrop()] if artifact_uses_crop else []
    pre_normalize_steps.extend([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
    pre_normalize_transform = transforms.Compose(pre_normalize_steps)
    pre_normalized_tensor = pre_normalize_transform(prepared_copy).unsqueeze(0)
    normalized_tensor = transform(prepared_copy).unsqueeze(0)
    tensor = normalized_tensor.to(classifier._device)

    with torch.inference_mode():
        outputs = model(tensor)
        severity_probability_tensor = severity_probabilities(outputs, classifier._ordinal_mode)[0].detach().cpu()
        stage1_probability_tensor = torch.softmax(outputs["stage1_logits"], dim=1)[0].detach().cpu()
        stage2_probability_tensor = torch.softmax(outputs["stage2_logits"], dim=1)[0].detach().cpu()

    prediction = classifier._prediction_from_outputs(outputs)
    severity_logits = outputs.get("severity_logits")
    raw_logits = severity_logits[0].detach().cpu().tolist() if severity_logits is not None else None
    predicted_class_argmax = int(torch.argmax(severity_probability_tensor).item())
    labels = [GRADE_LABELS[index] for index in range(5)]
    probability_vector = [float(value) for value in severity_probability_tensor.tolist()]

    evidence = None
    explanation = None
    evidence_error = None
    explainability_error = None
    diagnostic_image_id = "forensic-diagnostic-image"
    diagnostic_session_id = "forensic-diagnostic-session"
    if not args.skip_evidence:
        try:
            evidence = await get_evidence_service().analyze(prepared_bytes, diagnostic_image_id, diagnostic_session_id, args.eye)
        except Exception as exc:  # Report the failure; never substitute output.
            evidence_error = {"type": type(exc).__name__, "message": str(exc)}
    if not args.skip_explainability and evidence is not None:
        try:
            explainability = ExplainabilityService(
                classifier=classifier,
                stability_enabled=settings.explainability_stability_enabled,
                counterfactual_enabled=settings.explainability_counterfactual_enabled,
                max_stability_variants=settings.explainability_max_stability_variants,
            )
            explanation = await explainability.analyze(
                prepared_bytes,
                diagnostic_image_id,
                diagnostic_session_id,
                evidence,
                run_stability=args.run_stability,
                run_counterfactual=args.run_counterfactual,
            )
        except Exception as exc:  # Report the failure; never substitute output.
            explainability_error = {"type": type(exc).__name__, "message": str(exc)}

    evidence_strength = derive_lesion_evidence_strength(evidence) if evidence is not None else None
    vessel_status = derive_vessel_evidence_status(evidence) if evidence is not None else "UNAVAILABLE"
    agreement = explanation.attention_lesion_agreement if explanation is not None else None
    stability = explanation.explanation_stability if explanation is not None else None
    quality_final_dict = quality_final.to_dict()
    common_guard_inputs = dict(
        quality_score=quality_final.quality_score,
        raw_confidence=prediction.raw_confidence,
        probabilities=prediction.probabilities,
        classifier_logits=prediction.severity_logits,
        model_predictions=[],
        quality_feature_vector=quality_final.feature_vector,
        predicted_grade=prediction.predicted_grade,
        predicted_grade_label=prediction.predicted_grade_label,
        referable_dr=prediction.referable_dr,
        model_version=prediction.model_version,
    )
    guard_engine = get_retinaguard_service()
    primary_guard = await guard_engine.evaluate_async(
        RetinaGuardInputs(**common_guard_inputs), prepared_bytes, classifier
    )
    enriched_guard = await guard_engine.evaluate_async(
        RetinaGuardInputs(
            **common_guard_inputs,
            lesion_evidence_strength=evidence_strength,
            vessel_evidence_status=vessel_status,
            attention_lesion_agreement=agreement,
            explanation_stability=stability,
        ),
        prepared_bytes,
        classifier,
    )
    primary_triage = ScreeningPipelineService._triage_payload(prediction, primary_guard)
    enriched_triage = ScreeningPipelineService._triage_payload(prediction, enriched_guard)

    configured_checksum = settings.classifier_model_sha256 or None
    manifest_checksum = checkpoint_manifest.get("checkpoint_sha256")
    architecture = {
        "runtime_class": model.__class__.__name__ if model is not None else None,
        "backbone": classifier._artifact_config.get("backbone", settings.classifier_backbone),
        "model_config": model_config,
        "ordinal_mode": classifier._ordinal_mode,
        "model_eval_mode": bool(model is not None and not model.training),
        "device": str(classifier._device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) if model is not None else None,
    }

    output = {
        "diagnostic": {
            "command": "scripts/diagnose_inference.py",
            "read_only": True,
            "database_written": False,
            "model_artifacts_written": False,
            "clinical_correctness_claim": False,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "input_metadata": {
            "original": original_metadata,
            "prepared": prepared_metadata,
            "quality_gate_prepared_input": "enhanced_derivative" if enhancement_applied else "original_bytes",
        },
        "quality_gate": {
            "initial": quality_initial.to_dict(),
            "final": quality_final_dict,
            "enhancement_applied": enhancement_applied,
            "enhancement_passes": enhancement_passes,
            "clinical_ai_allowed_by_gate": quality_final.quality_decision == TrustGateDecision.GRADABLE,
            "classifier_execution_in_this_diagnostic": "forensic_only_when_not_gradable",
        },
        "preprocessing": {
            "training": {
                "train_transform": _transform_metadata(train_transform),
                "validation_transform": _transform_metadata(validation_transform),
                "source": "scripts/train_classifier.py::make_transforms",
            },
            "inference": {
                "transform": _transform_metadata(transform),
                "source": "backend/app/ml/inference/classifier.py::TorchDRClassificationService._load",
                "input_color_conversion": "PIL Image.convert('RGB')",
                "bgr_conversion": "none in classifier; OpenCV BGR is used only inside quality/evidence CV paths",
                "resize_or_crop": f"RetinalFieldCrop -> Resize(({input_size}, {input_size}))" if artifact_uses_crop else f"Resize(({input_size}, {input_size})); no crop",
                "registered_artifact_retinal_field_crop": artifact_uses_crop,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "pre_normalization_tensor": _tensor_metadata(pre_normalized_tensor, "after Resize + ToTensor, before Normalize"),
                "normalized_tensor": _tensor_metadata(normalized_tensor, "after Resize + ToTensor + Normalize"),
            },
            "training_validation_matches_inference": _transform_metadata(build_inference_transform(input_size, artifact_uses_crop))["transforms"] == _transform_metadata(transform)["transforms"],
            "training_augmentation_not_used_at_inference": True,
        },
        "checkpoint": {
            "configured_path": str(configured_path) if configured_path else None,
            "filename": configured_path.name if configured_path else None,
            "actual_sha256": checkpoint_sha,
            "settings_sha256": configured_checksum,
            "manifest_sha256": manifest_checksum,
            "settings_checksum_matches_actual": configured_checksum.lower() == checkpoint_sha if configured_checksum and checkpoint_sha else None,
            "manifest_checksum_matches_actual": str(manifest_checksum).lower() == checkpoint_sha if manifest_checksum and checkpoint_sha else None,
            "model_version": classifier._artifact_config.get("model_version") or settings.classifier_model_version,
            "state_dict_key_count": len(checkpoint_payload.get("state_dict", {})),
            "weights_loaded": True,
            "load_state_dict_missing_keys": list(load_result.missing_keys),
            "load_state_dict_unexpected_keys": list(load_result.unexpected_keys),
        },
        "architecture": architecture,
        "classifier": {
            "class_mapping": {str(index): label for index, label in enumerate(labels)},
            "raw_severity_logits": raw_logits,
            "severity_probability_vector_class_order": labels,
            "softmax_probabilities": probability_vector,
            "softmax_probability_sum": sum(probability_vector),
            "predicted_class_argmax_before_business_logic": predicted_class_argmax,
            "prediction_service_payload": _prediction_payload(prediction),
            "stage1_logits": outputs["stage1_logits"][0].detach().cpu().tolist(),
            "stage1_probabilities": stage1_probability_tensor.tolist(),
            "stage2_logits": outputs["stage2_logits"][0].detach().cpu().tolist(),
            "stage2_probabilities": stage2_probability_tensor.tolist(),
            "referable_formula": "sum(probabilities[index] for index in referable_grades)",
            "referable_grades": list(prediction.referable_mapping["referable_grades"]),
            "referable_probability": prediction.referable_probability,
            "referable_boolean": prediction.referable_dr,
        },
        "uncertainty": {
            "primary": primary_guard.uncertainty,
            "enriched": enriched_guard.uncertainty,
            "mc_dropout_enabled": settings.retinaguard_mc_dropout_enabled,
        },
        "model_disagreement": {
            "input_additional_models": [],
            "primary": primary_guard.model_disagreement,
            "enriched": enriched_guard.model_disagreement,
        },
        "retinaguard": {
            "master_api_path": {
                "state": primary_guard.to_dict(),
                "triage": primary_triage,
                "ordering": "quality -> classifier -> uncertainty/disagreement -> RetinaGuard -> triage; optional evidence is queued afterward",
            },
            "enriched_comparison_only": {
                "state": enriched_guard.to_dict(),
                "triage": enriched_triage,
                "ordering": "evidence/explainability supplied before RetinaGuard; this is comparable to the direct trust route, not the current master /screening/run ordering",
            },
            "grade_changed_by_retinaguard": False,
        },
        "evidence": {
            "availability": _module_summary(evidence),
            "error": evidence_error,
            "changes_classifier_grade": False,
            "changes_master_api_prediction": False,
            "role": "supporting evidence; not a classifier override",
        },
        "explainability": {
            "availability": _explainability_summary(explanation),
            "error": explainability_error,
            "changes_classifier_grade": False,
            "role": "model-linked explanation and engineering agreement diagnostic",
        },
        "final_projections": {
            "master_api_response_values": {
                "quality_blocked": quality_final.quality_decision != TrustGateDecision.GRADABLE,
                "classification": _prediction_payload(prediction) if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "forensic_classifier_output_when_quality_blocked": _prediction_payload(prediction) if quality_final.quality_decision != TrustGateDecision.GRADABLE else None,
                "retinaguard": primary_guard.to_dict() if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "triage": primary_triage if quality_final.quality_decision == TrustGateDecision.GRADABLE else {"recommendation": "RECAPTURE_IMAGE", "clinical_ai_started": False},
                "model_versions": {
                    "preprocessing": "image-trust-gate-v1",
                    "dr_classifier": prediction.model_version if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                    "dr_backbone": prediction.backbone if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                    "retinaguard": primary_guard.configuration.get("version") if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                    "confidence_calibration": primary_guard.configuration.get("calibration_version") if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                    "optional_evidence": _module_summary(evidence),
                    "explainability": _explainability_summary(explanation),
                },
            },
            "frontend_results_page_values": {
                "quality_blocked": quality_final.quality_decision != TrustGateDecision.GRADABLE,
                "dr_severity": prediction.predicted_grade_label if quality_final.quality_decision == TrustGateDecision.GRADABLE else "Not available",
                "dr_grade_level": f"Level {prediction.predicted_grade} of 4" if quality_final.quality_decision == TrustGateDecision.GRADABLE else "No model output",
                "referable_dr": ("Yes" if prediction.referable_dr else "No") if quality_final.quality_decision == TrustGateDecision.GRADABLE else "Not available",
                "referable_probability": prediction.referable_probability if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "confidence": prediction.raw_confidence if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "model_version": prediction.model_version if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "trust_score": primary_guard.trust_score if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "trust_category": primary_guard.trust_category if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "recommended_action": primary_triage["recommendation"] if quality_final.quality_decision == TrustGateDecision.GRADABLE else "Recapture image",
            },
            "pdf_report_values": {
                "quality_blocked": quality_final.quality_decision != TrustGateDecision.GRADABLE,
                "dr_grade": prediction.predicted_grade_label if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "referable_dr": prediction.referable_dr if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "confidence": prediction.raw_confidence if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "recommended_action": primary_triage["recommendation"] if quality_final.quality_decision == TrustGateDecision.GRADABLE else "RECAPTURE_IMAGE",
                "trust_score": primary_guard.trust_score if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "reliability_state": primary_guard.trust_category if quality_final.quality_decision == TrustGateDecision.GRADABLE else None,
                "evidence_status": primary_guard.to_dict().get("evidence_status") if quality_final.quality_decision == TrustGateDecision.GRADABLE else "NOT_RUN",
                "explanation_status": primary_guard.to_dict().get("explanation_status") if quality_final.quality_decision == TrustGateDecision.GRADABLE else "NOT_RUN",
                "ood_status": primary_guard.to_dict().get("ood_status") if quality_final.quality_decision == TrustGateDecision.GRADABLE else "NOT_RUN",
                "report_source": "backend/app/api/routes/reports.py::_build_payload and _pdf_bytes",
            },
        },
        "provenance": {
            "settings_environment": settings.environment,
            "classifier_source": "backend/app/ml/inference/classifier.py",
            "classifier_architecture_source": "backend/app/ml/models/classifier.py",
            "training_source": "scripts/train_classifier.py",
            "quality_gate_source": "backend/app/ml/quality/trust_gate.py",
            "evidence_source": "backend/app/ml/evidence/service.py",
            "explainability_source": "backend/app/ml/explainability/service.py",
            "retinaguard_source": "backend/app/ml/trust/guard.py",
            "master_orchestration_source": "backend/app/services/screening_pipeline.py and backend/app/api/routes/screening.py",
            "frontend_rendering_source": "frontend/src/pages/ResultsPage.tsx",
            "report_rendering_source": "backend/app/api/routes/reports.py",
            "dataset_version": checkpoint_manifest.get("dataset_version") or checkpoint_payload.get("dataset_version"),
            "clinical_validation_claim": False,
        },
    }
    return _json_value(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RETINA-NEXUS image inference forensic diagnostic")
    parser.add_argument("image", help="One JPEG or PNG retinal image")
    parser.add_argument("--eye", choices=("left", "right"), default="right")
    parser.add_argument("--output", help="Optional JSON output path; stdout is always emitted")
    parser.add_argument("--skip-evidence", action="store_true", help="Do not run lesion/vessel evidence inference")
    parser.add_argument("--skip-explainability", action="store_true", help="Do not run Grad-CAM/agreement")
    parser.add_argument("--run-stability", action="store_true", help="Run configured perturbation stability diagnostics")
    parser.add_argument("--run-counterfactual", action="store_true", help="Run configured experimental counterfactual diagnostic")
    args = parser.parse_args()
    try:
        result = asyncio.run(_run(args))
    except Exception as exc:
        print(json.dumps({"diagnostic_error": {"type": type(exc).__name__, "message": str(exc)}}, indent=2))
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
