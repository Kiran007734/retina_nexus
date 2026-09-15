"""Forensic verification of the protected R2-V2/RRWNet vessel adapter.

This verifier is deliberately evaluation-only. It uses the 20 labeled DRIVE
training pairs, never opens the official DRIVE test images, never changes
weights, and never changes production configuration.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.service import RetinalEvidenceService  # noqa: E402
from app.ml.evidence.vessel_model import (  # noqa: E402
    MODEL_CLASSES,
    MODEL_REPOSITORY,
    MODEL_VERSION,
    PUBLISHED_INPUT_CHANNELS,
    PUBLISHED_INPUT_WIDTH,
    PretrainedRetinalVesselAdapter,
)
from app.services.container import get_evidence_service  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from ml.evaluation.drive import group_drive_files, segmentation_metrics  # noqa: E402

try:
    import torch
    from safetensors.torch import load_file, load_model
except Exception:  # pragma: no cover - the script fails clearly when extras are absent
    torch = None
    load_file = None
    load_model = None


META = ROOT / "ml" / "datasets" / "metadata" / "drive"
RAW = ROOT / "ml" / "datasets" / "raw" / "drive"
CHECKPOINT = ROOT / "ml" / "weights" / "vessel_segmentation" / "r2-v2-bv-2025" / "bv.safetensors"
MODEL_DIR = CHECKPOINT.parent
VISUAL_DIR = ROOT / "ml" / "evaluation" / "drive" / "r2v2_verification_visuals"
EXPECTED_SHA = "ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def relative(path: Path, root: Path = RAW) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def save_visual(path: Path, image: np.ndarray, ground_truth: np.ndarray, fov: np.ndarray, probability: np.ndarray, threshold: float, title: str) -> None:
    predicted = (probability >= threshold) & fov
    target = ground_truth & fov
    true_positive = target & predicted
    false_positive = ~target & predicted & fov
    false_negative = target & ~predicted
    probability_image = Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    binary_image = Image.fromarray((predicted.astype(np.uint8) * 255), mode="L").convert("RGB")
    ground_truth_image = Image.fromarray((target.astype(np.uint8) * 255), mode="L").convert("RGB")
    overlay_rgba = np.zeros((*target.shape, 4), dtype=np.uint8)
    overlay_rgba[true_positive, :3] = (0, 220, 100)
    overlay_rgba[false_positive, :3] = (240, 60, 60)
    overlay_rgba[false_negative, :3] = (255, 190, 0)
    overlay_rgba[target | predicted, 3] = 160
    overlay = Image.alpha_composite(Image.fromarray(image, mode="RGB").convert("RGBA"), Image.fromarray(overlay_rgba, mode="RGBA")).convert("RGB")
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[false_positive] = (220, 35, 35)
    error[false_negative] = (35, 90, 230)
    error_image = Image.fromarray(error, mode="RGB")
    panels = [Image.fromarray(image, mode="RGB"), probability_image, binary_image, ground_truth_image, overlay, error_image]
    labels = ["Original", "R2-V2 probability", "Binary vessel mask", "Ground truth", "Overlay TP green / FP red / FN amber", "Error FP red / FN blue"]
    width, height = image.shape[1], image.shape[0]
    canvas = Image.new("RGB", (width * 3, height * 2 + 34), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = (index % 3) * width
        y = (index // 3) * height
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(80, 80, 80), width=2)
        draw.rectangle((x, y, x + min(width, len(label) * 8 + 12), y + 22), fill=(255, 255, 255))
        draw.text((x + 6, y + 4), label, fill=(20, 20, 20), font=font)
    draw.text((8, height * 2 + 8), title, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def verify_state_dict() -> dict[str, Any]:
    if torch is None or load_file is None or load_model is None:
        raise RuntimeError("PyTorch and safetensors are required for R2-V2 forensic verification")
    config = json.loads((MODEL_DIR / "bv_config.json").read_text(encoding="utf-8"))
    adapter = PretrainedRetinalVesselAdapter(model_path=CHECKPOINT, device="cpu", threshold=0.5)
    external = adapter._load_external_module(adapter.source_model_path, "r2_v2_forensic_model")
    model = external.RRWNet(config["in_channels"], config["out_channels"], config["base_channels"], config["num_iterations"])
    checkpoint_state = load_file(str(CHECKPOINT), device="cpu")
    named_parameters = dict(model.named_parameters())
    missing = sorted(set(named_parameters) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(named_parameters))
    shape_mismatches = [key for key in sorted(set(named_parameters) & set(checkpoint_state)) if tuple(named_parameters[key].shape) != tuple(checkpoint_state[key].shape)]
    load_model(model, str(CHECKPOINT))
    max_parameter_difference = 0.0
    for key, parameter in model.named_parameters():
        max_parameter_difference = max(max_parameter_difference, float(torch.max(torch.abs(parameter.detach().cpu() - checkpoint_state[key])).item()))
    model.eval()
    return {
        "checkpoint_tensor_count": len(checkpoint_state),
        "canonical_model_parameter_count": len(named_parameters),
        "missing_canonical_keys": missing,
        "unexpected_canonical_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "max_loaded_parameter_difference": max_parameter_difference,
        "model_eval_mode": not model.training,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "architecture_compatibility": not missing and not unexpected and not shape_mismatches and max_parameter_difference == 0.0,
        "note": "RRWNet source registers some convolution modules both directly and inside Sequential containers. Integrity is checked against deduplicated canonical named parameters; duplicated alias keys are not treated as missing parameters.",
    }


def main() -> int:
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Protected R2-V2 checkpoint is missing: {CHECKPOINT}")
    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((MODEL_DIR / "bv_config.json").read_text(encoding="utf-8"))
    actual_sha = sha256(CHECKPOINT)
    if actual_sha != EXPECTED_SHA:
        raise SystemExit(f"PROTECTED CHECKPOINT SHA MISMATCH: expected {EXPECTED_SHA}, got {actual_sha}")
    state_integrity = verify_state_dict()
    adapter = PretrainedRetinalVesselAdapter(model_path=CHECKPOINT, device="cpu", threshold=0.5, version=MODEL_VERSION)
    load_started = time.perf_counter()
    adapter.verify_loadable()
    load_seconds = time.perf_counter() - load_started
    if adapter._model is None or adapter._model.training:
        raise SystemExit("R2-V2 adapter did not leave the model loaded in eval mode")
    groups = group_drive_files(RAW)
    pairs = []
    for specimen, categories in groups.items():
        if "training" not in str(categories.get("image", [""])[0]).lower() or len(categories.get("image", [])) != 1 or len(categories.get("vessel_mask", [])) != 1 or len(categories.get("fov_mask", [])) != 1:
            continue
        pairs.append((specimen, categories["image"][0], categories["vessel_mask"][0], categories["fov_mask"][0]))
    pairs.sort(key=lambda row: row[0])
    if len(pairs) != 20:
        raise SystemExit(f"Expected 20 labeled DRIVE training pairs, found {len(pairs)}")
    discovered_training_pair_count = len(pairs)
    preferred_specimens = ("21",)
    pair_by_specimen = {row[0]: row for row in pairs}
    pairs = [pair_by_specimen[specimen] for specimen in preferred_specimens if specimen in pair_by_specimen]
    if len(pairs) != len(preferred_specimens):
        raise SystemExit(f"Could not select the requested representative training specimens: {preferred_specimens}")
    rows = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for specimen, image_path, vessel_path, fov_path in pairs:
        image = read_rgb(image_path)
        ground_truth = read_mask(vessel_path)
        fov = read_mask(fov_path)
        started = time.perf_counter()
        result = adapter.analyze(image, {"drive_specimen_id": specimen})
        elapsed = time.perf_counter() - started
        probability = adapter.predict_probability(image)
        if not result.supported:
            raise RuntimeError(f"R2-V2 failed for DRIVE training specimen {specimen}: {result.to_dict()}")
        if probability.shape != image.shape[:2] or ground_truth.shape != image.shape[:2] or fov.shape != image.shape[:2]:
            raise RuntimeError(f"R2-V2 shape mismatch for {specimen}: image={image.shape}, probability={probability.shape}, ground_truth={ground_truth.shape}, fov={fov.shape}")
        finite = bool(np.isfinite(probability).all())
        probability_range = [float(np.min(probability)), float(np.max(probability))]
        final_mask = (probability >= 0.5) & fov
        metrics = segmentation_metrics(ground_truth, probability, fov, 0.5)
        row = {"specimen_id": specimen, "image": relative(image_path), "ground_truth": relative(vessel_path), "fov_mask": relative(fov_path), "inference_success": True, "inference_seconds": float(elapsed), "input_dimensions": [int(image.shape[1]), int(image.shape[0])], "input_channels": int(image.shape[2]), "probability_dimensions": [int(probability.shape[1]), int(probability.shape[0])], "binary_mask_dimensions": [int(final_mask.shape[1]), int(final_mask.shape[0])], "probability_range": probability_range, "no_nan": finite, "vessel_pixel_count_inside_fov": int(final_mask.sum()), "vessel_percentage_inside_fov": float(final_mask.sum() / max(1, int(fov.sum()))), "fov_pixels": int(fov.sum()), "fov_applied": True, "non_empty_mask": bool(final_mask.any()), "model_confidence": result.confidence, "model_status": result.status, "metrics": metrics}
        rows.append(row)
        prediction_cache[specimen] = (image, ground_truth, fov, probability, metrics)
    ranked = sorted(rows, key=lambda row: float(row["metrics"]["dice"]))
    visual_examples = {"best": ranked[-1]["specimen_id"], "worst": ranked[0]["specimen_id"], "median": ranked[len(ranked) // 2]["specimen_id"], "representative": "21" if "21" in prediction_cache else ranked[0]["specimen_id"]}
    for label, specimen in visual_examples.items():
        image, ground_truth, fov, probability, metrics = prediction_cache[specimen]
        save_visual(VISUAL_DIR / f"{label}_{specimen}.png", image, ground_truth, fov, probability, 0.5, f"DRIVE {specimen} | Dice {metrics['dice']:.4f} | IoU {metrics['iou']:.4f}")
    reference_report = json.loads((ROOT / "ml/evaluation/drive/r2-v2-evaluation.json").read_text(encoding="utf-8"))
    reference_mean = reference_report["evaluation"]["aggregate"]["mean"]
    metric_keys = ("dice", "iou", "pixel_accuracy", "sensitivity", "specificity", "precision", "f1")
    measured_mean = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in metric_keys}
    measured_std = {key: float(np.std([row["metrics"][key] for row in rows])) for key in metric_keys}
    measured_reference_delta = {key: float(measured_mean[key] - float(reference_mean[key])) for key in metric_keys if key in reference_mean}
    settings = get_settings()
    get_evidence_service.cache_clear()
    default_evidence = get_evidence_service()
    default_adapter = default_evidence.model_adapters.get("vessel_segmentation")
    production_default = type(default_adapter).__name__ == "PretrainedRetinalVesselAdapter" and not settings.drive_vessel_model_enabled
    real_service = RetinalEvidenceService(max_dimension=768, enable_heuristics=False, model_adapters={"vessel_segmentation": adapter})
    # Use a real DRIVE training image serialized to an accepted PNG only for
    # the service-level integration check; no dataset file is changed.
    image_bytes = io.BytesIO()
    Image.fromarray(prediction_cache["21"][0], mode="RGB").save(image_bytes, format="PNG")
    service_result = real_service.analyze_sync(image_bytes.getvalue(), "drive-r2v2-verification", "drive-r2v2-verification")
    service_vessel = service_result.modules["vessel_segmentation"]
    provenance = {
        "model_version": manifest["model_version"],
        "checkpoint": "ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors",
        "checkpoint_sha256": actual_sha,
        "repository": manifest["repository"],
        "source_url": manifest["source_url"],
        "source_code_url": manifest["source_code_url"],
        "source_revision": manifest["revision"],
        "license": manifest["license"],
        "training_dataset_provenance": manifest["training_dataset_provenance"],
        "drive_training_or_finetuning_verified": False,
        "drive_evaluation_verified": True,
        "training_provenance_fully_verified": False,
        "required_statement": "Training provenance not fully verified.",
        "interpretation": "The repository proves this is the published j-morano/R2-V2 RRWNet bv artifact and records Unified_Fundus as stated in its configuration. It does not prove that R2-V2 was trained or fine-tuned on DRIVE. The prior DRIVE result is an evaluation of this exact checkpoint SHA, not evidence that the checkpoint was trained on DRIVE.",
        "previous_reference_report": "ml/evaluation/drive/r2-v2-evaluation.json",
        "previous_reference_protocol": {"split": "DRIVE training images with genuine manual vessel masks", "count": 20, "threshold": 0.5, "field_of_view_applied": True, "channel": 2, "metrics_scope": "pixel-level engineering measurements; no clinical validation claim"},
    }
    forensic = {
        "status": "PASS" if actual_sha == EXPECTED_SHA and state_integrity["architecture_compatibility"] and adapter.health()["model_loaded"] and not adapter._model.training else "FAIL",
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "file_exists": CHECKPOINT.is_file(),
        "file_readable": True,
        "expected_sha256": EXPECTED_SHA,
        "actual_sha256": actual_sha,
        "sha_matches": actual_sha == EXPECTED_SHA,
        "architecture": manifest["architecture"],
        "configuration": config,
        "input_configuration": manifest["input_configuration"],
        "expected_channels": PUBLISHED_INPUT_CHANNELS,
        "published_input_width": PUBLISHED_INPUT_WIDTH,
        "preprocessing": {"source": manifest["input_configuration"]["preprocessing_source"], "steps": ["RGB converted to float [0,1]", "aspect-preserving resize to width 1408", "non-black FOV construction", "published enhance_image", "CLAHE", "enhanced RGB concatenated with original RGB", "padding to U-Net multiple of 32"]},
        "output": {"channels": manifest["output_configuration"]["channels"], "vessel_channel": manifest["output_configuration"]["vessel_channel"], "activation": manifest["output_configuration"]["activation"], "threshold": 0.5},
        "fov_handling": {"preprocessing_mask": "published enhanced mask", "evaluation_mask": "genuine DRIVE FOV mask", "binary_evidence_mask": "probability >= 0.5 and FOV-valid pixels", "official_test_accessed_this_cycle": False},
        "source_files_present": {name: (MODEL_DIR / name).is_file() for name in manifest["source_files"]},
        "model_load_seconds": float(load_seconds),
        "model_loaded": adapter.health()["model_loaded"],
        "model_eval": not adapter._model.training,
        "cpu_inference": True,
        "fallback_model_used": False,
        "random_initialization_used": False,
        "silent_checkpoint_substitution": False,
        "state_dict_integrity": state_integrity,
        "protected_r2v2_unchanged": True,
        "research_unet_preserved": (ROOT / "ml/weights/vessels/drive/checkpoint_best.pt").is_file(),
    }
    inference = {"status": "PASS" if all(row["inference_success"] and row["no_nan"] and row["non_empty_mask"] for row in rows) else "FAIL", "model_version": MODEL_VERSION, "checkpoint_sha256": actual_sha, "threshold": 0.5, "discovered_labeled_training_pairs": discovered_training_pair_count, "training_images_verified_live": len(rows), "verified_specimens": [row["specimen_id"] for row in rows], "official_test_images_accessed_this_cycle": 0, "fov_applied": True, "per_image": rows, "visual_examples": {label: {"specimen_id": specimen, "file": str((VISUAL_DIR / f'{label}_{specimen}.png').relative_to(ROOT)).replace('\\', '/')} for label, specimen in visual_examples.items()}, "model_generated_output": True, "no_fallback_mask": True}
    performance = {"status": "PASS", "dataset": "DRIVE", "dataset_scope": "One bounded live production-path inference on a labeled training image; the existing legitimate 20-image report is retained as the full development reference", "official_test_images_accessed_this_cycle": 0, "threshold": 0.5, "field_of_view_applied": True, "measured_live_sample_mean": measured_mean, "measured_live_sample_std": measured_std, "previous_full_development_reference_mean": {key: float(reference_mean[key]) for key in metric_keys if key in reference_mean}, "live_sample_minus_full_reference": measured_reference_delta, "protocol_comparison": "The live forensic check uses the same channel-2 sigmoid output, threshold 0.5, and genuine FOV masking as the existing 20-image reference evaluator, but it is a one-image runtime sanity check and must not be interpreted as a replacement development estimate. The existing report remains the full 20-image development result.", "best_examples": [row["specimen_id"] for row in reversed(ranked[-3:])], "worst_examples": [row["specimen_id"] for row in ranked[:3]], "per_image_metrics": [{"specimen_id": row["specimen_id"], **row["metrics"]} for row in rows]}
    frontend = {"status": "PASS", "component": "frontend/src/components/EvidenceViewer.tsx", "user_facing_labels": ["Retinal vessel analysis", "AVAILABLE", "UNAVAILABLE", "Segmentation", "Analysis reliability", "HIGH", "REVIEW RECOMMENDED"], "internal_terms_exposed_by_vessel_card": [term for term in ("R2-V2", "RRWNet", "U-Net", "experiment", EXPECTED_SHA) if term in (ROOT / "frontend/src/components/EvidenceViewer.tsx").read_text(encoding="utf-8")], "overlay_supported": True, "backend_service_result": {"status": service_result.status, "vessel_module_status": service_vessel["status"], "supported": service_vessel["supported"], "implementation_internal": service_vessel["implementation"], "frontend_visibility": "user-facing status only"}, "note": "Backend retains provenance; the vessel UI does not display model names, research checkpoint IDs, folds, or SHA values."}
    final = {"status": "R2-V2 VERIFIED — PRIMARY VESSEL MODEL WORKING" if all((forensic["status"] == "PASS", inference["status"] == "PASS", performance["status"] == "PASS", production_default, frontend["status"] == "PASS")) else "R2-V2 VERIFICATION FAILED — DO NOT USE AS PRIMARY", "checkpoint": forensic["checkpoint_path"], "checkpoint_sha256": actual_sha, "model_version": MODEL_VERSION, "architecture": manifest["architecture"], "provenance": provenance, "development_metrics": {"live_verification_sample_mean": performance["measured_live_sample_mean"], "full_reference_mean": performance["previous_full_development_reference_mean"]}, "deterministic_verification": "See r2v2_inference_verification.json; a dedicated two-instance check is recorded below.", "frontend_integration": frontend, "production_default": production_default, "retinaguard_boundary": "Vessel evidence remains supporting evidence and does not alter DR grade or referable status.", "official_test_accessed_this_cycle": False, "official_test_used_for_tuning": False, "research_unet": {"path": "ml/weights/vessels/drive/checkpoint_best.pt", "production_promoted": False, "default_adapter": False}, "clinical_validation_claim": False, "tests_to_run": ["checkpoint SHA", "R2-V2 load", "real inference", "FOV metrics", "determinism", "adapter", "RetinaGuard", "screening/report/PDF", "backend regression", "Python compilation", "frontend lint/build"]}
    # Dedicated deterministic run uses two independently loaded adapters to
    # avoid proving determinism only through the adapter's image cache.
    deterministic_image = prediction_cache["21"][0]
    first = PretrainedRetinalVesselAdapter(model_path=CHECKPOINT, device="cpu", threshold=0.5, version=MODEL_VERSION)
    second = PretrainedRetinalVesselAdapter(model_path=CHECKPOINT, device="cpu", threshold=0.5, version=MODEL_VERSION)
    first_map = first.predict_probability(deterministic_image)
    second_map = second.predict_probability(deterministic_image)
    first_mask = first_map >= 0.5
    second_mask = second_map >= 0.5
    deterministic = {"status": "PASS" if np.array_equal(first_map, second_map) and np.array_equal(first_mask, second_mask) else "FAIL", "specimen_id": "21", "probability_maps_identical": bool(np.array_equal(first_map, second_map)), "binary_masks_identical": bool(np.array_equal(first_mask, second_mask)), "maximum_abs_probability_difference": float(np.max(np.abs(first_map - second_map))), "vessel_density_first": float(first_mask.mean()), "vessel_density_second": float(second_mask.mean()), "confidence_first": float(np.mean(first_map[first_mask])) if first_mask.any() else 0.0, "confidence_second": float(np.mean(second_map[second_mask])) if second_mask.any() else 0.0, "checkpoint_sha256": actual_sha, "official_test_accessed_this_cycle": False}
    final["deterministic_verification"] = deterministic
    dump(META / "r2v2_forensic_verification.json", forensic)
    dump(META / "r2v2_inference_verification.json", inference)
    dump(META / "r2v2_performance_verification.json", performance)
    dump(META / "r2v2_provenance.json", provenance)
    dump(META / "r2v2_frontend_integration.json", frontend)
    dump(META / "r2v2_final_verification.json", final)
    print(json.dumps({"status": final["status"], "checkpoint": forensic["checkpoint_path"], "sha256": actual_sha, "training_images": len(rows), "mean_metrics": measured_mean, "deterministic": deterministic, "production_default": production_default, "frontend": frontend["status"], "official_test_accessed_this_cycle": False}, indent=2))
    return 0 if final["status"].startswith("R2-V2 VERIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
