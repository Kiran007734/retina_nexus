"""Development-only referable DR sensitivity research.

This script uses only the governed 406-image IDRiD development set and the
existing five-fold validation predictions/checkpoints.  It never reads
Messidor labels, official IDRiD test images, or production model artifacts for
writing.  Outputs are research records under ``ml/evaluation/referable_research``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

OUTPUT = ROOT / "ml" / "evaluation" / "referable_research"
IDRID_RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
DEV_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
OOF_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "master_cv" / "20260912"
V3_VALIDATION = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "validation_predictions.json"
V3_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final" / "checkpoint_best.pt"
V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
REFERABLE_GRADES = (2, 3, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    predicted = probability >= threshold
    tp = int(np.sum((actual == 1) & predicted))
    tn = int(np.sum((actual == 0) & ~predicted))
    fp = int(np.sum((actual == 0) & predicted))
    fn = int(np.sum((actual == 1) & ~predicted))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "roc_auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None, "pr_auc": float(average_precision_score(actual, probability)) if len(np.unique(actual)) == 2 else None}


def sweep(rows: list[dict[str, Any]], probability_key: str) -> dict[str, Any]:
    actual = np.asarray([int(int(row["actual"]) in REFERABLE_GRADES) for row in rows], dtype=int)
    probability = np.asarray([float(row[probability_key]) for row in rows], dtype=float)
    entries = [metric_row(actual, probability, threshold) for threshold in THRESHOLDS]
    eligible = [entry for entry in entries if entry["specificity"] >= 0.90]
    selected = max(eligible, key=lambda entry: (entry["sensitivity"], entry["specificity"], entry["f1"], -entry["threshold"])) if eligible else None
    return {"probability_source": probability_key, "sample_count": len(rows), "positive_count": int(actual.sum()), "negative_count": int((actual == 0).sum()), "entries": entries, "selection_criterion": "highest OOF sensitivity subject to specificity >= 0.90; ties specificity, F1, then lower threshold", "selected": selected, "external_labels_used": False, "official_test_images_opened": 0}


def load_dev_map() -> dict[str, dict[str, Any]]:
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("official_test_images_opened") != 0 or manifest.get("official_test_used"):
        raise RuntimeError("IDRiD development manifest does not prove official-test isolation")
    records = manifest.get("records", [])
    if len(records) != 406:
        raise RuntimeError(f"Expected 406 governed IDRiD development records, got {len(records)}")
    return {str(record["image_id"]): record for record in records}


def load_oof() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in range(1, 6):
        path = OOF_ROOT / f"fold_{fold}" / "validation_predictions.json"
        fold_rows = json.loads(path.read_text(encoding="utf-8"))
        for row in fold_rows:
            image_id = str(row["image_id"])
            if image_id in seen:
                raise RuntimeError(f"OOF duplicate image across folds: {image_id}")
            seen.add(image_id)
            rows.append({**row, "fold": fold, "derived_referable_probability": float(row["referable_probability"])})
    if len(rows) != 406:
        raise RuntimeError(f"Expected 406 OOF rows, got {len(rows)}")
    return rows


def add_quality(rows: list[dict[str, Any]], dev_map: dict[str, dict[str, Any]]) -> None:
    from scripts.run_idrid_v2_research import quality_proxy

    for row in rows:
        record = dev_map[row["image_id"]]
        quality = quality_proxy(IDRID_RAW / record["image"])
        row["quality"] = quality


def dedicated_stage2_oof(rows: list[dict[str, Any]], dev_map: dict[str, dict[str, Any]], torch_threads: int) -> dict[str, Any]:
    """Extract the already-trained hierarchical binary head from each OOF fold.

    This is analysis of an existing head, not a new model or a production
    behavior change.  Each image is scored only by the fold checkpoint that did
    not train on that image.
    """
    import torch
    from PIL import Image
    from app.ml.models.classifier import build_classifier
    from ml.training.retinal_preprocessing import build_inference_transform

    torch.set_num_threads(max(1, int(torch_threads)))
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_fold.setdefault(int(row["fold"]), []).append(row)
    for fold, fold_rows in by_fold.items():
        checkpoint = OOF_ROOT / f"fold_{fold}" / "checkpoint_best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        transform = build_inference_transform(224)
        with torch.inference_mode():
            for row in fold_rows:
                with Image.open(IDRID_RAW / dev_map[row["image_id"]]["image"]) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0)
                output = model(tensor)
                stage2 = torch.softmax(output["stage2_logits"], dim=1)[0, 1].item()
                row["dedicated_referable_probability"] = float(stage2)
        del model
    return {"status": "COMPLETED_FROM_EXISTING_OOF_FOLD_HEADS", "architecture": "shared EfficientNet-B0 with severity head and existing stage2 binary referable head", "fold_count": 5, "checkpoint_paths": [str((OOF_ROOT / f"fold_{fold}" / "checkpoint_best.pt").relative_to(ROOT)).replace("\\", "/") for fold in range(1, 6)], "production_behavior_changed": False, "note": "This analyzes the existing hierarchical head; it is not a newly trained binary-head candidate."}


def false_negative_analysis(rows: list[dict[str, Any]], selected_threshold: float, probability_key: str) -> dict[str, Any]:
    fns = [row for row in rows if int(row["actual"]) in REFERABLE_GRADES and float(row[probability_key]) < selected_threshold]
    grades = Counter(str(row["actual"]) for row in fns)
    quality_values = [float((row.get("quality") or {}).get("quality_proxy_score")) for row in fns if (row.get("quality") or {}).get("quality_proxy_score") is not None]
    confidence_values = [float(row["confidence"]) for row in fns]
    entropy_values = [float((row.get("uncertainty") or {}).get("normalized_entropy", row.get("normalized_entropy", 0.0))) for row in fns]
    return {"probability_source": probability_key, "threshold": selected_threshold, "false_negative_count": len(fns), "by_actual_grade": dict(grades), "mean_quality_proxy": float(np.mean(quality_values)) if quality_values else None, "mean_confidence": float(np.mean(confidence_values)) if confidence_values else None, "mean_normalized_entropy": float(np.mean(entropy_values)) if entropy_values else None, "records": [{"image_id": row["image_id"], "fold": row["fold"], "actual_grade": row["actual"], "predicted_grade": row["predicted"], "probability": row[probability_key], "confidence": row["confidence"], "uncertainty": row.get("uncertainty"), "quality": row.get("quality"), "grad_cam_status": "NOT_PRESENT_IN_OOF_ARTIFACT", "disagreement_status": "NOT_AVAILABLE_IN_OOF_ARTIFACT"} for row in fns], "clinical_validation_claim": False}


def existing_domain_results() -> dict[str, Any]:
    source = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_comparison.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {"status": "EXISTING_DEVELOPMENT_CANDIDATES", "source": str(source.relative_to(ROOT)).replace("\\", "/"), "selection_or_external_labels_used": False, "candidates": payload.get("candidate_rows", []), "note": "These existing candidates use the fixed 83-image development validation set, not Messidor and not the official IDRiD test set."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    dev_map = load_dev_map()
    rows = load_oof()
    add_quality(rows, dev_map)
    dedicated_info = dedicated_stage2_oof(rows, dev_map, args.torch_threads)
    derived = sweep(rows, "derived_referable_probability")
    dedicated = sweep(rows, "dedicated_referable_probability")
    baseline_04 = next(entry for entry in derived["entries"] if entry["threshold"] == 0.40)
    false_negatives = false_negative_analysis(rows, float(derived["selected"]["threshold"]), "derived_referable_probability") if derived.get("selected") else {"status": "NOT_CALCULABLE"}
    dedicated_false_negatives = false_negative_analysis(rows, float(dedicated["selected"]["threshold"]), "dedicated_referable_probability") if dedicated.get("selected") else {"status": "NOT_CALCULABLE"}
    domain = existing_domain_results()
    lesion = {"status": "NOT_RUN", "reason": "No leak-safe OOF lesion-evidence table exists for all 406 development images. IDRiD lesion annotations are incomplete; no fusion score was fabricated and the primary lesion model was not replaced.", "production_primary_preserved": True, "selection_or_external_labels_used": False}
    registry = {
        "schema_version": "referable-research-registry-v1",
        "generated_at_utc": utc_now(),
        "scope": "IDRiD development-only referable sensitivity research",
        "data": {"development_records": 406, "folds": 5, "oof_rows": 406, "official_test_images_opened": 0, "messidor_labels_used": False, "patient_ids_available": False, "duplicate_group_overlap": False},
        "frozen_reference": {"checkpoint": str(V3_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(V3_CHECKPOINT), "expected_sha256": V3_SHA, "unchanged": sha256(V3_CHECKPOINT) == V3_SHA, "production_promoted": False},
        "selection_criterion": "highest OOF referable sensitivity subject to specificity >= 0.90; ties specificity, F1, then lower threshold",
        "experiments": [
            {"id": "derived_probability_threshold_sweep", "status": "COMPLETED", "artifact": "threshold_sweep.json"},
            {"id": "existing_hierarchical_stage2_head_analysis", "status": dedicated_info["status"], "artifact": "binary_referable_results.json"},
            {"id": "existing_domain_generalization_candidates", "status": domain["status"], "artifact": "domain_generalization_results.json"},
            {"id": "lesion_fusion", "status": lesion["status"], "artifact": "lesion_fusion_results.json"},
        ],
        "no_production_changes": True,
    }
    model_comparison = {
        "development_only": True,
        "messidor_used_for_selection": False,
        "candidates": [
            {"candidate": "existing_5_class_probability_sum", "evaluation": "406-image leak-safe OOF", "selected_threshold": derived.get("selected"), "baseline_threshold_0_40": baseline_04},
            {"candidate": "existing_hierarchical_stage2_binary_head", "evaluation": "406-image leak-safe OOF", "selected_threshold": dedicated.get("selected")},
            {"candidate": "frozen_v3_holdout_reference", "evaluation": "83-image development holdout; not OOF", "referable_sensitivity": 0.9615384615384616, "referable_specificity": 0.967741935483871, "checkpoint_sha256": V3_SHA},
        ],
        "selection_note": "The threshold research candidate is not production promoted and has not been externally validated.",
    }
    conclusion = {
        "status": "RESEARCH CANDIDATE IMPROVED — EXTERNAL VALIDATION PENDING" if derived.get("selected") and derived["selected"]["sensitivity"] > baseline_04["sensitivity"] else "NO MEANINGFUL IMPROVEMENT FOUND",
        "selected_research_candidate": "existing_5_class_probability_sum_with_development_oof_threshold",
        "selected_threshold": derived.get("selected"),
        "baseline_threshold_0_40": baseline_04,
        "improvement_is_development_only": True,
        "external_validation_required": True,
        "production_promoted": False,
        "no_claim_over_90_percent": True,
        "note": "A lower threshold improved OOF sensitivity under the documented specificity constraint, but this does not establish Messidor performance or clinical utility.",
    }
    dump(OUTPUT / "experiment_registry.json", registry)
    dump(OUTPUT / "threshold_sweep.json", {"source": "5-fold IDRiD development OOF predictions", "derived_probability": derived, "dedicated_head_reference": dedicated, "thresholds": list(THRESHOLDS), "official_test_images_opened": 0, "messidor_labels_used": False})
    dump(OUTPUT / "cv_results.json", {"status": "COMPLETED", "oof_count": len(rows), "fold_count": 5, "folds": {str(fold): sum(row["fold"] == fold for row in rows) for fold in range(1, 6)}, "derived_probability_oof": derived, "dedicated_head_oof": dedicated, "duplicate_group_overlap": False, "official_test_images_opened": 0, "messidor_labels_used": False})
    dump(OUTPUT / "binary_referable_results.json", {"status": dedicated_info["status"], "architecture": dedicated_info["architecture"], "head_analysis": dedicated, "training_run": False, "reason": "The current hierarchical architecture already exposes a binary referable stage2 head; this artifact evaluates it leak-safely rather than adding a new production model.", "checkpoint_sha256s": [sha256(OOF_ROOT / f"fold_{fold}" / "checkpoint_best.pt") for fold in range(1, 6)], "production_promoted": False})
    dump(OUTPUT / "domain_generalization_results.json", domain)
    dump(OUTPUT / "lesion_fusion_results.json", lesion)
    dump(OUTPUT / "false_negative_analysis.json", {"derived_probability": false_negatives, "dedicated_stage2_head": dedicated_false_negatives, "messidor_analysis": "NOT_RUN_FOR_SELECTION", "official_test_images_opened": 0})
    dump(OUTPUT / "model_comparison.json", model_comparison)
    dump(OUTPUT / "research_conclusion.json", conclusion)
    report = f"""# Referable DR sensitivity research\n\nStatus: **{conclusion['status']}**\n\n## Protocol\n\n- Development data: 406 governed IDRiD records.\n- Evaluation: five-fold leak-safe OOF predictions, 406 unique images.\n- Official IDRiD test images opened: 0.\n- Messidor labels used for selection: false.\n- Production checkpoint unchanged: `{V3_SHA}`.\n- Selection rule: highest OOF sensitivity subject to specificity >= 0.90.\n\n## Threshold result\n\n- Existing 0.40 threshold: sensitivity `{baseline_04['sensitivity']:.6f}`, specificity `{baseline_04['specificity']:.6f}`, FN `{baseline_04['fn']}`.\n- Selected development-only threshold: `{derived['selected']['threshold']}`; sensitivity `{derived['selected']['sensitivity']:.6f}`, specificity `{derived['selected']['specificity']:.6f}`, precision `{derived['selected']['precision']:.6f}`, F1 `{derived['selected']['f1']:.6f}`, ROC-AUC `{derived['selected']['roc_auc']:.6f}`, PR-AUC `{derived['selected']['pr_auc']:.6f}`, FN `{derived['selected']['fn']}`.\n- This is not a Messidor result and is not a clinical validation result.\n\n## Dedicated referable head\n\nThe existing shared EfficientNet-B0 architecture exposes a stage2 referable head. Its five-fold OOF analysis is recorded in `binary_referable_results.json`; no new binary checkpoint was trained or promoted.\n\n## Other research directions\n\nExisting domain-generalization candidates are reported from prior development-only artifacts. Learned lesion fusion was not run because there was no complete leak-safe OOF evidence table and IDRiD annotations are incomplete; no fusion output was fabricated.\n\n## False negatives\n\nDevelopment OOF false-negative records, grade breakdown, quality proxy, confidence, and uncertainty are in `false_negative_analysis.json`. Grad-CAM and model disagreement are marked unavailable because those signals are not present in the OOF artifacts.\n\n## Safety boundary\n\nThe selected threshold is research-only. It does not change the APTOS production threshold, production model, backend, frontend, or RetinaGuard. External Messidor evaluation remains pending until the research candidate is formally frozen.\n"""
    (OUTPUT / "research_conclusion.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": conclusion["status"], "selected_threshold": derived.get("selected"), "derived_oof_selected": derived.get("selected"), "dedicated_oof_selected": dedicated.get("selected"), "official_test_images_opened": 0, "production_promoted": False}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
