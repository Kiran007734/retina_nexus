"""Assemble the referable DR V2 research artifacts without production changes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml" / "evaluation" / "referable_v2"


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_jsonl(relative: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(relative: str, payload: Any) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

    predicted = (probability >= threshold).astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    sensitivity = recall_score(actual, predicted, zero_division=0)
    specificity = tn / max(1, tn + fp)
    precision = precision_score(actual, predicted, zero_division=0)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "accuracy": float(accuracy_score(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": int(actual.sum()),
    }


def threshold_rows(actual: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    return [binary_metrics(actual, probability, threshold) for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)]


def main() -> int:
    generated = datetime.now(timezone.utc).isoformat()
    oof = load("ml/evaluation/referable_v2/oof_results.json")
    primary_external_summary = load("ml/evaluation/referable_v2/primary_binary_messidor/evaluation_summary.json")
    primary_external = read_jsonl("ml/evaluation/referable_v2/primary_binary_messidor/predictions.jsonl")
    verifier_summary = load("ml/evaluation/referable_v2/retguard_messidor/run_summary.json")
    verifier_rows = read_jsonl("ml/evaluation/referable_v2/retguard_messidor/predictions.jsonl")
    aptos_summary = load("ml/evaluation/messidor2/authoritative_classifier/evaluation_summary.json")
    idrid_summary = load("ml/evaluation/messidor2/authoritative_idrid_v3/evaluation_summary.json")
    domain = load("ml/evaluation/referable_research/domain_generalization_results.json")
    false_negative_existing = load("ml/evaluation/referable_research/false_negative_analysis.json")
    messidor_manifest = load("ml/evaluation/messidor2/authoritative_external_manifest.json")
    if len(primary_external) != 1744 or len(verifier_rows) != 1744:
        raise RuntimeError("Primary or verifier evaluation is not complete")
    primary_by_id = {row["image_id"]: row for row in primary_external}
    verifier_by_id = {row["image_id"]: row for row in verifier_rows}
    if set(primary_by_id) != set(verifier_by_id):
        raise RuntimeError("Primary and verifier image IDs do not match")

    primary_probability = np.asarray([float(row["referable_probability"]) for row in primary_external], dtype=float)
    primary_actual = np.asarray([int(row["actual_referable"]) for row in primary_external], dtype=int)
    verifier_probability = np.asarray([float(verifier_by_id[row["image_id"]]["probability"]) for row in primary_external], dtype=float)
    verifier_actual = np.asarray([int(verifier_by_id[row["image_id"]]["actual_referable"]) for row in primary_external], dtype=int)
    if not np.array_equal(primary_actual, verifier_actual):
        raise RuntimeError("Primary and verifier labels differ")

    primary_threshold = float(primary_external_summary["fixed_development_threshold"])
    verifier_threshold = float(verifier_summary["threshold"])
    primary_metrics = binary_metrics(primary_actual, primary_probability, primary_threshold)
    verifier_metrics = binary_metrics(primary_actual, verifier_probability, verifier_threshold)
    verifier_diagnostics = threshold_rows(primary_actual, verifier_probability)
    primary_decision = primary_probability >= primary_threshold
    verifier_decision = verifier_probability >= verifier_threshold
    disagreement = primary_decision != verifier_decision
    review_flag = disagreement | np.asarray([bool(verifier_by_id[row["image_id"]]["ood_flagged"]) for row in primary_external])
    safety_metrics = binary_metrics(primary_actual, review_flag.astype(float), 0.5)
    primary_fn = (primary_actual == 1) & ~primary_decision
    verifier_caught = primary_fn & verifier_decision
    both_missed = primary_fn & ~verifier_decision

    fusion_rows: list[dict[str, Any]] = []
    for row in primary_external:
        verifier = verifier_by_id[row["image_id"]]
        primary_p = float(row["referable_probability"])
        verifier_p = float(verifier["probability"])
        primary_d = bool(primary_p >= primary_threshold)
        verifier_d = bool(verifier_p >= verifier_threshold)
        is_disagreement = primary_d != verifier_d
        low_confidence = abs(primary_p - primary_threshold) < 0.05 or abs(verifier_p - verifier_threshold) < 0.05
        if is_disagreement:
            state = "DISAGREE"
        elif low_confidence:
            state = "AGREE_LOW_CONFIDENCE"
        else:
            state = "AGREE_HIGH_CONFIDENCE"
        fusion_rows.append({"image_id": row["image_id"], "actual_referable": row["actual_referable"], "primary_probability": primary_p, "primary_decision": primary_d, "verifier_probability": verifier_p, "verifier_decision": verifier_d, "verifier_ood_flagged": bool(verifier["ood_flagged"]), "state": state, "review_recommended": bool(is_disagreement or verifier["ood_flagged"]), "rule_version": "referable-v2-safety-review-v1"})

    domain_candidates = domain.get("candidates", [])
    domain_by_candidate = {str(row.get("candidate")): row for row in domain_candidates}
    domain_effects = []
    if "a" in domain_by_candidate and "c" in domain_by_candidate:
        domain_effects.append({"comparison": "candidate_a_control_vs_candidate_c_domain_balanced", "sensitivity_delta": domain_by_candidate["c"]["referable_sensitivity"] - domain_by_candidate["a"]["referable_sensitivity"], "specificity_delta": domain_by_candidate["c"]["referable_specificity"] - domain_by_candidate["a"]["referable_specificity"], "note": "Existing 83-image development validation comparison; not a new experiment and not external selection."})
    if "a" in domain_by_candidate and "d" in domain_by_candidate:
        domain_effects.append({"comparison": "candidate_a_control_vs_candidate_d_combined_finetune", "sensitivity_delta": domain_by_candidate["d"]["referable_sensitivity"] - domain_by_candidate["a"]["referable_sensitivity"], "specificity_delta": domain_by_candidate["d"]["referable_specificity"] - domain_by_candidate["a"]["referable_specificity"], "note": "Existing 83-image development validation comparison; not a new experiment and not external selection."})

    research_registry = {
        "schema_version": "retina-nexus-referable-v2-registry-v1",
        "generated_at_utc": generated,
        "scope": "research-only binary referable DR and independent verification",
        "production_behavior_changed": False,
        "production_promoted": False,
        "messidor_used_for_selection": False,
        "official_idrid_test_images_opened": 0,
        "development_data": {"records": 406, "folds": 5, "patient_ids_available": False, "duplicate_conflict_exclusions_preserved": True, "oof_artifact": "ml/evaluation/referable_v2/oof_predictions.jsonl"},
        "primary_candidate": {"name": "IDRiD V3 shared EfficientNet-B0 stage2 referable head", "role": "RESEARCH_ONLY_PRIMARY_BINARY_CANDIDATE", "positive_definition": "IDRiD grade 2/3/4", "negative_definition": "IDRiD grade 0/1", "checkpoint": "ml/weights/classifiers/idrid/research/v3/20260912/v3_b_domain_robust/checkpoint_best.pt", "checkpoint_sha256": sha256("ml/weights/classifiers/idrid/research/v3/20260912/v3_b_domain_robust/checkpoint_best.pt"), "frozen_development_threshold": primary_threshold, "threshold_source": "5-fold development OOF; specificity constrained >= 0.85", "training_status": "EXISTING_FROZEN_HEAD_ANALYZED; NO_NEW_PRODUCTION_CHECKPOINT"},
        "verifier": {"name": "RETGUARD DR v1.0.0", "role": "INDEPENDENT_VERIFIER", "checkpoint_sha256": verifier_summary["onnx_sha256"], "fixed_threshold": verifier_threshold, "weights_license": "CC BY-NC 4.0 research use; commercial licensing requires separate agreement", "source": "https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0"},
        "experiments": ["baseline weighted binary head analysis", "controlled development OOF threshold sweep", "existing domain-generalization candidate comparison", "independent verifier zero-shot evaluation", "safety disagreement analysis"],
        "not_run": ["uncontrolled training sweep", "Messidor threshold tuning", "learned fusion training", "production promotion"],
    }
    dump("ml/evaluation/referable_v2/research_registry.json", research_registry)

    dump("ml/evaluation/referable_v2/cv_results.json", {"status": "COMPLETED", "primary": oof, "development_data": research_registry["development_data"], "official_test_images_opened": 0, "messidor_labels_used": False, "clinical_validation_claim": False})
    dump("ml/evaluation/referable_v2/threshold_sweep.json", {"primary_oof": {"entries": oof["thresholds"], "selected": oof["selected"], "selection_criterion": oof["selection_criterion"]}, "verifier_external_diagnostics": {"entries": verifier_diagnostics, "fixed_threshold": verifier_threshold, "selection_allowed": False, "note": "External threshold rows are descriptive diagnostics only; no Messidor threshold was selected."}, "messidor_used_for_selection": False})

    fn_rows = []
    for row, verifier in zip(primary_external, verifier_rows):
        if int(row["actual_referable"]) == 1 and not bool(row["referable_at_frozen_0_10"]):
            fn_rows.append({"image_id": row["image_id"], "actual_grade": row["label"], "primary_probability": row["referable_probability"], "primary_false_negative": True, "verifier_probability": verifier["probability"], "verifier_caught": bool(verifier["decision"]), "verifier_ood_flagged": verifier["ood_flagged"], "quality_status": "not available in classifier-only external artifacts", "lesion_evidence": "not available for full external population", "grad_cam": "not available for full external population"})
    dump("ml/evaluation/referable_v2/false_negative_analysis.json", {"development_oof": false_negative_existing, "external_descriptive": {"dataset": "Messidor-2 authoritative original labels", "primary_threshold": primary_threshold, "false_negative_count": len(fn_rows), "by_grade": {str(grade): sum(int(row["actual_grade"]) == grade for row in fn_rows) for grade in (2, 3, 4)}, "records": fn_rows, "verifier_caught_count": int(verifier_caught.sum()), "both_missed_count": int(both_missed.sum()), "clinical_validation_claim": False}, "limitations": ["Lesion/Grad-CAM/quality fields are not available for the complete classifier external population; no evidence was fabricated."]})

    dump("ml/evaluation/referable_v2/domain_generalization.json", {"status": "EXISTING_DEVELOPMENT_ABLATIONS_REUSED", "source": "ml/evaluation/referable_research/domain_generalization_results.json", "selection_or_external_labels_used": False, "candidates": domain_candidates, "effects": domain_effects, "interpretation": "Domain augmentation candidates were previously compared on development validation only; no Messidor selection occurred and no new uncontrolled sweep was run."})

    verifier_candidates = {
        "schema_version": "retina-nexus-referable-v2-verifier-candidates-v1",
        "candidates": [
            {"name": "RETGUARD DR v1.0.0", "status": "SELECTED_RESEARCH_VERIFIER", "architecture": "EfficientNetV2-M tf_efficientnetv2_m.in21k_ft_in1k", "publication": "RETGUARD retrospective multi-dataset research artifact, v1 submitted to medRxiv", "source": "https://github.com/anchor-neuro/retguard", "training_dataset": ["EyePACS", "DDR", "APTOS", "IDRiD", "DeepDRiD"], "task": "binary referable DR; grades 2/3/4 positive", "classes": 2, "input_resolution": [480, 480], "preprocessing": "circle crop, Lanczos resize, Ben Graham enhancement, LAB CLAHE, circular mask, ImageNet normalization; published 8-view D4 TTA", "checkpoint_source": "https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0", "license": "weights CC BY-NC 4.0; code PolyForm Noncommercial 1.0.0", "sha256": verifier_summary["onnx_sha256"], "requirements": "onnxruntime CPU; official package preprocessing; OOD gate artifact", "limitations": ["Research-only; no regulatory clearance", "No gradability pathway", "Released DR ONNX export failed the documented logit-parity tolerance", "Target hardware/population prospective validation absent", "Commercial rights require separate review"]},
            {"name": "RETFound", "status": "CANDIDATE_NOT_INTEGRATED", "architecture": "retinal foundation model / ViT-Large family", "publication": "A foundation model for generalizable disease detection from retinal images, Nature 2023", "source": "https://github.com/rmaphoh/RETFound", "training_dataset": "Retinal foundation pretraining; exact task head not compatible without project-specific fine-tuning", "task": "foundation features / fine-tuned retinal disease tasks", "classes": "head-dependent", "input_resolution": "implementation-dependent", "preprocessing": "repository-specific", "checkpoint_source": "official repository instructions", "license": "requires source/weight rights review", "sha256": None, "limitations": ["No compatible frozen binary referable head installed", "No checkpoint adopted or evaluated in this cycle"]},
            {"name": "DeepDR Plus", "status": "CANDIDATE_NOT_INTEGRATED", "architecture": "ResNet fundus/time-to-progression system", "publication": "Official project repository", "source": "https://github.com/drpredict/DeepDR_Plus", "training_dataset": "project-specific longitudinal fundus/meta datasets", "task": "personalized time to DR progression, not a standalone referable classifier", "classes": "task-dependent", "input_resolution": "input edge >448 documented", "preprocessing": "repository-specific", "checkpoint_source": "project training instructions", "license": "not established for learned weights in this audit", "sha256": None, "limitations": ["Task mismatch for independent binary referable verification", "No checkpoint adopted or evaluated"]},
        ],
        "selected_verifier": "RETGUARD DR v1.0.0",
        "selection_basis": "Documented binary referable task, official release weights, reproducible ONNX inference, published checksum and model card; research-only role.",
    }
    dump("ml/evaluation/referable_v2/verifier_candidates.json", verifier_candidates)

    verifier_validation = {"status": verifier_summary["status"], "model": verifier_candidates["candidates"][0], "dataset": {"manifest": "ml/evaluation/messidor2/authoritative_external_manifest.json", "images": verifier_summary["requested_count"], "original_only": True, "labels_used_for_selection": False}, "fixed_threshold": verifier_threshold, "metrics": verifier_metrics, "threshold_diagnostics_not_used_for_selection": verifier_diagnostics, "ood_flag_rate": float(np.mean([bool(row["ood_flagged"]) for row in verifier_rows])), "latency": {"mean_ms": float(np.mean([float(row["latency_ms"]) for row in verifier_rows])), "median_ms": float(np.median([float(row["latency_ms"]) for row in verifier_rows]))}, "clinical_validation_claim": False, "production_promoted": False}
    dump("ml/evaluation/referable_v2/verifier_validation.json", verifier_validation)

    fusion = {"schema_version": "retina-nexus-referable-v2-safety-fusion-v1", "status": "RESEARCH_ONLY_FIXED_RULE_ANALYSIS", "learned_fusion_trained": False, "messidor_used_for_rule_selection": False, "primary_threshold": primary_threshold, "verifier_threshold": verifier_threshold, "state_rule": {"DISAGREE": "primary and verifier binary decisions differ", "AGREE_LOW_CONFIDENCE": "decisions agree and either probability is within 0.05 of its frozen threshold", "AGREE_HIGH_CONFIDENCE": "decisions agree and neither probability is within 0.05 of its frozen threshold", "INSUFFICIENT_EVIDENCE": "reserved for missing/failed signals; not silently converted to a prediction"}, "review_rule": "review_recommended = disagreement OR verifier OOD flag; review flag is not a diagnosis and does not rewrite grade", "population": len(fusion_rows), "agreement_count": int((~disagreement).sum()), "disagreement_count": int(disagreement.sum()), "disagreement_rate": float(disagreement.mean()), "primary_metrics": primary_metrics, "verifier_metrics": verifier_metrics, "safety_review_flag_metrics": safety_metrics, "primary_false_negative_count": int(primary_fn.sum()), "verifier_caught_primary_false_negative_count": int(verifier_caught.sum()), "both_missed_false_negative_count": int(both_missed.sum()), "sensitivity_improvement_from_review_flag": float(safety_metrics["sensitivity"] - primary_metrics["sensitivity"]), "specificity_impact": float(safety_metrics["specificity"] - primary_metrics["specificity"]), "rows": fusion_rows, "production_promoted": False, "clinical_validation_claim": False}
    dump("ml/evaluation/referable_v2/fusion_research.json", fusion)

    aptos_05 = aptos_summary["referable_thresholds"]["threshold_0.50"]
    aptos_02 = aptos_summary["referable_thresholds"]["threshold_0.20"]
    idrid_05 = idrid_summary["referable_thresholds"]["threshold_0.50"]
    idrid_02 = idrid_summary["referable_thresholds"]["threshold_0.20"]
    external_results = {"dataset": {"manifest": "ml/evaluation/messidor2/authoritative_external_manifest.json", "original_images": 1748, "label_matched_images": 1744, "label_status": "local/adjudicated; not independently proven official clinical ground truth"}, "threshold_selection_on_messidor": False, "models": {"APTOS_EfficientNet_B0": {"threshold_0.50": aptos_05, "threshold_0.20": aptos_02}, "IDRiD_V3": {"threshold_0.50": idrid_05, "threshold_0.20": idrid_02}, "IDRiD_V3_stage2_binary_head": {"fixed_development_threshold_0.10": primary_metrics}, "RETGUARD_DR": {"fixed_published_threshold_0.204983": verifier_metrics}, "safety_review_flag": safety_metrics}, "clinical_validation_claim": False, "target_status": "TARGET_NOT_YET_DEMONSTRATED"}
    dump("ml/evaluation/referable_v2/messidor_external_results.json", external_results)

    report = f"""# RETINA-NEXUS Referable DR V2 Research Report

Generated: `{generated}`

## Status

**RESEARCH ONLY. Production behavior is unchanged.** Messidor-2 was used only after the primary threshold and independent verifier were frozen. No Messidor threshold, model, augmentation, calibration, or fusion rule was selected from these labels.

The SIH target remains **TARGET NOT YET DEMONSTRATED**. The target requires sensitivity above 90% and specificity above 85% on the same appropriate evaluation.

## Primary binary candidate

The best available binary candidate is the existing IDRiD V3 shared EfficientNet-B0 stage2 referable head. It explicitly predicts referable probability rather than deriving it from severity argmax.

- Positive: IDRiD grade 2/3/4
- Negative: IDRiD grade 0/1
- Data: 406 development records, five-fold leak-safe OOF
- Frozen threshold: `{primary_threshold}`
- Checkpoint SHA: `{research_registry['primary_candidate']['checkpoint_sha256']}`
- Production promoted: `false`

Development OOF: sensitivity `{oof['selected']['sensitivity']:.4f}`, specificity `{oof['selected']['specificity']:.4f}`, ROC-AUC `{oof['roc_auc']:.4f}`, PR-AUC `{oof['pr_auc']:.4f}`, FN `{oof['selected']['fn']}`. OOF ECE is `{oof['ece_10_bins']:.4f}` and Brier score `{oof['brier_score']:.4f}`; raw probabilities are not clinically calibrated.

## Domain generalization

Previously completed development-only domain candidates were reused. The comparison is recorded in `domain_generalization.json`; no uncontrolled new sweep was run. Existing candidate comparisons do not establish external generalization.

## Independent verifier

RETGUARD DR v1.0.0 was selected as the research verifier because it provides a documented binary referable task, official release weights, checksum, preprocessing, and model card. It is licensed for research/noncommercial use and is not integrated into production. [Official repository](https://github.com/anchor-neuro/retguard), [release](https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0).

- Architecture: EfficientNetV2-M
- Input: 480×480 fundus photograph
- Protocol: official preprocessing and 8-view D4 TTA
- ONNX SHA-256: `{verifier_summary['onnx_sha256']}`
- Published threshold: `{verifier_threshold}`
- External inference: `{verifier_summary['successful_count']}` / `{verifier_summary['requested_count']}`, zero failures
- External metrics: sensitivity `{verifier_metrics['sensitivity']:.4f}`, specificity `{verifier_metrics['specificity']:.4f}`, ROC-AUC `{verifier_metrics['roc_auc']:.4f}`, PR-AUC `{verifier_metrics['pr_auc']:.4f}`, FN `{verifier_metrics['fn']}`, FP `{verifier_metrics['fp']}`

The verifier’s released model card reports known research limitations, including absent gradability handling and failed logit-parity tolerance for the released ONNX export. Those limitations prevent production promotion.

## Safety disagreement analysis

The fixed safety layer does not average probabilities and does not invent a diagnosis. It recommends review when primary and verifier decisions disagree or when the verifier OOD flag is raised.

- Agreement: `{int((~disagreement).sum())}` / 1,744
- Disagreement rate: `{disagreement.mean():.4f}`
- Primary false negatives: `{int(primary_fn.sum())}`
- Verifier-caught primary false negatives: `{int(verifier_caught.sum())}`
- Both missed: `{int(both_missed.sum())}`
- Safety review-flag sensitivity: `{safety_metrics['sensitivity']:.4f}`
- Safety review-flag specificity: `{safety_metrics['specificity']:.4f}`
- Sensitivity change: `{safety_metrics['sensitivity'] - primary_metrics['sensitivity']:.4f}`
- Specificity change: `{safety_metrics['specificity'] - primary_metrics['specificity']:.4f}`

These are descriptive external results against local/adjudicated labels, not clinical safety estimates. Lesion evidence was not fused because a complete leak-safe OOF lesion table was unavailable; missing evidence was not fabricated.

## Model comparison on Messidor-2

| Candidate | Frozen operating point | Sensitivity | Specificity | FN | FP |
|---|---:|---:|---:|---:|---:|
| APTOS EfficientNet-B0 | 0.50 | {aptos_05['sensitivity']:.4f} | {aptos_05['specificity']:.4f} | {aptos_05['false_negative']} | {aptos_05['false_positive']} |
| APTOS EfficientNet-B0 | 0.20 | {aptos_02['sensitivity']:.4f} | {aptos_02['specificity']:.4f} | {aptos_02['false_negative']} | {aptos_02['false_positive']} |
| IDRiD V3 severity-derived | 0.50 | {idrid_05['sensitivity']:.4f} | {idrid_05['specificity']:.4f} | {idrid_05['fn']} | {idrid_05['fp']} |
| IDRiD V3 severity-derived | 0.20 | {idrid_02['sensitivity']:.4f} | {idrid_02['specificity']:.4f} | {idrid_02['fn']} | {idrid_02['fp']} |
| IDRiD V3 stage2 binary head | 0.10 | {primary_metrics['sensitivity']:.4f} | {primary_metrics['specificity']:.4f} | {primary_metrics['fn']} | {primary_metrics['fp']} |
| RETGUARD DR verifier | 0.204983 | {verifier_metrics['sensitivity']:.4f} | {verifier_metrics['specificity']:.4f} | {verifier_metrics['fn']} | {verifier_metrics['fp']} |

## Final decision

No model should replace the current production model. The binary head and RETGUARD verifier remain research-only. The next step is independent prospective/target-device validation and a pre-registered safety review protocol before any promotion decision.

## Artifacts

- `research_registry.json`
- `cv_results.json`
- `threshold_sweep.json`
- `false_negative_analysis.json`
- `domain_generalization.json`
- `verifier_candidates.json`
- `verifier_validation.json`
- `fusion_research.json`
- `messidor_external_results.json`
- `oof_predictions.jsonl`
- `primary_binary_messidor/`
- `retguard_messidor/`
"""
    report = report.replace("\u00d7", "x").replace("\u2019", "'")
    (OUT / "final_research_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT.relative_to(ROOT)).replace("\\", "/"), "primary_external": primary_metrics, "verifier_external": verifier_metrics, "disagreement_rate": float(disagreement.mean()), "target": "TARGET_NOT_YET_DEMONSTRATED", "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
