"""Build research-only referable DR fusion artifacts from frozen predictions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml" / "evaluation" / "fusion_final"
PRIMARY_THRESHOLD = 0.10
VERIFIER_THRESHOLD = 0.204983
CHECKPOINT_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
VERIFIER_SHA = "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b"
THRESHOLDS = tuple(sorted(set([round(value, 2) for value in np.arange(0.05, 0.951, 0.05)] + [PRIMARY_THRESHOLD, VERIFIER_THRESHOLD])))


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_jsonl(relative: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(relative: str, payload: Any) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checksum(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confusion(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((actual == 1) & (predicted == 1))),
        "tn": int(np.sum((actual == 0) & (predicted == 0))),
        "fp": int(np.sum((actual == 0) & (predicted == 1))),
        "fn": int(np.sum((actual == 1) & (predicted == 0))),
    }


def metrics(actual: np.ndarray, score: np.ndarray, predicted: np.ndarray, threshold: Any, score_semantics: str) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    counts = confusion(actual, predicted)
    sensitivity = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    specificity = counts["tn"] / max(1, counts["tn"] + counts["fp"])
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "threshold": threshold,
        "score_semantics": score_semantics,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "fnr": float(1.0 - sensitivity),
        "fpr": float(1.0 - specificity),
        "roc_auc": float(roc_auc_score(actual, score)),
        "pr_auc": float(average_precision_score(actual, score)),
        "support": int(actual.sum()),
        "negative_support": int((actual == 0).sum()),
        "confusion_matrix": counts,
    }


def fixed_metrics(actual: np.ndarray, score: np.ndarray, predicted: np.ndarray, name: str, threshold: Any, semantics: str) -> dict[str, Any]:
    return {"strategy": name, "operating_point": metrics(actual, score, predicted, threshold, semantics)}


def calibration(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(actual[mask].mean()) - float(probability[mask].mean()))
    return {"ece_10_bins": float(ece), "brier_score": float(brier_score_loss(actual, probability))}


def threshold_sweep(actual: np.ndarray, score: np.ndarray, name: str, semantics: str) -> dict[str, Any]:
    rows = [metrics(actual, score, score >= threshold, threshold, semantics) for threshold in THRESHOLDS]
    eligible = [row for row in rows if row["specificity"] >= 0.85]
    selected = max(eligible, key=lambda row: (row["sensitivity"], row["f1"], row["specificity"], -float(row["threshold"]))) if eligible else None
    return {"strategy": name, "thresholds": rows, "selected": selected, "selection_criterion": "highest development sensitivity subject to specificity >= 0.85; ties F1, specificity, then lower threshold", "selection_data": "406-image IDRiD development OOF only", "messidor_used_for_selection": False}


def group_metrics(actual: np.ndarray, primary: np.ndarray, verifier: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(mask.sum()),
        "ground_truth_referable": int(actual[mask].sum()),
        "ground_truth_non_referable": int((actual[mask] == 0).sum()),
        "primary": confusion(actual[mask], primary[mask]),
        "verifier": confusion(actual[mask], verifier[mask]),
    }


def fit_logistic_oof(rows: list[dict[str, Any]], actual: np.ndarray, primary: np.ndarray, verifier: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    disagreement = (primary >= PRIMARY_THRESHOLD).astype(float) != (verifier >= VERIFIER_THRESHOLD).astype(float)
    features = np.column_stack([primary, verifier, disagreement.astype(float)])
    prediction = np.zeros(len(rows), dtype=float)
    coefficients: list[dict[str, Any]] = []
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=20260914)
        model.fit(features[train], actual[train])
        prediction[test] = model.predict_proba(features[test])[:, 1]
        coefficients.append({"held_out_fold": fold, "intercept": float(model.intercept_[0]), "coefficients": [float(value) for value in model.coef_[0]]})
    return prediction, {"method": "logistic_regression", "features": ["primary_probability", "verifier_probability", "binary_disagreement"], "fold_count": len(coefficients), "coefficients_by_held_out_fold": coefficients, "strict_fold_held_out": True, "target_seen_only_in_training_folds": True}


def apply_strategy(name: str, primary: np.ndarray, verifier: np.ndarray, threshold: Any) -> tuple[np.ndarray, np.ndarray, str]:
    primary_decision = primary >= PRIMARY_THRESHOLD
    verifier_decision = verifier >= VERIFIER_THRESHOLD
    if name == "primary_only":
        return primary, primary_decision, "primary probability at frozen development threshold"
    if name == "verifier_only":
        return verifier, verifier_decision, "verifier probability at published fixed threshold"
    if name == "and_rule":
        return np.minimum(primary / PRIMARY_THRESHOLD, verifier / VERIFIER_THRESHOLD), primary_decision & verifier_decision, "minimum of component threshold ratios; binary AND at ratio >= 1"
    if name == "or_rule":
        return np.maximum(primary / PRIMARY_THRESHOLD, verifier / VERIFIER_THRESHOLD), primary_decision | verifier_decision, "maximum of component threshold ratios; binary OR at ratio >= 1"
    if name == "probability_average":
        score = (primary + verifier) / 2.0
    elif name == "weighted_probability_average":
        score = 0.75 * primary + 0.25 * verifier
    elif name == "maximum_probability":
        score = np.maximum(primary, verifier)
    elif name == "minimum_probability":
        score = np.minimum(primary, verifier)
    elif name == "learned_logistic":
        score = primary
    else:
        raise ValueError(name)
    return score, score >= float(threshold), "experimental uncalibrated composite score"


def main() -> int:
    generated = datetime.now(timezone.utc).isoformat()
    primary_rows = read_jsonl("ml/evaluation/referable_v2/oof_predictions.jsonl")
    verifier_rows = read_jsonl("ml/evaluation/fusion_final/retguard_idrid_oof/predictions.jsonl")
    verifier_summary = load("ml/evaluation/fusion_final/retguard_idrid_oof/run_summary.json")
    if len(primary_rows) != 406 or len(verifier_rows) != 406 or verifier_summary["status"] != "COMPLETE":
        raise RuntimeError("Fusion requires complete 406-record primary and verifier outputs")
    primary_by_id = {row["image_id"]: row for row in primary_rows}
    verifier_by_id = {row["image_id"]: row for row in verifier_rows}
    if set(primary_by_id) != set(verifier_by_id):
        raise RuntimeError("Primary and verifier IDs are not aligned")
    ordered_ids = sorted(primary_by_id)
    primary_probability = np.asarray([float(primary_by_id[image_id]["referable_probability"]) for image_id in ordered_ids], dtype=float)
    verifier_probability = np.asarray([float(verifier_by_id[image_id]["probability"]) for image_id in ordered_ids], dtype=float)
    actual = np.asarray([int(primary_by_id[image_id]["actual_referable"]) for image_id in ordered_ids], dtype=int)
    folds = [int(primary_by_id[image_id]["fold"]) for image_id in ordered_ids]
    primary_decision = primary_probability >= PRIMARY_THRESHOLD
    verifier_decision = verifier_probability >= VERIFIER_THRESHOLD
    primary_fn = (actual == 1) & ~primary_decision
    verifier_fn = (actual == 1) & ~verifier_decision
    disagreement = primary_decision != verifier_decision

    table = []
    for index, image_id in enumerate(ordered_ids):
        table.append({
            "image_id": image_id,
            "fold": folds[index],
            "primary_probability": float(primary_probability[index]),
            "primary_referable": bool(primary_decision[index]),
            "verifier_probability": float(verifier_probability[index]),
            "verifier_referable": bool(verifier_decision[index]),
            "ground_truth": int(actual[index]),
            "primary_correct": bool(primary_decision[index] == actual[index]),
            "verifier_correct": bool(verifier_decision[index] == actual[index]),
            "disagreement": bool(disagreement[index]),
            "false_negative_by_primary": bool(primary_fn[index]),
            "false_negative_by_verifier": bool(verifier_fn[index]),
            "both_false_negative": bool(primary_fn[index] & verifier_fn[index]),
            "verifier_ood_flagged": bool(verifier_by_id[image_id]["ood_flagged"]),
        })
    dump("ml/evaluation/fusion_final/fusion_prediction_table.json", {"schema_version": "retina-nexus-fusion-prediction-table-v1", "generated_at_utc": generated, "records": table, "count": len(table), "id_alignment_verified": True, "official_idrid_test_images_opened": 0, "messidor_used_for_selection": False, "verifier_dataset_overlap_risk": "RETGUARD's published training datasets include IDRiD; sample-level training membership is not available, so this is not an independent IDRiD estimate."})

    disagreement_analysis = {
        "disagreement_rate": float(disagreement.mean()),
        "disagreement_count": int(disagreement.sum()),
        "primary_referable_verifier_non_referable": group_metrics(actual, primary_decision, verifier_decision, primary_decision & ~verifier_decision),
        "primary_non_referable_verifier_referable": group_metrics(actual, primary_decision, verifier_decision, ~primary_decision & verifier_decision),
        "primary_false_negative_count": int(primary_fn.sum()),
        "verifier_caught_primary_false_negative_count": int((primary_fn & verifier_decision).sum()),
        "verifier_false_negative_count": int(verifier_fn.sum()),
        "both_false_negative_count": int((primary_fn & verifier_fn).sum()),
        "verifier_positive_primary_false_negative_ground_truth_referable": int((primary_fn & verifier_decision & (actual == 1)).sum()),
        "verifier_positive_primary_false_negative_ground_truth_non_referable": int((primary_fn & verifier_decision & (actual == 0)).sum()),
        "interpretation": "Verifier-positive primary-negative cases are complementary only as research evidence; their false-positive burden is measured and no diagnosis is rewritten.",
        "thresholds": {"primary": PRIMARY_THRESHOLD, "verifier": VERIFIER_THRESHOLD},
        "verifier_dataset_overlap_risk": True,
        "verifier_overlap_note": "RETGUARD's published training datasets include IDRiD; sample-level membership is unknown.",
    }
    dump("ml/evaluation/fusion_final/disagreement_analysis.json", disagreement_analysis)

    strategies: dict[str, dict[str, Any]] = {}
    strategies["primary_only"] = fixed_metrics(actual, primary_probability, primary_decision, "primary_only", PRIMARY_THRESHOLD, "primary probability")
    strategies["verifier_only"] = fixed_metrics(actual, verifier_probability, verifier_decision, "verifier_only", VERIFIER_THRESHOLD, "verifier probability")
    and_score, and_decision, and_semantics = apply_strategy("and_rule", primary_probability, verifier_probability, 1.0)
    or_score, or_decision, or_semantics = apply_strategy("or_rule", primary_probability, verifier_probability, 1.0)
    strategies["and_rule"] = fixed_metrics(actual, and_score, and_decision, "and_rule", "component thresholds", and_semantics)
    strategies["or_rule"] = fixed_metrics(actual, or_score, or_decision, "or_rule", "component thresholds", or_semantics)
    sweeps = {}
    for name in ("probability_average", "weighted_probability_average", "maximum_probability", "minimum_probability"):
        score, _, semantics = apply_strategy(name, primary_probability, verifier_probability, 0.5)
        sweeps[name] = threshold_sweep(actual, score, name, semantics)
        if sweeps[name]["selected"] is not None:
            selected = sweeps[name]["selected"]
            strategies[name] = {"strategy": name, "operating_point": selected, "selection_source": "development OOF only"}

    simple_sensitivity = max(row["operating_point"]["sensitivity"] for row in strategies.values() if row["strategy"] != "primary_only")
    primary_sensitivity = strategies["primary_only"]["operating_point"]["sensitivity"]
    simple_trigger = bool(simple_sensitivity >= primary_sensitivity + 0.01 and any(row["operating_point"]["specificity"] >= 0.85 for row in strategies.values() if row["strategy"] != "primary_only"))
    learned: dict[str, Any] = {"status": "NOT_RUN", "reason": "Simple fusion did not meet the predeclared trigger."}
    learned_probability = None
    if simple_trigger:
        learned_probability, learned_meta = fit_logistic_oof(primary_rows, actual, primary_probability, verifier_probability)
        learned_sweep = threshold_sweep(actual, learned_probability, "learned_logistic", "strict fold-held-out logistic probability")
        learned = {"status": "COMPLETED", "trigger": "simple fusion sensitivity >= primary sensitivity + 0.01 and specificity >= 0.85", "model": learned_meta, "threshold_analysis": learned_sweep, "calibration": calibration(actual, learned_probability), "verifier_dataset_overlap_risk": True}
        if learned_sweep["selected"] is not None:
            strategies["learned_logistic"] = {"strategy": "learned_logistic", "operating_point": learned_sweep["selected"], "selection_source": "development OOF only"}
    baselines = {"status": "COMPLETED", "development_records": 406, "primary_threshold": PRIMARY_THRESHOLD, "verifier_threshold": VERIFIER_THRESHOLD, "strategy_definitions": {"and_rule": "primary decision AND verifier decision at their frozen thresholds", "or_rule": "primary decision OR verifier decision at their frozen thresholds", "probability_average": "(primary_probability + verifier_probability) / 2; uncalibrated research score", "weighted_probability_average": "0.75 * primary_probability + 0.25 * verifier_probability; fixed research weight, not learned or clinically calibrated", "maximum_probability": "max(primary_probability, verifier_probability); uncalibrated research score", "minimum_probability": "min(primary_probability, verifier_probability); uncalibrated research score"}, "simple_strategies": strategies, "threshold_sweeps": sweeps, "blind_probability_averaging_not_promoted": True, "learned_fusion": learned, "verifier_dataset_overlap_risk": True, "messidor_used_for_selection": False, "production_promoted": False, "clinical_validation_claim": False}
    dump("ml/evaluation/fusion_final/fusion_baselines.json", baselines)

    threshold_output = {"data": "IDRiD 406-image development OOF only", "messidor_used_for_selection": False, "primary": {"threshold": PRIMARY_THRESHOLD, "operating_point": strategies["primary_only"]["operating_point"]}, "verifier": {"threshold": VERIFIER_THRESHOLD, "operating_point": strategies["verifier_only"]["operating_point"]}, "simple_fusion": sweeps, "learned_fusion": learned}
    dump("ml/evaluation/fusion_final/fusion_thresholds.json", threshold_output)

    candidate_names = [name for name in strategies if name not in {"primary_only", "verifier_only"}]
    best_fusion_name = max(candidate_names, key=lambda name: (strategies[name]["operating_point"]["sensitivity"], strategies[name]["operating_point"]["f1"], strategies[name]["operating_point"]["specificity"])) if candidate_names else None
    best_fusion = strategies[best_fusion_name]["operating_point"] if best_fusion_name else None
    dump("ml/evaluation/fusion_final/fusion_cv_results.json", {"status": "COMPLETED", "development_records": 406, "primary": strategies["primary_only"]["operating_point"], "verifier": strategies["verifier_only"]["operating_point"], "best_simple_or_learned_fusion": {"strategy": best_fusion_name, "operating_point": best_fusion}, "all_operating_points": strategies, "selection_rule": "research description only; highest development sensitivity subject to specificity >= 0.85 where applicable, never external selection", "verifier_dataset_overlap_risk": True, "verifier_overlap_note": "RETGUARD's published training datasets include IDRiD; sample-level membership is unknown.", "target_status": "TARGET_NOT_YET_DEMONSTRATED"})

    best_fn = int(best_fusion["confusion_matrix"]["fn"]) if best_fusion else int(strategies["primary_only"]["operating_point"]["confusion_matrix"]["fn"])
    primary_fn_count = int(strategies["primary_only"]["operating_point"]["confusion_matrix"]["fn"])
    primary_fp_count = int(strategies["primary_only"]["operating_point"]["confusion_matrix"]["fp"])
    best_fp = int(best_fusion["confusion_matrix"]["fp"]) if best_fusion else primary_fp_count
    dump("ml/evaluation/fusion_final/false_negative_analysis.json", {"primary_fn": primary_fn_count, "verifier_fn": int(strategies["verifier_only"]["operating_point"]["confusion_matrix"]["fn"]), "fusion_fn": best_fn, "both_fn": int((primary_fn & verifier_fn).sum()), "primary_false_negatives_detected_after_fusion": primary_fn_count - best_fn, "fn_reduction_percent": float((primary_fn_count - best_fn) / max(1, primary_fn_count) * 100.0), "fp_increase": best_fp - primary_fp_count, "fp_increase_percent": float((best_fp - primary_fp_count) / max(1, primary_fp_count) * 100.0), "net_change": {"sensitivity": float(best_fusion["sensitivity"] - strategies["primary_only"]["operating_point"]["sensitivity"]) if best_fusion else 0.0, "specificity": float(best_fusion["specificity"] - strategies["primary_only"]["operating_point"]["specificity"]) if best_fusion else 0.0}, "development_only": True, "clinical_validation_claim": False})

    evidence = {
        "status": "SUPPORTING_EVIDENCE_NOT_DISEASE_PROBABILITY",
        "disease_severity_controller": "primary IDRiD severity classifier; evidence does not rewrite grade",
        "referable_fusion_controller": "research-only classifier/verifier analysis; no production rule changed",
        "signals": {
            "primary_lesion_model": {"status": "AVAILABLE_AS_RESEARCH_EVIDENCE", "artifact": "ml/datasets/metadata/idrid/idrid_lesion_final_report.json", "training_images": 54, "official_test_images_opened_for_development": 0, "clinical_validation_claim": False},
            "idrid_lesion_model": {"status": "AVAILABLE_AS_RESEARCH_EVIDENCE", "artifact": "ml/datasets/metadata/idrid/idrid_lesion_final_report.json", "complete_406_oof_alignment": False, "reason": "Lesion annotations and model coverage do not provide a complete leak-safe 406-image disease-fusion table."},
            "r2_v2_vessel": {"status": "SUPPORTING_ONLY", "artifact": "ml/evaluation/drive/r2-v2-evaluation.json", "dataset": "DRIVE", "aligned_to_idrid_406": False},
            "drive_unet_research": {"status": "SUPPORTING_ONLY", "dataset": "DRIVE", "aligned_to_idrid_406": False},
            "localization": {"status": "PARTIAL_RESEARCH_EVIDENCE", "artifact": "ml/datasets/metadata/idrid/idrid_localization_final_report.json", "disease_probability": False},
            "grad_cam": {"status": "EXPLANATION_ONLY", "disease_probability": False},
            "image_quality": {"status": "RELIABILITY_SIGNAL_ONLY", "disease_probability": False},
            "uncertainty": {"status": "RELIABILITY_SIGNAL_ONLY", "disease_probability": False},
        },
        "fusion_rule": "supporting, contradictory, consistent, or unavailable evidence modifies review priority only; no lesion/vessel/localization signal is converted into a disease probability",
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    dump("ml/evaluation/fusion_final/evidence_fusion_results.json", evidence)

    ood = np.asarray([bool(verifier_by_id[image_id]["ood_flagged"]) for image_id in ordered_ids], dtype=bool)
    review_flag = disagreement | ood
    review_metrics = metrics(actual, review_flag.astype(float), review_flag, "review flag", "disagreement OR verifier OOD")
    trust_states = {"TRUSTED": int((~review_flag).sum()), "UNCERTAIN": int(review_flag.sum()), "UNRELIABLE": 0}
    retinaguard = {"status": "RESEARCH_ONLY_REVIEW_ANALYSIS", "rule": "UNCERTAIN when classifier/verifier disagree or verifier OOD is flagged; no grade rewrite", "trust_state_counts": trust_states, "review_flag_count": int(review_flag.sum()), "primary_false_negative_capture_count": int((primary_fn & review_flag).sum()), "primary_false_negative_capture_rate": float((primary_fn & review_flag).sum() / max(1, primary_fn.sum())), "review_flag_metrics": review_metrics, "primary_reference": strategies["primary_only"]["operating_point"], "production_promoted": False, "clinical_validation_claim": False}
    dump("ml/evaluation/fusion_final/retinaguard_fusion_results.json", retinaguard)

    primary_external = read_jsonl("ml/evaluation/referable_v2/primary_binary_messidor/predictions.jsonl")
    verifier_external = read_jsonl("ml/evaluation/referable_v2/retguard_messidor/predictions.jsonl")
    primary_external_by_id = {row["image_id"]: row for row in primary_external}
    verifier_external_by_id = {row["image_id"]: row for row in verifier_external}
    external_ids = sorted(set(primary_external_by_id) & set(verifier_external_by_id))
    ext_actual = np.asarray([int(primary_external_by_id[image_id]["actual_referable"]) for image_id in external_ids], dtype=int)
    ext_primary = np.asarray([float(primary_external_by_id[image_id]["referable_probability"]) for image_id in external_ids], dtype=float)
    ext_verifier = np.asarray([float(verifier_external_by_id[image_id]["probability"]) for image_id in external_ids], dtype=float)
    external_results: dict[str, Any] = {"dataset": "Messidor-2 authoritative original label-matched records", "records": len(external_ids), "threshold_selection_on_messidor": False, "labels_used_for_selection": False, "models": {}}
    for name in ("primary_only", "verifier_only", "and_rule", "or_rule"):
        score, prediction, semantics = apply_strategy(name, ext_primary, ext_verifier, 1.0)
        threshold = PRIMARY_THRESHOLD if name == "primary_only" else VERIFIER_THRESHOLD if name == "verifier_only" else "component thresholds"
        external_results["models"][name] = metrics(ext_actual, score, prediction, threshold, semantics)
    for name in ("probability_average", "weighted_probability_average", "maximum_probability", "minimum_probability"):
        if name in strategies:
            threshold = strategies[name]["operating_point"]["threshold"]
            score, prediction, semantics = apply_strategy(name, ext_primary, ext_verifier, threshold)
            external_results["models"][name] = metrics(ext_actual, score, prediction, threshold, semantics)
    if learned_probability is not None and "learned_logistic" in strategies:
        from sklearn.linear_model import LogisticRegression

        development_features = np.column_stack([primary_probability, verifier_probability, disagreement.astype(float)])
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=20260914).fit(development_features, actual)
        external_features = np.column_stack([ext_primary, ext_verifier, (ext_primary >= PRIMARY_THRESHOLD) != (ext_verifier >= VERIFIER_THRESHOLD)])
        ext_score = model.predict_proba(external_features)[:, 1]
        ext_threshold = strategies["learned_logistic"]["operating_point"]["threshold"]
        external_results["models"]["learned_logistic"] = metrics(ext_actual, ext_score, ext_score >= ext_threshold, ext_threshold, "development-fitted logistic applied without external tuning")
    external_results["clinical_validation_claim"] = False
    external_results["target_status"] = "TARGET_NOT_YET_DEMONSTRATED"
    dump("ml/evaluation/fusion_final/messidor_fusion_results.json", external_results)

    external_best = external_results["models"].get(best_fusion_name) if best_fusion_name else None
    report_lines = [
        "# RETINA-NEXUS Fusion Research Report",
        "",
        f"Generated: `{generated}`",
        "",
        "## Governance",
        "",
        "Research-only analysis. No checkpoint, production route, threshold, or RetinaGuard production rule was modified. Messidor labels were not used for fitting, threshold selection, calibration, or fusion selection. Official IDRiD test images opened: `0`.",
        "",
        "## Development results",
        "",
        f"Primary IDRiD V3: sensitivity `{strategies['primary_only']['operating_point']['sensitivity']:.4f}`, specificity `{strategies['primary_only']['operating_point']['specificity']:.4f}`, F1 `{strategies['primary_only']['operating_point']['f1']:.4f}`, FN `{strategies['primary_only']['operating_point']['confusion_matrix']['fn']}` at frozen threshold `{PRIMARY_THRESHOLD}`.",
        f"RETGUARD verifier: sensitivity `{strategies['verifier_only']['operating_point']['sensitivity']:.4f}`, specificity `{strategies['verifier_only']['operating_point']['specificity']:.4f}`, F1 `{strategies['verifier_only']['operating_point']['f1']:.4f}`, FN `{strategies['verifier_only']['operating_point']['confusion_matrix']['fn']}` at published threshold `{VERIFIER_THRESHOLD}`.",
        f"Disagreement: `{disagreement_analysis['disagreement_count']}` / 406 (`{disagreement_analysis['disagreement_rate']:.4f}`). Verifier-caught primary FNs: `{disagreement_analysis['verifier_caught_primary_false_negative_count']}`. Both-model FNs: `{disagreement_analysis['both_false_negative_count']}`.",
        f"Best research fusion operating point: `{best_fusion_name}` with sensitivity `{best_fusion['sensitivity']:.4f}`, specificity `{best_fusion['specificity']:.4f}`, FN `{best_fusion['confusion_matrix']['fn']}`, FP `{best_fusion['confusion_matrix']['fp']}`." if best_fusion else "No fusion operating point met the research criteria.",
        f"Learned fusion: `{learned['status']}`. It is strict fold-held-out logistic regression only and is not promoted. Calibration ECE/Brier: `{learned.get('calibration', {}).get('ece_10_bins', 'not run')}` / `{learned.get('calibration', {}).get('brier_score', 'not run')}`.",
        "",
        "## False-negative and evidence interpretation",
        "",
        f"Relative to primary, the selected research fusion changes sensitivity by `{(best_fusion['sensitivity'] - strategies['primary_only']['operating_point']['sensitivity']):.4f}` and specificity by `{(best_fusion['specificity'] - strategies['primary_only']['operating_point']['specificity']):.4f}`." if best_fusion else "No selected fusion delta is reported.",
        "The verifier checkpoint is architecture/checkpoint-independent from the primary, but its published training datasets include IDRiD; sample-level membership is unknown. Therefore the 406-image verifier and learned-fusion results are overlap-risk development evidence, not independent validation.",
        "Lesion, vessel, localization, Grad-CAM, quality, and uncertainty signals remain supporting or reliability evidence. They are not converted into disease probabilities and do not rewrite severity. A complete leak-safe 406-image lesion evidence table was unavailable, so lesion fusion was not fabricated.",
        f"Research RetinaGuard review flag (disagreement OR verifier OOD) sensitivity: `{review_metrics['sensitivity']:.4f}`; specificity: `{review_metrics['specificity']:.4f}`. This is a review-priority analysis, not a clinical trust guarantee.",
        "",
        "## External evaluation",
        "",
        f"Messidor application used `{len(external_ids)}` cached label-matched records exactly once after development selection was frozen. Frozen `{best_fusion_name}` result: sensitivity `{external_best['sensitivity']:.4f}`, specificity `{external_best['specificity']:.4f}`, FN `{external_best['confusion_matrix']['fn']}`, FP `{external_best['confusion_matrix']['fp']}`. Results are descriptive only; see `messidor_fusion_results.json`. SIH >90% sensitivity and >85% specificity is **TARGET_NOT_YET_DEMONSTRATED** because the verifier has known IDRiD training-overlap risk and the external labels are not clinical validation.",
        "",
        "## Final decision",
        "",
        "Production promotion is not justified. The primary remains the severity controller; the verifier and evidence modules remain research-only complementary signals pending independent prospective and target-device validation.",
        "",
        "## Provenance",
        "",
        f"Primary checkpoint SHA-256: `{CHECKPOINT_SHA}`.",
        f"RETGUARD ONNX SHA-256: `{VERIFIER_SHA}`.",
        "",
        "Artifacts: `fusion_prediction_table.json`, `disagreement_analysis.json`, `fusion_baselines.json`, `fusion_thresholds.json`, `fusion_cv_results.json`, `false_negative_analysis.json`, `evidence_fusion_results.json`, `retinaguard_fusion_results.json`, and `messidor_fusion_results.json`.",
    ]
    (OUT / "final_fusion_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT.relative_to(ROOT)).replace("\\", "/"), "best_fusion": best_fusion_name, "learned_fusion": learned["status"], "messidor_records": len(external_ids), "target_status": "TARGET_NOT_YET_DEMONSTRATED", "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
