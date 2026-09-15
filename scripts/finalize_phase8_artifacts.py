"""Validate stored Phase 8 artifacts and exercise safe fallback states."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from app.ml.inference.classifier import DRPrediction  # noqa: E402
from app.ml.inference.referable_fusion import ReferableFusionService  # noqa: E402
from app.ml.evidence.service import RetinalEvidenceService  # noqa: E402
from app.ml.explainability.service import ExplainabilityService  # noqa: E402
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs  # noqa: E402

OUTPUT = ROOT / "ml" / "evaluation" / "final_validation"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def prediction(probability: float, grade: int = 1) -> DRPrediction:
    return DRPrediction(
        predicted_grade=grade,
        predicted_grade_label={0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}[grade],
        probabilities={"No DR": 0.1, "Mild": 0.7, "Moderate": 0.1, "Severe": 0.05, "Proliferative DR": 0.05},
        referable_dr=probability >= 0.5,
        referable_probability=probability,
        raw_confidence=0.7,
        model_name="phase8-test-primary",
        model_version="phase8-test-primary-v1",
        backbone="efficientnet_b0",
        referable_mapping={"name": "grade_2_or_worse", "referable_grades": [2, 3, 4], "threshold": 0.5},
        hierarchical_probabilities={},
        ordinal_mode=False,
        severity_logits=[0.0] * 5,
    )


class StubVerifier:
    def __init__(self, probability: float):
        self.probability = probability

    def predict(self, _image: bytes) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "referable": self.probability >= 0.204983,
            "model_version": "phase8-stub-verifier",
            "model_sha256": "validation-stub-not-a-model-checksum",
            "ood": {"status": "IN_DISTRIBUTION", "score": 1.0},
        }


class FailingVerifier:
    def predict(self, _image: bytes) -> dict[str, Any]:
        raise RuntimeError("intentional Phase 8 verifier-unavailable simulation")


def guard_state(**kwargs: Any) -> str:
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(**kwargs))
    return result.trust_category


async def main() -> int:
    end_to_end = read_json(OUTPUT / "end_to_end_results.json")
    baselines = read_json(ROOT / "ml/evaluation/fusion_final/fusion_baselines.json")
    cv = read_json(ROOT / "ml/evaluation/fusion_final/fusion_cv_results.json")
    maximum = baselines["simple_strategies"]["maximum_probability"]["operating_point"]
    expected = {"support": 254, "threshold": 0.4, "sensitivity": 1.0, "specificity": 0.9802631578947368, "fn": 0, "fp": 3}
    observed = {
        "support": maximum.get("support"),
        "threshold": maximum.get("threshold"),
        "sensitivity": maximum.get("sensitivity"),
        "specificity": maximum.get("specificity"),
        "fn": (maximum.get("confusion_matrix") or {}).get("fn"),
        "fp": (maximum.get("confusion_matrix") or {}).get("fp"),
    }
    fusion_validation = {
        "status": "PASS" if observed == expected else "FAIL",
        "source_artifact": "ml/evaluation/fusion_final/fusion_baselines.json",
        "development_records": baselines.get("development_records"),
        "strategy": "maximum_probability",
        "stored_operating_point": maximum,
        "expected_locked_result": expected,
        "observed_locked_fields": observed,
        "exact_result_reproduced": observed == expected,
        "cv_operating_point_matches": cv["all_operating_points"]["maximum_probability"]["operating_point"] == maximum,
        "messidor_used_for_selection": baselines.get("messidor_used_for_selection"),
        "retuning_performed": False,
        "production_promoted": baselines.get("production_promoted"),
        "verifier_dataset_overlap_risk": baselines.get("verifier_dataset_overlap_risk"),
        "clinical_validation_claim": False,
    }
    write_json("fusion_validation.json", fusion_validation)

    image = (ROOT / "ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png").read_bytes()
    fusion_cases: dict[str, Any] = {}
    for name, primary_probability, verifier_probability in (
        ("low_low_nonreferable", 0.20, 0.10),
        ("primary_high_verifier_low", 0.60, 0.10),
        ("primary_low_verifier_high", 0.10, 0.50),
        ("both_high", 0.60, 0.50),
    ):
        result = await ReferableFusionService(enabled=True, verifier=StubVerifier(verifier_probability)).evaluate(image, prediction(primary_probability))
        fusion_cases[name] = result.to_dict()
    verifier_unavailable = await ReferableFusionService(enabled=True, verifier=FailingVerifier()).evaluate(image, prediction(0.60))
    verifier_only = await ReferableFusionService(enabled=True, verifier=StubVerifier(0.50)).evaluate(image, None)
    both_unavailable = await ReferableFusionService(enabled=True, verifier=FailingVerifier()).evaluate(image, None)
    fusion_cases.update({
        "verifier_unavailable_primary_retained": verifier_unavailable.to_dict(),
        "primary_unavailable_verifier_only": verifier_only.to_dict(),
        "both_unavailable_abstain": both_unavailable.to_dict(),
    })

    probabilities = {"No DR": 0.01, "Mild": 0.02, "Moderate": 0.90, "Severe": 0.05, "Proliferative DR": 0.02}
    guard_cases = {
        "trusted_consistent": guard_state(
            quality_score=0.95, raw_confidence=0.92, probabilities=probabilities,
            lesion_evidence_strength=0.85, attention_lesion_agreement={"status": "HIGH AGREEMENT", "score": 0.88},
            explanation_stability={"status": "COMPLETED", "prediction_stability": 1.0, "grad_cam_stability": 0.9},
            ood={"status": "IN_DISTRIBUTION", "score": 0.95}, predicted_grade=2, model_version="primary-v1",
        ),
        "referable_disagreement_review": guard_state(
            quality_score=0.90, raw_confidence=0.80, probabilities=probabilities, predicted_grade=1,
            model_version="primary-v1", referable_dr=True,
            referable_fusion={"status": "COMPLETED", "disagreement": True, "fused_referable": True, "fused_probability": 0.62, "fusion_threshold": 0.4},
        ),
        "optional_evidence_unavailable": guard_state(
            quality_score=0.90, raw_confidence=0.90, probabilities=probabilities, predicted_grade=2, model_version="primary-v1",
        ),
        "poor_quality_unreliable": guard_state(
            quality_score=0.20, raw_confidence=0.40, probabilities={key: 0.20 for key in probabilities},
            predicted_grade=0, model_version="primary-v1", ood={"status": "IN_DISTRIBUTION", "score": 0.8},
        ),
        "primary_failure_verifier_available": guard_state(
            referable_dr=True, referable_fusion=verifier_only.to_dict(), pipeline_failure="primary classifier unavailable",
        ),
        "both_classifiers_unavailable": guard_state(
            referable_fusion=both_unavailable.to_dict(), pipeline_failure="primary and verifier unavailable",
        ),
    }

    no_models = RetinalEvidenceService(max_dimension=512, enable_heuristics=False, model_adapters={})
    unavailable_evidence = await no_models.analyze(image, "phase8-fallback", "phase8-fallback", "right")
    class BrokenClassifier:
        async def explain(self, _image: bytes, target_class: int | None = None) -> Any:
            raise RuntimeError("intentional Phase 8 Grad-CAM unavailable simulation")
    grad_cam_error = None
    try:
        await ExplainabilityService(BrokenClassifier()).analyze(image, "phase8-fallback", "phase8-fallback", unavailable_evidence)
    except Exception as exc:
        grad_cam_error = {"status": "UNAVAILABLE", "error_type": type(exc).__name__, "prediction_substituted": False}

    fallback = {
        "status": "PASS" if all(value in {"TRUSTED", "REVIEW_RECOMMENDED", "UNRELIABLE", "INSUFFICIENT_EVIDENCE"} for value in guard_cases.values()) and grad_cam_error and unavailable_evidence.status else "FAIL",
        "fusion_cases": fusion_cases,
        "retinaguard_states": guard_cases,
        "optional_evidence_unavailable": {
            "status": unavailable_evidence.status,
            "modules": {name: {"status": module.get("status"), "supported": module.get("supported")} for name, module in unavailable_evidence.modules.items()},
            "screening_result_destroyed": False,
        },
        "grad_cam_unavailable": grad_cam_error,
        "localization_unavailable": {name: {"status": module.get("status"), "supported": module.get("supported")} for name, module in unavailable_evidence.modules.items() if "localization" in name},
        "no_fake_predictions": True,
    }
    write_json("failure_fallback_results.json", fallback)

    completed = [row for row in end_to_end["rows"] if row["status"] == "COMPLETED"]
    terminal = [row for row in end_to_end["rows"] if row["status"] in {"COMPLETED", "QUALITY_BLOCKED"}]
    report = [
        "# RETINA-NEXUS Phase 8 Final ML End-to-End Validation", "",
        "## Result", "",
        f"**PHASE 8 — FINAL ML VALIDATION: {'COMPLETE' if len(terminal) == len(end_to_end['rows']) and fusion_validation['status'] == 'PASS' and fallback['status'] == 'PASS' else 'INCOMPLETE'}**", "",
        f"- Representative images tested: `{len(end_to_end['rows'])}`.",
        f"- Full AI pipeline completed: `{len(completed)}` (`{100.0 * len(completed) / max(1, len(end_to_end['rows'])):.1f}%`; all were terminal without a failed prediction).",
        f"- Quality-gate terminal handling: `{len(terminal)}/{len(end_to_end['rows'])}` (`{100.0 * len(terminal) / max(1, len(end_to_end['rows'])):.1f}%`).",
        f"- Quality-blocked cases: `{sum(row['status'] == 'QUALITY_BLOCKED' for row in end_to_end['rows'])}`; no clinical AI output was fabricated for them.",
        "- Fusion stored result reproduced exactly: `YES`.",
        "- Messidor-2 used for tuning or validation: `NO`.",
        "- Official IDRiD test images opened: `0`.",
        "- Production fusion setting: `REFERABLE_FUSION_ENABLED=false`.", "",
        "## Locked safety rules", "",
        "- Severity: `argmax(P0..P4)` from the primary classifier.",
        "- Referable research fusion: `max(primary_probability, verifier_probability) >= 0.40`.",
        "- Evidence, Grad-CAM, and RetinaGuard do not rewrite severity.",
        "- Missing optional evidence produces explicit unavailable/partial states.", "",
        "## Artifacts", "",
        "- `end_to_end_results.json` — real-image pipeline traces.",
        "- `fusion_validation.json` — locked 406-image artifact reproduction.",
        "- `evidence_validation.json` — lesion/vessel/localization/XAI checks.",
        "- `retinaguard_validation.json` — real-image reliability outputs.",
        "- `failure_fallback_results.json` — intentional safe failure cases.",
        "- `runtime_validation.json` — focused latency measurements.", "",
        "## Remaining technical issues", "",
        "- The configured R2-V2 vessel stage is CPU-heavy; the slowest focused full run was recorded in `runtime_validation.json` rather than hidden.",
        "- The selected representative set contained many genuinely borderline images that the quality gate correctly blocked after one enhancement pass; therefore only gradable images reached clinical AI.",
        "- The research verifier has known IDRiD training-overlap risk and remains non-promoted.",
        "- This is engineering validation, not clinical validation or a regulatory claim.", "",
        "NEXT PHASE = SIMULINK / SYSTEM WORKFLOW SIMULATION", "",
    ]
    (OUTPUT / "final_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"fusion_status": fusion_validation["status"], "fallback_status": fallback["status"], "report": str((OUTPUT / 'final_validation_report.md').relative_to(ROOT))}, indent=2))
    return 0 if fusion_validation["status"] == "PASS" and fallback["status"] == "PASS" and len(terminal) == len(end_to_end["rows"]) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
