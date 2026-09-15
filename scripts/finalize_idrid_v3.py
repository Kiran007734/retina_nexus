"""Attach the single post-freeze external result to the V3 research artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
EXTERNAL = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v3_domain_robust_zero_shot" / "idrid_v3_messidor2_zero_shot.json"


def load(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def dump(name: str, payload):
    (META / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    external = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    registry = load("idrid_v3_experiment_registry.json")
    selected = load("idrid_v3_selected_candidate.json")
    comparison = load("idrid_v3_comparison.json")
    reference = {
        "status": "COMPLETED_SINGLE_ZERO_SHOT_EVALUATION",
        "report": str(EXTERNAL.relative_to(ROOT)).replace("\\", "/"),
        "dataset": external["dataset"],
        "metrics": external["metrics"],
        "threshold_policy": external["threshold_policy"],
        "checkpoint": external["model"],
        "clinical_validation_claim": False,
        "used_for_selection": False,
        "official_idrid_test_images_opened": 0,
    }
    registry["external_validation"] = reference
    registry["final_conclusion"] = "V3 FAILED — RETURN TO V2/V1"
    registry["final_conclusion_reason"] = "V3 did not improve Messidor-2 referable sensitivity or QWK over the frozen V2 external result; no further external threshold tuning was performed."
    selected["external_validation"] = reference
    selected["final_conclusion"] = registry["final_conclusion"]
    selected["final_conclusion_reason"] = registry["final_conclusion_reason"]
    selected["production_promoted"] = False
    selected["official_test_images_opened"] = 0
    comparison["post_freeze_external_evaluation"] = reference
    comparison["final_conclusion"] = registry["final_conclusion"]
    comparison["final_conclusion_reason"] = registry["final_conclusion_reason"]
    dump("idrid_v3_experiment_registry.json", registry)
    dump("idrid_v3_selected_candidate.json", selected)
    dump("idrid_v3_comparison.json", comparison)
    print(json.dumps({"conclusion": registry["final_conclusion"], "report": reference["report"], "used_for_selection": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
