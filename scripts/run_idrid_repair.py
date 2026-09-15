"""Quick, bounded IDRiD max-quality repair cycle.

Runs only two current-384px experiments: weighted sampling and class-balanced
focal loss.  Results and any new checkpoint are isolated below
``ml/weights/classifiers/idrid/research/repair``.  The max-quality checkpoint,
all prior checkpoints, production configuration, and official test set are
never modified or opened.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import scripts.run_idrid_max_quality as base  # noqa: E402


REPAIR_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "repair"
META_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"
CURRENT_COMPARISON = META_ROOT / "idrid_max_quality_experiment_comparison.json"
CURRENT_CV = META_ROOT / "idrid_max_quality_cv_report.json"
MAX_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "max_quality" / "final" / "checkpoint_best.pt"
APTOS = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
APTOS_SHA = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run the bounded current-resolution IDRiD repair cycle")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    if sha256(APTOS) != APTOS_SHA:
        raise RuntimeError("APTOS production checkpoint changed; refusing repair cycle")
    records = base.load_idrid_development()
    if sha256(MAX_CHECKPOINT) != "075d7e41714cac28cfd0b60eba39fc50b81f7a8788d743004cd9dda702e1cbab":
        raise RuntimeError("Max-quality checkpoint changed; refusing repair cycle")
    # Redirect only the imported runner's research output constants.  This
    # prevents any repair operation from touching max_quality artifacts.
    base.CV_ROOT = REPAIR_ROOT / "cv"
    base.OUTPUT_ROOT = REPAIR_ROOT
    # ``final_train`` uses the imported module's FINAL_ROOT global.  Redirect
    # it explicitly as well; the frozen max-quality checkpoint is immutable.
    base.FINAL_ROOT = REPAIR_ROOT / "final"
    experiments = ("b0_crop_384_sampler", "b0_crop_384_focal")
    results = {}
    for name in experiments:
        cached_cv = REPAIR_ROOT / f"{name}_cv.json"
        if cached_cv.exists():
            results[name] = json.loads(cached_cv.read_text(encoding="utf-8"))
        else:
            results[name] = base.train_candidate(name, records, args, torch)
    current = json.loads(CURRENT_COMPARISON.read_text(encoding="utf-8"))
    current_candidate = current["results"]["b0_crop_384"]
    current_summary = current_candidate["summary"]
    eligible = [value for value in results.values() if value["summary"]["quadratic_weighted_kappa"]["mean"] >= current_summary["quadratic_weighted_kappa"]["mean"] + 0.01 and value["summary"]["f1"]["mean"] >= current_summary["f1"]["mean"] and value["summary"]["accuracy"]["mean"] >= current_summary["accuracy"]["mean"] - 0.01]
    selected = max(eligible, key=lambda value: (value["summary"]["quadratic_weighted_kappa"]["mean"], value["summary"]["f1"]["mean"], value["summary"]["accuracy"]["mean"]))["candidate"] if eligible else None
    repair_comparison = {"schema_version": "idrid-repair-comparison-1", "generated_at": datetime.now(timezone.utc).isoformat(), "current_model": {"model_version": "idrid-max-quality-20260912-v1", "checkpoint": str(MAX_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": sha256(MAX_CHECKPOINT), "summary": current_summary}, "experiments": results, "selected_repaired_candidate": selected, "selection_rule": "QWK mean must improve by >=0.01; macro-F1 must not decrease; accuracy must not decrease by more than 0.01", "official_test_images_opened": 0, "production_promoted": False}
    dump(META_ROOT / "repair_comparison.json", repair_comparison)
    final_manifest = None
    if selected:
        rows = base.pooled_oof(selected, args.folds)
        metrics = base._metrics([row["actual"] for row in rows], [row["probabilities"] for row in rows])
        dump(META_ROOT / "repair_cv_report.json", {"candidate": selected, "metrics": metrics, "fold_summary": results[selected]["summary"], "out_of_fold": True, "fold_count": args.folds, "official_test_images_opened": 0, "production_promoted": False})
        dump(META_ROOT / "repair_error_analysis.json", base.error_analysis(rows))
        final_manifest = base.final_train(selected, records, args, torch)
        repair_checkpoint = REPAIR_ROOT / "final" / "checkpoint_best.pt"
        repair_payload = torch.load(repair_checkpoint, map_location="cpu", weights_only=False)
        repair_payload["model_version"] = "idrid-repair-20260912-v1"
        repair_payload["production_promoted"] = False
        repair_payload["official_test_images_opened"] = 0
        torch.save(repair_payload, repair_checkpoint)
        final_manifest.update({"model_version": "idrid-repair-20260912-v1", "research_status": "FROZEN_RESEARCH_ONLY", "production_promoted": False, "official_test_images_opened": 0, "repair_comparison": "ml/datasets/metadata/idrid/repair_comparison.json", "repair_cv_report": "ml/datasets/metadata/idrid/repair_cv_report.json", "checkpoint_sha256": sha256(REPAIR_ROOT / "final" / "checkpoint_best.pt")})
        (REPAIR_ROOT / "final" / "model_manifest.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        dump(META_ROOT / "repair_cv_report.json", {"status": "NO_MEANINGFUL_IMPROVEMENT_FOUND", "current_model_retained": True, "current_cv_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_cv_report.json", "official_test_images_opened": 0})
        dump(META_ROOT / "repair_error_analysis.json", {"status": "CURRENT_MODEL_RETAINED", "source": "ml/datasets/metadata/idrid/idrid_max_quality_error_analysis.json", "official_test_images_opened": 0})
    dump(META_ROOT / "repair_final_status.json", {"status": "REPAIRED_MODEL_FROZEN" if selected else "NO_MEANINGFUL_IMPROVEMENT_FOUND_CURRENT_MODEL_RETAINED", "selected_repaired_candidate": selected, "current_model": "idrid-max-quality-20260912-v1", "new_model_manifest": str((REPAIR_ROOT / "final" / "model_manifest.json").relative_to(ROOT)).replace("\\", "/") if final_manifest else None, "checkpoint_sha256": final_manifest.get("checkpoint_sha256") if final_manifest else sha256(MAX_CHECKPOINT), "official_test_images_opened": 0, "production_promoted": False, "external_datasets_used": False})
    print(json.dumps({"selected_repaired_candidate": selected, "current_model_retained": selected is None, "official_test_images_opened": 0, "production_promoted": False, "comparison": "ml/datasets/metadata/idrid/repair_comparison.json"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
