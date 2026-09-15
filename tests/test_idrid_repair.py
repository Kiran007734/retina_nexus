"""Regression guards for the bounded IDRiD quick-repair cycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MAX_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "max_quality" / "final" / "checkpoint_best.pt"
APTOS_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"


def test_repair_cycle_retained_frozen_model_without_official_test_access():
    status = json.loads((META / "repair_final_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "NO_MEANINGFUL_IMPROVEMENT_FOUND_CURRENT_MODEL_RETAINED"
    assert status["selected_repaired_candidate"] is None
    assert status["official_test_images_opened"] == 0
    assert status["production_promoted"] is False
    assert status["checkpoint_sha256"] == "075d7e41714cac28cfd0b60eba39fc50b81f7a8788d743004cd9dda702e1cbab"


def test_repair_experiments_are_recorded_and_do_not_create_replacement_checkpoint():
    comparison = json.loads((META / "repair_comparison.json").read_text(encoding="utf-8"))
    assert set(comparison["experiments"]) == {"b0_crop_384_sampler", "b0_crop_384_focal"}
    assert comparison["selected_repaired_candidate"] is None
    assert comparison["official_test_images_opened"] == 0
    assert comparison["production_promoted"] is False
    assert not (MAX_CHECKPOINT.parent.parent.parent / "repair" / "final" / "checkpoint_best.pt").exists()


def test_repair_cycle_preserves_protected_checkpoint_hashes():
    max_digest = hashlib.sha256(MAX_CHECKPOINT.read_bytes()).hexdigest()
    aptos_digest = hashlib.sha256(APTOS_CHECKPOINT.read_bytes()).hexdigest()
    assert max_digest == "075d7e41714cac28cfd0b60eba39fc50b81f7a8788d743004cd9dda702e1cbab"
    assert aptos_digest == "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
