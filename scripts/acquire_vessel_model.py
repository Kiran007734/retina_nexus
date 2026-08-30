"""Acquire and verify the published R2-V2 ``bv`` vessel segmentor.

The script uses the official Hugging Face Hub download mechanism for the
published safetensors checkpoint, configuration, and inference source files.
The large artifact remains ignored by Git; a local manifest records its
provenance and checksum. No vessel mask or model is fabricated when access or
verification fails.

Examples:
    python scripts/acquire_vessel_model.py
    python scripts/acquire_vessel_model.py --verify-only
    python scripts/acquire_vessel_model.py --revision <commit-or-tag>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "j-morano/R2-V2"
DEFAULT_VERSION = "r2-v2-bv-2025"
DEFAULT_TARGET = ROOT / "ml" / "weights" / "vessel_segmentation" / DEFAULT_VERSION
FILES = ("bv.safetensors", "bv_config.json", "model.py", "preprocessing.py", "transformations.py")
EXPECTED_CONFIG = {
    "model": "RRWNet",
    "in_channels": 6,
    "out_channels": 3,
    "base_channels": 64,
    "num_iterations": 5,
}
CLASSES = {"0": "artery", "1": "vein", "2": "blood_vessels"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target if args.target.is_absolute() else (ROOT / args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("huggingface_hub is required. Install backend/requirements-ml.txt before acquiring the vessel model.") from exc
        try:
            for filename in FILES:
                downloaded = Path(hf_hub_download(repo_id=args.repository, filename=filename, revision=args.revision, local_dir=str(target)))
                expected = target / filename
                if downloaded.resolve() != expected.resolve() and downloaded.is_file():
                    expected.write_bytes(downloaded.read_bytes())
        except Exception as exc:
            raise SystemExit(f"Could not acquire the published vessel model from {args.repository}@{args.revision}: {exc}") from exc

    missing = [filename for filename in FILES if not (target / filename).is_file()]
    if missing:
        raise SystemExit(f"Vessel model artifact is incomplete under {target}; missing {', '.join(missing)}. Re-run without --verify-only.")

    try:
        config = json.loads((target / "bv_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read the published R2-V2 configuration: {exc}") from exc
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise SystemExit(f"R2-V2 configuration mismatch for {key!r}: expected {expected!r}, got {config.get(key)!r}")

    resolved_revision = args.revision
    try:
        from huggingface_hub import model_info

        resolved_revision = model_info(args.repository, revision=args.revision).sha or args.revision
    except Exception as exc:  # pragma: no cover - network/authentication state is environment-specific
        print(f"Warning: could not resolve the immutable Hugging Face revision: {exc}", file=sys.stderr)

    weights_path = target / "bv.safetensors"
    manifest = {
        "manifest_version": "1.0.0",
        "model_version": args.version,
        "model_name": "R2-V2 RRWNet bv vessel segmentor",
        "model_type": "binary_vessel_segmentation",
        "repository": args.repository,
        "source_url": f"https://huggingface.co/{args.repository}",
        "source_code_url": "https://github.com/j-morano/R2-V2",
        "revision_requested": args.revision,
        "revision": resolved_revision,
        "license": "CC BY 4.0 (declared by the model repository and model card)",
        "architecture": "RRWNet (R2-V2 bv variant)",
        "config": config,
        "classes": CLASSES,
        "training_dataset_provenance": "Unified_Fundus (as stated in the published bv_config.json; constituent datasets are not asserted here)",
        "input_configuration": {"width": 1408, "channels": 6, "padding_multiple": 32, "preprocessing_source": "published preprocessing.py"},
        "output_configuration": {"channels": CLASSES, "vessel_channel": 2, "activation": "sigmoid"},
        "source_files": list(FILES[1:]),
        "checkpoint": str(weights_path.relative_to(ROOT)).replace("/", "\\"),
        "checkpoint_sha256": sha256(weights_path),
        "artifact_status": "MODEL_DOWNLOADED",
        "clinical_validation_claim": False,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "note": "Research model used for supporting retinal vessel evidence. RETINA-NEXUS has verified real inference but has not evaluated segmentation accuracy without authorized DRIVE ground truth.",
    }
    manifest_path = target / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "weights": str(weights_path), "manifest": str(manifest_path), "sha256": manifest["checkpoint_sha256"], "revision": resolved_revision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
