"""Acquire and verify the published pretrained retinal-lesion segmentor.

The script never creates a model artifact locally. It downloads the exact
``model.safetensors`` file from the public Hugging Face model repository and
records its checksum, architecture, class mapping, source, and license in a
local manifest. Weights remain ignored by Git.

Examples:
    python scripts/acquire_lesion_model.py
    python scripts/acquire_lesion_model.py --verify-only
    python scripts/acquire_lesion_model.py --revision <commit-or-tag>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d"
DEFAULT_VERSION = "fundus-lesions-unet-seresnext50-all-v1"
DEFAULT_TARGET = ROOT / "ml" / "weights" / "lesion_segmentation" / DEFAULT_VERSION
EXPECTED_CONFIG = {
    "arch": "unet",
    "encoder": "seresnext50_32x4d",
    "n_classes": 5,
}
CLASSES = {
    "0": "background",
    "1": "cotton_wool_spot",
    "2": "exudate",
    "3": "hemorrhage",
    "4": "microaneurysm",
}


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
    weights_path = target / "model.safetensors"
    config_path = target / "config.json"

    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("huggingface_hub is required. Install backend/requirements-ml.txt before acquiring the lesion model.") from exc
        for filename in ("model.safetensors", "config.json"):
            downloaded = Path(
                hf_hub_download(
                    repo_id=args.repository,
                    filename=filename,
                    revision=args.revision,
                    local_dir=str(target),
                )
            )
            if downloaded.resolve() != (target / filename).resolve() and downloaded.is_file():
                (target / filename).write_bytes(downloaded.read_bytes())

    if not weights_path.is_file():
        raise SystemExit(f"Lesion model weights are missing: {weights_path}. Re-run without --verify-only.")
    if not config_path.is_file():
        raise SystemExit(f"Lesion model config is missing: {config_path}. Re-run without --verify-only.")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read the downloaded lesion model config: {exc}") from exc
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise SystemExit(f"Downloaded model config mismatch for {key!r}: expected {expected!r}, got {config.get(key)!r}")

    # Record the immutable Hub commit whenever the network is available. The
    # requested ref remains in the manifest so the acquisition is reproducible
    # even when an offline verification cannot resolve it.
    resolved_revision = args.revision
    try:
        from huggingface_hub import model_info

        resolved_revision = model_info(args.repository, revision=args.revision).sha or args.revision
    except Exception as exc:  # pragma: no cover - depends on network/auth state
        print(f"Warning: could not resolve the immutable Hugging Face revision: {exc}", file=sys.stderr)

    checksum = sha256(weights_path)
    manifest = {
        "manifest_version": "1.0.0",
        "model_version": args.version,
        "model_name": "fundus-lesions-toolkit U-Net SE-ResNeXt-50",
        "model_type": "semantic_segmentation",
        "repository": args.repository,
        "source_url": f"https://huggingface.co/{args.repository}",
        "source_code_url": "https://github.com/ClementPla/fundus-lesions-toolkit",
        "revision_requested": args.revision,
        "revision": resolved_revision,
        "license": "MIT (declared by the model repository and model card)",
        "architecture": "U-Net with SE-ResNeXt-50 32x4d encoder",
        "config": config,
        "classes": CLASSES,
        "trained_on": ["IDRiD", "DDR", "FGADR", "MESSIDOR", "RETLES"],
        "input_resolution": 1024,
        "checkpoint": str(weights_path.relative_to(ROOT)).replace("/", "\\"),
        "checkpoint_sha256": checksum,
        "artifact_status": "MODEL_DOWNLOADED",
        "clinical_validation_claim": False,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "note": "Research checkpoint used as supporting retinal lesion evidence. It is not a clinical diagnosis and has not been validated by RETINA-NEXUS.",
    }
    manifest_path = target / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "weights": str(weights_path), "config": str(config_path), "manifest": str(manifest_path), "sha256": checksum}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
