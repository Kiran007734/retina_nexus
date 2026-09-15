"""Preflight the official local Qwen3-VL model; never reports availability without real image inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.qwen_vl_verifier import QwenVLVerifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ml/weights/qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", default="ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png")
    parser.add_argument("--output", default="ml/evaluation/localhost_validation/qwen_preflight.json")
    args = parser.parse_args()
    root = Path(args.model_path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts) if root.exists() else []
    report = {"model_id": QwenVLVerifier.MODEL_ID, "runtime": "transformers-local", "model_path": str(root), "files_present": [str(p.relative_to(root)) for p in files], "required_files": {name: (root / name).exists() for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")}, "status": "NOT_CONFIGURED", "reason": "Complete official model shards are not present; no inference was attempted."}
    if all(report["required_files"].values()):
        image = Path(args.image)
        verifier = QwenVLVerifier(root)
        import asyncio
        result = asyncio.run(verifier.verify(image.read_bytes(), {"quality": "gradability checked", "primary": "not supplied in preflight"}))
        report.update({"status": result.get("status"), "inference": result, "device": verifier.device})
    report["file_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.stat().st_size < 50_000_000}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "AVAILABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
