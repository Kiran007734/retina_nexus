"""Evaluate registered DR checkpoints under one reproducible benchmark protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import ReferableDRMapping, build_classifier  # noqa: E402
from ml.evaluation.benchmark import benchmark_matrix, compare_benchmark_results  # noqa: E402
from ml.training.classification_dataset import FundusClassificationDataset  # noqa: E402
from ml.training.losses import build_ordinal_loss  # noqa: E402
from scripts.dataset_common import DatasetError, get_definition, raw_path_for, split_directory  # noqa: E402
from scripts.train_classifier import load_manifest, select_device, make_transforms, run_epoch  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DR classifier checkpoints on a governance split")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--raw-dir")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH", help="Repeat for each checkpoint; all are evaluated on the same split")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--referable-min-grade", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="ml/experiments/benchmark_results.json")
    return parser


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise DatasetError(f"Checkpoint must use NAME=PATH syntax: {spec}")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise DatasetError(f"Checkpoint does not exist: {checkpoint}")
    return name.strip(), checkpoint


def main() -> int:
    args = build_parser().parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        print("BENCHMARK SETUP ERROR: PyTorch/torchvision could not be imported. Install backend/requirements-ml.txt; no weights were assumed.", file=sys.stderr)
        print(f"Original error: {exc}", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("BENCHMARK SETUP ERROR: --batch-size must be positive", file=sys.stderr)
        return 2
    try:
        definition = get_definition(args.dataset)
        raw_root = raw_path_for(definition, args.raw_dir)
        manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else split_directory(args.dataset) / "splits.json"
        payload = load_manifest(manifest_path, raw_root)
        checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoint]
        checkpoint_metadata = [(name, path, torch.load(path, map_location="cpu", weights_only=False)) for name, path in checkpoint_specs]
        input_sizes = {int(metadata.get("model_config", {}).get("input_size", 224)) for _, _, metadata in checkpoint_metadata}
        if len(input_sizes) != 1:
            raise DatasetError("All benchmark checkpoints must use the same input size for a fair comparison")
        transform, _ = make_transforms(input_sizes.pop())
        dataset = FundusClassificationDataset(manifest_path, raw_root, args.split, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        device = select_device(torch, args.device)
        mapping = ReferableDRMapping(name=f"grade_{args.referable_min_grade}_or_worse", referable_grades=tuple(range(args.referable_min_grade, 5)))
        results: list[dict[str, Any]] = []
        for name, checkpoint_path, checkpoint in checkpoint_metadata:
            try:
                model_config = checkpoint.get("model_config", {})
                backbone = model_config.get("backbone")
                if not backbone:
                    raise ValueError("checkpoint has no model_config.backbone")
                ordinal_mode = bool(model_config.get("ordinal_mode", False))
                model = build_classifier(backbone=backbone, num_classes=5, pretrained=False, ordinal_mode=ordinal_mode).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                evaluation_loss = build_ordinal_loss("plain", [int(record["label"]) for record in dataset.records]) if ordinal_mode else torch.nn.CrossEntropyLoss()
                metrics = run_epoch(model, loader, None, evaluation_loss, mapping, ordinal_mode, device, torch, None, False)
                results.append({
                    "name": name, "checkpoint": str(checkpoint_path),
                    "model_version": checkpoint.get("model_version") or checkpoint.get("artifact", {}).get("model_version", "unversioned"),
                    "backbone": backbone, "ordinal_mode": ordinal_mode,
                    "dataset": args.dataset, "dataset_version": checkpoint.get("dataset_version", "unspecified"),
                    "split": args.split, "metrics": metrics,
                    "clinical_validation_claim": False,
                })
            except Exception as exc:
                print(f"BENCHMARK SETUP ERROR for {name}: {exc}", file=sys.stderr)
                return 2
        report = {
            "benchmark_matrix": benchmark_matrix(),
            "protocol": {"dataset": args.dataset, "manifest": str(manifest_path), "raw_dir": str(raw_root), "split": args.split, "device": str(device), "sample_count": len(dataset), "referable_mapping": mapping.to_dict()},
            "results": results,
            "ranking_for_reporting_only": compare_benchmark_results(results),
            "note": "Models are compared under the same split and protocol. This report does not declare a universally superior architecture or make a clinical validation claim.",
        }
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"Benchmark report written: {output}")
        for item in compare_benchmark_results(results):
            print(f"{item['name']}: f1={item['metrics']['f1']:.4f} accuracy={item['metrics']['accuracy']:.4f}")
        return 0
    except (DatasetError, ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"BENCHMARK SETUP ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
