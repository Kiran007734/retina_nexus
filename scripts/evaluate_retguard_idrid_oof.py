"""Evaluate the frozen RETGUARD verifier on the governed 406-image IDRiD development set."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_retguard_messidor import FastRetguardDR, sha256  # noqa: E402

DEFAULT_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
DEFAULT_OOF = ROOT / "ml" / "evaluation" / "referable_v2" / "oof_predictions.jsonl"
DEFAULT_WEIGHTS = ROOT / "ml" / "weights" / "backup_verifier" / "retguard" / "v1.0.0"
DEFAULT_OUTPUT = ROOT / "ml" / "evaluation" / "fusion_final" / "retguard_idrid_oof"
MODEL_SHA = "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b"
THRESHOLD = 0.204983

_PREDICTOR: Any = None


def init_worker(weights_dir: str) -> None:
    global _PREDICTOR
    _PREDICTOR = FastRetguardDR(weights_dir)


def evaluate_chunk(records: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    if _PREDICTOR is None:
        raise RuntimeError("RETGUARD predictor was not initialized")
    import numpy as np
    from retguard.calibrate import apply_temperature, sigmoid
    from retguard.constants import DR_TEMPERATURE
    from retguard.preprocess import load_image

    started = time.perf_counter()
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        views: list[np.ndarray] = []
        for record in batch_records:
            tensor = _PREDICTOR._preprocess(load_image(ROOT / record["image_path"]))
            horizontal_flip = np.flip(tensor, axis=2)
            vertical_flip = np.flip(tensor, axis=1)
            views.extend([
                tensor,
                np.rot90(tensor, k=1, axes=(1, 2)),
                np.rot90(tensor, k=2, axes=(1, 2)),
                np.rot90(tensor, k=3, axes=(1, 2)),
                horizontal_flip,
                vertical_flip,
                np.rot90(horizontal_flip, k=1, axes=(1, 2)),
                np.rot90(vertical_flip, k=1, axes=(1, 2)),
            ])
        stacked = np.ascontiguousarray(np.stack(views, axis=0))
        logits, features = _PREDICTOR._session.run(["logit", _PREDICTOR._gap], {"input": stacked})
        logits = logits.reshape(len(batch_records), 8).mean(axis=1)
        probabilities = np.asarray(sigmoid(apply_temperature(logits, DR_TEMPERATURE)), dtype=float)
        ood_scores = _PREDICTOR._gate.score(features[::8])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for index, record in enumerate(batch_records):
            probability = float(probabilities[index])
            ood_score = float(ood_scores[index])
            output.append({
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "actual_grade": int(record["actual_grade"]),
                "actual_referable": int(record["actual_referable"]),
                "probability": probability,
                "decision": bool(probability >= THRESHOLD),
                "threshold": THRESHOLD,
                "ood_score": ood_score,
                "ood_flagged": bool(ood_score > _PREDICTOR._gate.threshold),
                "latency_ms": elapsed_ms / len(batch_records),
                "tta": True,
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--oof", default=str(DEFAULT_OOF))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--count", type=int, default=0, help="Bounded smoke count; 0 evaluates all 406 records")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).expanduser().resolve()
    onnx_path = weights_dir / "retguard_dr_v1.0.0.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"RETGUARD DR ONNX model is missing: {onnx_path}")
    actual_sha = sha256(onnx_path)
    if actual_sha != MODEL_SHA:
        raise RuntimeError(f"RETGUARD DR SHA-256 mismatch: expected {MODEL_SHA}, got {actual_sha}")

    manifest = json.loads(Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8"))
    if manifest.get("official_test_images_opened") != 0:
        raise RuntimeError("The governed IDRiD development manifest reports official test access")
    records = {str(row["image_id"]): row for row in manifest.get("records", [])}
    oof_rows = [json.loads(line) for line in Path(args.oof).expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 406 or len(oof_rows) != 406 or len({row["image_id"] for row in oof_rows}) != 406:
        raise RuntimeError("Expected exactly 406 governed development records")
    evaluation_records = []
    for row in oof_rows:
        record = records[str(row["image_id"])]
        evaluation_records.append({
            "image_id": str(row["image_id"]),
            "image_path": "ml/datasets/raw/idrid/" + str(record["image"]),
            "actual_grade": int(row["actual_grade"]),
            "actual_referable": int(row["actual_referable"]),
        })
    if set(row["image_id"] for row in evaluation_records) != set(records):
        raise RuntimeError("OOF and development manifest IDs do not align")
    if args.count:
        evaluation_records = evaluation_records[: max(1, min(args.count, len(evaluation_records)))]

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    chunks = [evaluation_records[index : index + max(1, args.batch_size)] for index in range(0, len(evaluation_records), max(1, args.batch_size))]
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context, initializer=init_worker, initargs=(str(weights_dir),)) as pool:
        futures = {pool.submit(evaluate_chunk, chunk, max(1, args.batch_size)): chunk for chunk in chunks}
        for index, future in enumerate(as_completed(futures), start=1):
            chunk = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:  # pragma: no cover - failure artifact path
                failures.append({"image_ids": [record["image_id"] for record in chunk], "error": f"{type(exc).__name__}: {exc}"})
            if index % 5 == 0 or index == len(chunks):
                print(json.dumps({"completed_records": min(index * max(1, args.batch_size), len(evaluation_records)), "requested": len(evaluation_records), "failures": len(failures)}), flush=True)
    rows.sort(key=lambda row: row["image_id"])
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "status": "COMPLETE" if not failures and len(rows) == len(evaluation_records) else "PARTIAL",
        "requested_count": len(evaluation_records),
        "successful_count": len(rows),
        "failed_count": len(failures),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "manifest": str(Path(args.manifest).resolve().relative_to(ROOT)).replace("\\", "/"),
        "oof_source": str(Path(args.oof).resolve().relative_to(ROOT)).replace("\\", "/"),
        "weights_dir": str(weights_dir.relative_to(ROOT)).replace("\\", "/"),
        "onnx_sha256": actual_sha,
        "threshold": THRESHOLD,
        "tta": True,
        "official_idrid_test_images_opened": 0,
        "messidor_labels_used": False,
        "clinical_validation_claim": False,
        "failures": failures,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
