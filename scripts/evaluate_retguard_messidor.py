"""Evaluate the independently released RETGUARD DR verifier on Messidor originals.

This is a zero-shot descriptive evaluation. The authoritative Messidor labels
are never used for model, threshold, calibration, or fusion selection here.
The verifier is research-only and is kept outside the production model path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_external_manifest.json"
DEFAULT_WEIGHTS = ROOT / "ml" / "weights" / "backup_verifier" / "retguard" / "v1.0.0"
DEFAULT_OUTPUT = ROOT / "ml" / "evaluation" / "referable_v2" / "retguard_messidor"
PRESET_THRESHOLD = 0.204983
MODEL_SHA = "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b"

_PREDICTOR: Any = None


class FastRetguardDR:
    """Equivalent DR inference with bounded ONNX thread pools.

    The released package constructs an ONNX Runtime session with the runtime
    default thread pool. That is correct but oversubscribes the CPU when this
    independent verifier is evaluated in parallel. This adapter preserves the
    released preprocessing, D4 logit averaging, temperature scaling, and OOD
    convention while setting one intra/inter-op thread per worker.
    """

    def __init__(self, weights_dir: str) -> None:
        import onnx
        import onnxruntime as ort
        from retguard.ood import MahalanobisGate
        from retguard.predictor import CAM_SPATIAL_NODE, GAP_OUTPUT_NODE

        self._preprocess = __import__("retguard.preprocess", fromlist=["preprocess_dr"]).preprocess_dr
        self._gate = MahalanobisGate.from_npz(Path(weights_dir) / "ood_gate_dr_v1.0.0.npz")
        model = onnx.load(str(Path(weights_dir) / "retguard_dr_v1.0.0.onnx"))
        self._gap = GAP_OUTPUT_NODE["dr"]
        self._spatial = CAM_SPATIAL_NODE["dr"]
        produced = {tensor for node in model.graph.node for tensor in node.output}
        for tensor_name, shape in ((self._gap, ["batch_size", 1280]), (self._spatial, ["batch_size", 1280, "h", "w"])):
            if tensor_name not in produced:
                raise RuntimeError(f"RETGUARD graph is missing required internal tensor {tensor_name}")
            model.graph.output.append(onnx.helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, shape))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])

    def predict(self, image_path: Path) -> tuple[float, bool, float, bool]:
        import numpy as np
        from retguard.calibrate import apply_temperature, sigmoid
        from retguard.constants import DR_TEMPERATURE
        from retguard.preprocess import load_image

        tensor = self._preprocess(load_image(image_path))
        horizontal_flip = np.flip(tensor, axis=2)
        vertical_flip = np.flip(tensor, axis=1)
        views = np.ascontiguousarray(np.stack([
            tensor,
            np.rot90(tensor, k=1, axes=(1, 2)),
            np.rot90(tensor, k=2, axes=(1, 2)),
            np.rot90(tensor, k=3, axes=(1, 2)),
            horizontal_flip,
            vertical_flip,
            np.rot90(horizontal_flip, k=1, axes=(1, 2)),
            np.rot90(vertical_flip, k=1, axes=(1, 2)),
        ], axis=0))
        logits, features = self._session.run(["logit", self._gap], {"input": views})
        probability = float(sigmoid(apply_temperature(float(logits.mean(axis=0)), DR_TEMPERATURE)))
        ood_score = float(self._gate.score(features[:1])[0])
        return probability, bool(probability >= PRESET_THRESHOLD), ood_score, bool(ood_score > self._gate.threshold)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if payload.get("evaluation_image_source") != "ORIGINAL_ONLY":
        raise RuntimeError("The verifier requires the authoritative original-image manifest")
    if len(records) != 1744:
        raise RuntimeError(f"Expected 1,744 label-matched records, got {len(records)}")
    return records


def init_worker(weights_dir: str) -> None:
    global _PREDICTOR
    _PREDICTOR = FastRetguardDR(weights_dir)


def evaluate_one(record: dict[str, Any]) -> dict[str, Any]:
    if _PREDICTOR is None:
        raise RuntimeError("RETGUARD predictor was not initialized")
    started = time.perf_counter()
    probability, decision, ood_score, ood_flagged = _PREDICTOR.predict(ROOT / record["image_path"])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "image_id": record["image_id"],
        "image_path": record["image_path"],
        "label": int(record["label"]),
        "actual_referable": int(int(record["label"]) >= 2),
        "probability": probability,
        "decision": decision,
        "threshold": PRESET_THRESHOLD,
        "ood_score": ood_score,
        "ood_flagged": ood_flagged,
        "latency_ms": elapsed_ms,
        "tta": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--count", type=int, default=0, help="Bounded smoke count; 0 evaluates all 1,744 records")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).expanduser().resolve()
    onnx_path = weights_dir / "retguard_dr_v1.0.0.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"RETGUARD DR ONNX model is missing: {onnx_path}")
    actual_sha = sha256(onnx_path)
    if actual_sha != MODEL_SHA:
        raise RuntimeError(f"RETGUARD DR SHA-256 mismatch: expected {MODEL_SHA}, got {actual_sha}")

    records = load_records(Path(args.manifest).expanduser().resolve())
    if args.count:
        records = records[: max(1, min(args.count, len(records)))]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context, initializer=init_worker, initargs=(str(weights_dir),)) as pool:
        futures = {pool.submit(evaluate_one, record): record for record in records}
        for index, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # pragma: no cover - failure artifact path
                failures.append({"image_id": record.get("image_id"), "error": f"{type(exc).__name__}: {exc}"})
            if index % 25 == 0 or index == len(records):
                print(json.dumps({"completed": index, "requested": len(records), "failures": len(failures)}), flush=True)
    rows.sort(key=lambda row: row["image_id"])
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "status": "COMPLETE" if not failures and len(rows) == len(records) else "PARTIAL",
        "requested_count": len(records),
        "successful_count": len(rows),
        "failed_count": len(failures),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "manifest": str(Path(args.manifest).resolve().relative_to(ROOT)).replace("\\", "/"),
        "weights_dir": str(weights_dir.relative_to(ROOT)).replace("\\", "/"),
        "onnx_sha256": actual_sha,
        "threshold": PRESET_THRESHOLD,
        "tta": True,
        "messidor_labels_used_for_selection": False,
        "clinical_validation_claim": False,
        "failures": failures,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
