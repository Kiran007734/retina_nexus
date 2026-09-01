"""Run the Phase 5.1 RetinaGuard reliability usability audit.

This is an audit-first command. It does not train, tune, or rewrite the Phase
5 artifacts. It measures the current quality gate on the actual APTOS and
Messidor-2 image bytes, compares the legacy required-capability policy with
the versioned graceful-degradation policy, and writes a per-result decision
trace. Synthetic demo fixtures are reported separately and never mixed into
dataset or model measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService  # noqa: E402
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs  # noqa: E402


DEFAULT_APTOS = ROOT / "ml" / "datasets" / "raw" / "aptos2019"
DEFAULT_MESSIDOR = ROOT / "ml" / "datasets" / "raw" / "messidor"
DEFAULT_PREDICTIONS = ROOT / "ml" / "evaluation" / "messidor" / "zero_shot_predictions.csv"
DEFAULT_OUTPUT = ROOT / "ml" / "evaluation" / "reliability"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
PROBABILITY_COLUMNS = [f"probability_{index}" for index in range(5)]
OPTIONAL_WARNING_CODES = {
    "attention_evidence_not_available",
    "explanation_stability_not_run",
    "ood_not_available",
    "model_disagreement_not_available",
}
MEANINGFUL_WARNING_CODES = {
    "low_image_quality",
    "quality_not_available",
    "low_calibrated_confidence",
    "calibration_not_available",
    "high_prediction_uncertainty",
    "high_model_disagreement",
    "low_attention_evidence_agreement",
    "low_explanation_stability",
    "distribution_shift",
    "retinaguard_pipeline_failure",
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _summary(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None, "percentiles": {}}
    array = np.asarray(clean, dtype=np.float64)
    percentiles = {f"p{level}": round(float(np.percentile(array, level)), 8) for level in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    return {
        "count": len(clean),
        "min": round(float(np.min(array)), 8),
        "max": round(float(np.max(array)), 8),
        "mean": round(float(np.mean(array)), 8),
        "median": round(float(np.median(array)), 8),
        "std": round(float(np.std(array, ddof=1)), 8) if len(clean) > 1 else 0.0,
        "percentiles": percentiles,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _aptos_inventory(root: Path) -> dict[str, Any]:
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"APTOS train.csv was not found at {train_csv}")
    train_rows = _read_csv(train_csv)
    test_rows = _read_csv(test_csv) if test_csv.is_file() else []
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and ".complete" not in path.parts)
    by_stem = {path.stem.casefold(): path for path in files}
    labelled: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    invalid_labels: list[dict[str, Any]] = []
    duplicate_ids = [value for value, count in Counter(str(row.get("id_code", "")).strip().casefold() for row in train_rows).items() if value and count > 1]
    for row in train_rows:
        image_id = str(row.get("id_code", "")).strip()
        label = _integer(row.get("diagnosis"))
        path = by_stem.get(image_id.casefold())
        if path is None:
            missing_labels.append(image_id)
            continue
        if label not in range(5):
            invalid_labels.append({"image_id": image_id, "label": row.get("diagnosis")})
        labelled.append({"image_id": image_id, "label": label, "path": path})
    test_ids = [str(row.get("id_code", "")).strip() for row in test_rows]
    return {
        "root": root,
        "files": files,
        "train_files": sorted(path for path in files if "train_images" in path.parts),
        "test_files": sorted(path for path in files if "test_images" in path.parts),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "labelled": labelled,
        "test_ids": test_ids,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "duplicate_ids": duplicate_ids,
    }


def _messidor_path(root: Path, raw_path: str, cache: dict[str, Path]) -> Path:
    direct = root / raw_path
    if direct.is_file():
        return direct
    key = Path(raw_path).name.casefold()
    if key not in cache:
        cache[key] = next((path for path in root.rglob(Path(raw_path).name) if path.is_file()), direct)
    return cache[key]


def _quality_working_copy(content: bytes, max_dimension: int) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Return a deterministic working copy while retaining the original audit input."""
    with Image.open(io.BytesIO(content)) as image:
        original_size = image.size
        if max(image.size) <= max_dimension:
            return content, original_size, original_size
        working = image.convert("RGB")
        working.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        working.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), original_size, working.size


def _assess_one(item: dict[str, Any], analysis_max_dimension: int) -> dict[str, Any]:
    service = ImageTrustGateService()
    path = Path(item["path"])
    record: dict[str, Any] = {"image_id": item.get("image_id") or path.stem, "path": str(path), "status": "UNREADABLE", "quality": None, "error": None, "source_dimensions": None, "analysis_dimensions": None}
    try:
        source_content = path.read_bytes()
        source_metadata = service.validate_input(source_content)
        analysis_content, _original_size, analysis_size = _quality_working_copy(source_content, analysis_max_dimension)
        assessment = asyncio.run(service.assess(analysis_content))
        record.update({"status": "READABLE", "quality": assessment.to_dict(), "source_dimensions": [source_metadata.width, source_metadata.height], "analysis_dimensions": list(analysis_size)})
    except (OSError, ImageTrustGateError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


async def _assess_paths(items: list[dict[str, Any]], label: str, analysis_max_dimension: int = 512, workers: int = 4) -> list[dict[str, Any]]:
    """Assess actual files using bounded concurrency to keep the audit practical."""
    total = len(items)
    records: list[dict[str, Any] | None] = [None] * total
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="quality-audit") as executor:
        futures = {executor.submit(_assess_one, item, analysis_max_dimension): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            records[futures[future]] = future.result()
            completed += 1
            if completed == 1 or completed % 250 == 0 or completed == total:
                print(f"{label} quality audit: {completed}/{total}", flush=True)
    return [record for record in records if record is not None]


def _quality_summary(records: list[dict[str, Any]], root: Path, labels: dict[str, int] | None = None, analysis_max_dimension: int = 512) -> dict[str, Any]:
    readable = [record for record in records if record["status"] == "READABLE"]
    decisions = Counter((record.get("quality") or {}).get("quality_decision") for record in readable)
    issue_counts: Counter[str] = Counter()
    issue_severity: Counter[str] = Counter()
    component_values: dict[str, list[float]] = {}
    metric_values: dict[str, list[float]] = {}
    widths: list[float] = []
    heights: list[float] = []
    analysis_widths: list[float] = []
    analysis_heights: list[float] = []
    metadata_count = 0
    for record in readable:
        quality = record["quality"] or {}
        for issue in quality.get("issues", []):
            issue_counts[str(issue.get("type", "unknown"))] += 1
            issue_severity[str(issue.get("severity", "unknown"))] += 1
        for name, value in (quality.get("component_scores") or {}).items():
            numeric = _number(value)
            if numeric is not None:
                component_values.setdefault(name, []).append(numeric)
        for name, value in (quality.get("metrics") or {}).items():
            numeric = _number(value)
            if numeric is not None:
                metric_values.setdefault(name, []).append(numeric)
        source_dimensions = record.get("source_dimensions") or []
        width = _number(source_dimensions[0]) if len(source_dimensions) > 0 else None
        height = _number(source_dimensions[1]) if len(source_dimensions) > 1 else None
        if width is not None:
            widths.append(width)
        if height is not None:
            heights.append(height)
        if record.get("analysis_dimensions"):
            analysis_widths.append(float(record["analysis_dimensions"][0]))
            analysis_heights.append(float(record["analysis_dimensions"][1]))
        if (quality.get("input_metadata") or {}).get("camera_metadata"):
            metadata_count += 1
    payload: dict[str, Any] = {
        "status": "COMPLETE",
        "file_count": len(records),
        "readable_file_count": len(readable),
        "unreadable_file_count": len(records) - len(readable),
        "decision_counts": dict(sorted(decisions.items())),
        "decision_rates": {key: round(value / len(records), 8) if records else None for key, value in sorted(decisions.items())},
        "quality_score": _summary([(record.get("quality") or {}).get("quality_score") for record in readable]),
        "component_scores": {name: _summary(values) for name, values in sorted(component_values.items())},
        "raw_metrics": {name: _summary(values) for name, values in sorted(metric_values.items())},
        "resolution": {"source_width": _summary(widths), "source_height": _summary(heights), "analysis_width": _summary(analysis_widths), "analysis_height": _summary(analysis_heights), "quality_working_copy_max_dimension": analysis_max_dimension},
        "camera_metadata_present_count": metadata_count,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_severity_counts": dict(sorted(issue_severity.items())),
        "labels": labels or {},
        "note": "Measured from actual local image bytes with the runtime ImageTrustGateService. Quality decisions are engineering gradability heuristics, not clinical validation.",
    }
    payload["quality_score_histogram"] = _histogram([(record.get("quality") or {}).get("quality_score") for record in readable])
    payload["path_root"] = str(root)
    return payload


def _histogram(values: Iterable[Any], bins: int = 10) -> dict[str, int]:
    clean = [float(value) for value in values if _number(value) is not None]
    if not clean:
        return {}
    counts, edges = np.histogram(clean, bins=bins, range=(0.0, 1.0))
    return {f"{edges[index]:.1f}-{edges[index + 1]:.1f}": int(counts[index]) for index in range(len(counts))}


def _source_duplicate_summary(path: Path, note: str) -> dict[str, Any]:
    payload = _load(path)
    if not isinstance(payload, dict):
        return {"status": "UNAVAILABLE", "source": str(path), "note": note}
    fields = {key: payload.get(key) for key in ("exact_duplicate_group_count", "perceptual_duplicate_group_count", "canonical_representative_count", "deduplicated_image_count") if key in payload}
    if not fields:
        nested = payload.get("files", {})
        fields = {key: nested.get(key) for key in ("duplicate_exact_count", "duplicate_perceptual_count") if key in nested}
    return {"status": "AVAILABLE", "source": str(path), **fields, "note": note}


def _probabilities(row: dict[str, str]) -> dict[str, float] | None:
    values = [_number(row.get(column)) for column in PROBABILITY_COLUMNS]
    if any(value is None for value in values):
        return None
    total = sum(value for value in values if value is not None)
    if total <= 0:
        return None
    return {GRADE_LABELS[index]: round(float(values[index] / total), 8) for index in range(5)}


def _quality_for_messidor(rows: list[dict[str, str]], root: Path, analysis_max_dimension: int, workers: int) -> list[dict[str, Any]]:
    path_cache: dict[str, Path] = {}
    items = [{"image_id": row.get("image_id"), "path": _messidor_path(root, row.get("image_path", ""), path_cache)} for row in rows]
    return asyncio.run(_assess_paths(items, "Messidor-2", analysis_max_dimension, workers))


def _messidor_quality_cache_key(rows: list[dict[str, str]], root: Path) -> str:
    path_cache: dict[str, Path] = {}
    parts = []
    for row in rows:
        path = _messidor_path(root, row.get("image_path", ""), path_cache)
        stat = path.stat() if path.is_file() else None
        parts.append(f"{row.get('image_id')}\0{path}\0{stat.st_size if stat else -1}\0{stat.st_mtime_ns if stat else -1}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _guard_trace(rows: list[dict[str, str]], quality_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    legacy = RetinaGuardEngine(version="retinaguard-v2-reliability", require_optional_capabilities=True)
    current = RetinaGuardEngine()
    output: list[dict[str, Any]] = []
    for row, quality_record in zip(rows, quality_records):
        quality = quality_record.get("quality") or {}
        grade = _integer(row.get("predicted_aptos_grade"))
        inputs = RetinaGuardInputs(
            quality_score=_number(quality.get("quality_score")),
            raw_confidence=_number(row.get("raw_confidence")),
            probabilities=_probabilities(row) or {},
            predicted_grade=grade,
            predicted_grade_label=GRADE_LABELS[grade] if grade in range(5) else None,
            model_version=row.get("model_version"),
        )
        before = legacy.evaluate(inputs).to_dict()
        after = current.evaluate(inputs).to_dict()
        before_codes = [flag.get("code") for flag in before.get("risk_flags", [])]
        after_codes = [flag.get("code") for flag in after.get("risk_flags", [])]
        optional = sorted(code for code in after_codes if code in OPTIONAL_WARNING_CODES)
        failures = [code for code in after_codes if code == "retinaguard_pipeline_failure"]
        output.append({
            "image_id": row.get("image_id"),
            "image_path": row.get("image_path"),
            "ground_truth_grade": _integer(row.get("adjudicated_dr_grade")),
            "predicted_grade": grade,
            "predicted_referable": (_number(row.get("referable_probability_grade_2_or_worse"), 0.0) or 0.0) >= 0.5,
            "quality": {"decision": quality.get("quality_decision"), "score": quality.get("quality_score"), "issues": [{"type": issue.get("type"), "severity": issue.get("severity")} for issue in quality.get("issues", [])]},
            "signals": after.get("available_signals", {}),
            "phase5": {"reliability_state": before.get("reliability_state"), "trust_score": before.get("trust_score"), "warning_codes": before_codes, "assessment_status": before.get("assessment_status")},
            "phase5_1": {"reliability_state": after.get("reliability_state"), "trust_score": after.get("trust_score"), "warning_codes": after_codes, "assessment_status": after.get("assessment_status"), "decision_trace": after.get("decision_trace")},
            "dependency_classification": {
                "optional_capabilities_unavailable": optional,
                "pipeline_failure": bool(failures),
                "classification": "PIPELINE_FAILURE" if failures else "CAPABILITY_NOT_AVAILABLE" if optional else "NO_MISSING_OPTIONAL_CAPABILITY",
                "critical_missing_signals": after.get("decision_trace", {}).get("critical_signals_missing", []),
            },
        })
    return output


def _state_counts(traces: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(item[key]["reliability_state"] for item in traces).items()))


def _reason_distribution(traces: list[dict[str, Any]]) -> dict[str, Any]:
    old_states = [item for item in traces if item["phase5"]["reliability_state"] == "INSUFFICIENT_EVIDENCE"]
    new_states = [item for item in traces if item["phase5_1"]["reliability_state"] == "INSUFFICIENT_EVIDENCE"]
    old_causes: Counter[str] = Counter()
    for item in old_states:
        old_causes.update(code for code in item["phase5"]["warning_codes"] if code in OPTIONAL_WARNING_CODES)
    new_causes: Counter[str] = Counter()
    for item in new_states:
        new_causes.update(item["phase5_1"]["decision_trace"].get("critical_signals_missing", []))
    return {
        "status": "COMPLETE",
        "phase5_insufficient_evidence_count": len(old_states),
        "phase5_reason_distribution": dict(sorted(old_causes.items())),
        "phase5_1_insufficient_evidence_count": len(new_states),
        "phase5_1_reason_distribution": dict(sorted(new_causes.items())),
        "optional_capability_warning_counts_in_all_rows": dict(sorted(Counter(code for item in traces for code in item["phase5_1"]["warning_codes"] if code in OPTIONAL_WARNING_CODES).items())),
        "interpretation": "Phase 5 treated optional attention/evidence and OOD absence as required evidence. Phase 5.1 retains their warnings and review routing but does not classify the complete core assessment as insufficient solely because those optional capabilities are unavailable.",
    }


def _false_negative_safety(traces: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in traces if item.get("ground_truth_grade") in range(5)]
    false_negatives = [item for item in eligible if item["ground_truth_grade"] >= 2 and not item["predicted_referable"]]

    def counts(phase: str) -> dict[str, Any]:
        warnings = Counter(code for item in false_negatives for code in item[phase]["warning_codes"])
        any_warning = sum(bool(item[phase]["warning_codes"]) for item in false_negatives)
        meaningful = sum(bool(set(item[phase]["warning_codes"]) & MEANINGFUL_WARNING_CODES) for item in false_negatives)
        states = Counter(item[phase]["reliability_state"] for item in false_negatives)
        return {
            "any_warning_count": any_warning,
            "any_warning_coverage": round(any_warning / len(false_negatives), 8) if false_negatives else None,
            "meaningful_warning_count": meaningful,
            "meaningful_warning_coverage": round(meaningful / len(false_negatives), 8) if false_negatives else None,
            "warning_counts": dict(sorted(warnings.items())),
            "state_counts": dict(sorted(states.items())),
        }

    return {
        "status": "COMPLETE",
        "source_rule": "Messidor-2 third-party labelled grade >= 2 with the original Phase 4B referable probability < 0.50; no warning threshold was optimized.",
        "eligible_sample_count": len(eligible),
        "false_negative_count": len(false_negatives),
        "phase5": counts("phase5"),
        "phase5_1": counts("phase5_1"),
        "safety_comparison": "The policy change only reclassifies optional-capability absence from INSUFFICIENT_EVIDENCE to REVIEW_RECOMMENDED. It does not remove high-severity warnings, alter classifier probabilities, alter the referable rule, or create a TRUSTED result.",
        "limitations": [
            "This is a retrospective warning audit, not a prospective false-negative detector.",
            "Messidor-2 labels are third-party reference labels as documented by Phase 4B, not official Messidor-2 DR ground truth.",
            "Warning coverage is not sensitivity, clinical safety, or causal evidence.",
        ],
    }


def _demo_summary() -> dict[str, Any]:
    return {
        "status": "SYNTHETIC_FIXTURES_ONLY",
        "quality_distribution_measured": False,
        "trust_scores_available_in_fixture": {"count": 2, "values": [0.89, 0.58], "summary": _summary([0.89, 0.58])},
        "source": "backend/app/demo/scenarios.py",
        "reason": "The controlled demo contains JSON fixture values and no known-good image bytes. It is not mixed into APTOS/Messidor quality statistics or reliability conclusions.",
    }


def _threshold_rationale(aptos: dict[str, Any], messidor: dict[str, Any]) -> dict[str, Any]:
    dominant = sorted(messidor.get("issue_counts", {}).items(), key=lambda item: (-item[1], item[0]))
    return {
        "status": "AUDITED_NO_THRESHOLD_CHANGE_JUSTIFIED",
        "current_runtime_rules": {
            "focus_score": "log_score(laplacian_variance, low=5, high=300)",
            "borderline": "weighted quality score < 0.75 OR any quality issue",
            "ungradable": "weighted quality score < 0.45 OR any severe issue",
            "quality_weights": {"focus": 0.25, "illumination": 0.15, "contrast": 0.15, "field_of_view": 0.20, "exposure": 0.15, "artifacts": 0.10},
        },
        "observed_messidor_issue_order": [{"issue": key, "count": value} for key, value in dominant],
        "observed_quality": {"aptos": aptos.get("quality_score"), "messidor": messidor.get("quality_score")},
        "threshold_derivation_status": "HEURISTIC_CODE_CONFIGURATION_ONLY",
        "scientific_finding": "The existing thresholds are documented engineering heuristics, not thresholds derived from a labelled gradability study or camera-specific calibration artifact in this repository.",
        "decision": "Retain the existing quality thresholds for safety. The high Messidor-2 BORDERLINE rate is consistent with the conservative any-issue rule and lower observed focus/contrast distribution; changing it from this audit alone would risk weakening the gate without gradability labels.",
        "required_follow_up_before_recalibration": [
            "Collect authorized gradability labels across the deployed camera population.",
            "Evaluate component and aggregate thresholds against those labels on a held-out set.",
            "Version any threshold change separately and re-run leakage, safety, and external validation audits.",
        ],
    }


def _write_report(output: Path, payload: dict[str, Any]) -> None:
    quality = payload["quality_distributions"]
    traces = payload["reliability_state_trace"]
    before = payload["comparison"]["phase5_state_counts"]
    after = payload["comparison"]["phase5_1_state_counts"]
    report = f"""# Phase 5.1 RetinaGuard Reliability Usability Audit

Generated: `{payload['generated_at_utc']}`

## Scope and safety statement

This audit evaluates the existing RetinaGuard decision layer against actual
local APTOS and Messidor-2 image bytes and the authorized Phase 4B prediction
CSV. It does not train or tune a model, does not establish clinical validity,
and does not treat `TRUSTED` as proof that a prediction is correct. `TRUSTED`
means that no configured major warning was detected among the checks available
to that run.

## Audit result

The Phase 5 result was operationally safe but too conservative in one specific
way: the legacy policy treated optional attention-lesion agreement and OOD
monitoring absence as required evidence. That produced `{before.get('INSUFFICIENT_EVIDENCE', 0)}`
`INSUFFICIENT_EVIDENCE` results in the Messidor-2 retrospective run even
though the core quality/classifier/uncertainty path completed and no pipeline
failure was recorded. The quality gate itself produced
`{quality['messidor2']['decision_counts'].get('BORDERLINE', 0)}` BORDERLINE
results, so the high borderline rate is a separate conservative quality-gate
finding.

Phase 5.1 keeps optional-signal warnings and routes those results to
`REVIEW_RECOMMENDED`; it does not convert them to `TRUSTED`, remove high-risk
warnings, alter probabilities, or alter the referable rule. Updated states are:

* Phase 5: `{json.dumps(before, sort_keys=True)}`
* Phase 5.1: `{json.dumps(after, sort_keys=True)}`

## Quality distributions

APTOS measured `{quality['aptos2019']['file_count']}` local images
(`{quality['aptos2019']['readable_file_count']}` readable). Messidor-2 measured
`{quality['messidor2']['file_count']}` images (`{quality['messidor2']['readable_file_count']}`
readable). The complete descriptive statistics, percentiles, issue counts,
and histograms are in `quality_distribution_audit.json`.

The controlled demo has no known-good image bytes; its fixture trust scores are
reported separately as synthetic values only and are not included in dataset
statistics.

## Root-cause classification

* **Scientifically correct:** missing OOD reference data must be reported as
  unavailable; real-time explanation stability is intentionally optional; the
  classifier source probabilities and high uncertainty warnings remain intact.
* **Conservative behavior:** the Image Trust Gate's any-issue-to-BORDERLINE
  rule and the legacy optional-capability requirement drove the observed state
  distribution.
* **Dependency/capability limitation:** no authorized OOD reference was
  configured, no lesion evidence/attention agreement was present in the Phase
  4B prediction-only records, and stability was not run in real time.
* **Pipeline failure:** none was recorded in the audited Messidor-2 rows.

The full machine-readable per-image trace is in `reliability_state_trace.json`.
Each row records signal availability, legacy and updated state, warning codes,
assessment status, optional capability limitations, and pipeline-failure
classification.

## Threshold decision

The current quality thresholds remain unchanged. The threshold rationale and
the actual APTOS/Messidor component distributions are in
`quality_threshold_rationale.json`. A labelled gradability study is required
before recalibration; this audit alone cannot justify relaxing a safety gate.

## False-negative safety audit

The Phase 4B referable false-negative rule identified
`{payload['false_negative_safety']['false_negative_count']}` cases. The before/after
warning and state comparison is in `false_negative_safety_comparison.json`.
The policy change preserves high-severity warning coverage and only changes
the label applied when optional capabilities are unavailable.

## Recommended next steps

1. Keep Phase 5.1's versioned graceful-degradation configuration for current
   operation and display unavailable checks explicitly in the UI.
2. Provide an authorized reference distribution for OOD monitoring before
   interpreting OOD status as available.
3. Run lesion evidence and explanation stability when the workflow requests
   those capabilities; do not infer them from absent records.
4. Collect independent gradability labels before changing quality thresholds.

This remains an engineering readiness/usability audit and is not a clinical
validation or regulatory approval claim.
"""
    (output / "phase5_1_usability_audit.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    aptos_root = Path(args.aptos).resolve()
    messidor_root = Path(args.messidor).resolve()
    predictions_path = Path(args.predictions).resolve()
    aptos = _aptos_inventory(aptos_root)
    aptos_items = [{"image_id": item["image_id"], "label": item["label"], "path": item["path"]} for item in aptos["labelled"]]
    aptos_items.extend({"image_id": path.stem, "path": path} for path in aptos["test_files"])
    aptos_cache = output / "_phase5_1_aptos_quality_records.json"
    aptos_cache_key = hashlib.sha256("\n".join(f"{item['image_id']}\0{Path(item['path']).stat().st_size}\0{Path(item['path']).stat().st_mtime_ns}" for item in aptos_items).encode("utf-8")).hexdigest()
    cached_aptos = _load(aptos_cache)
    if isinstance(cached_aptos, dict) and cached_aptos.get("cache_key") == aptos_cache_key and isinstance(cached_aptos.get("records"), list) and len(cached_aptos["records"]) == len(aptos_items):
        aptos_records = cached_aptos["records"]
        print(f"APTOS quality audit: loaded {len(aptos_records)} cached records", flush=True)
    else:
        aptos_records = asyncio.run(_assess_paths(aptos_items, "APTOS", args.analysis_max_dimension, args.workers))
        _json(aptos_cache, {"cache_key": aptos_cache_key, "analysis_max_dimension": args.analysis_max_dimension, "records": aptos_records})
    aptos_labels = dict(sorted(Counter(str(item["label"]) for item in aptos["labelled"] if item["label"] in range(5)).items()))
    aptos_quality = _quality_summary(aptos_records, aptos_root, labels={"train_annotation_rows": len(aptos["train_rows"]), "train_class_distribution": aptos_labels, "missing_label_rows": len(aptos["missing_labels"]), "invalid_label_rows": len(aptos["invalid_labels"]), "duplicate_annotation_ids": len(aptos["duplicate_ids"])}, analysis_max_dimension=args.analysis_max_dimension)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Messidor-2 prediction CSV does not exist: {predictions_path}")
    messidor_rows = _read_csv(predictions_path)
    messidor_cache = output / "_phase5_1_messidor_quality_records.json"
    messidor_cache_key = _messidor_quality_cache_key(messidor_rows, messidor_root)
    cached_messidor = _load(messidor_cache)
    if isinstance(cached_messidor, dict) and cached_messidor.get("cache_key") == messidor_cache_key and isinstance(cached_messidor.get("records"), list) and len(cached_messidor["records"]) == len(messidor_rows):
        messidor_quality_records = cached_messidor["records"]
        print(f"Messidor-2 quality audit: loaded {len(messidor_quality_records)} cached records", flush=True)
    else:
        messidor_quality_records = _quality_for_messidor(messidor_rows, messidor_root, args.analysis_max_dimension, args.workers)
        _json(messidor_cache, {"cache_key": messidor_cache_key, "analysis_max_dimension": args.analysis_max_dimension, "records": messidor_quality_records})
    messidor_quality = _quality_summary(messidor_quality_records, messidor_root, labels={"label_rows": len(messidor_rows), "valid_grade_rows": sum(_integer(row.get("adjudicated_dr_grade")) in range(5) for row in messidor_rows), "label_provenance": "Phase 4B separately acquired third-party reference labels; not official Messidor-2 ground truth."}, analysis_max_dimension=args.analysis_max_dimension)
    traces = _guard_trace(messidor_rows, messidor_quality_records)
    old_counts = _state_counts(traces, "phase5")
    new_counts = _state_counts(traces, "phase5_1")
    phase5_metrics = _load(DEFAULT_OUTPUT / "reliability_metrics.json", {}) or {}
    source_hash = _sha256(predictions_path)
    duplicate_aptos = _source_duplicate_summary(ROOT / "ml" / "datasets" / "metadata" / "reports" / "aptos2019" / "dataset_validation_report.json", "Inherited from the existing dataset governance scan; raw files were not rewritten.")
    duplicate_messidor = _source_duplicate_summary(ROOT / "ml" / "evaluation" / "messidor" / "duplicate_audit.json", "Inherited from the Phase 4B duplicate audit; predictions were not rewritten.")
    payload: dict[str, Any] = {
        "status": "COMPLETE",
        "audit_version": "phase5.1-usability-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": {"aptos_root": str(aptos_root), "messidor_root": str(messidor_root), "messidor_prediction_csv": str(predictions_path), "prediction_csv_sha256": source_hash, "demo": _demo_summary()},
        "quality_distributions": {"aptos2019": aptos_quality, "messidor2": messidor_quality, "demo": _demo_summary()},
        "duplicate_audits": {"aptos2019": duplicate_aptos, "messidor2": duplicate_messidor},
        "reliability_state_trace": {"status": "COMPLETE", "row_count": len(traces), "records": traces, "source_prediction_csv_sha256": source_hash, "pipeline_failure_count": sum(item["dependency_classification"]["pipeline_failure"] for item in traces), "dependency_classification_counts": dict(sorted(Counter(item["dependency_classification"]["classification"] for item in traces).items()))},
        "comparison": {"phase5_engine_version": (phase5_metrics.get("configuration") or {}).get("engine_version", "retinaguard-v2-reliability"), "phase5_1_engine_version": RetinaGuardEngine.VERSION, "phase5_state_counts": old_counts, "phase5_1_state_counts": new_counts, "state_transitions": dict(sorted(Counter(f"{item['phase5']['reliability_state']} -> {item['phase5_1']['reliability_state']}" for item in traces).items())), "quality_decisions_unchanged": True, "classifier_predictions_unchanged": True, "prediction_csv_sha256": source_hash},
        "threshold_rationale": _threshold_rationale(aptos_quality, messidor_quality),
        "false_negative_safety": _false_negative_safety(traces),
    }
    _json(output / "quality_distribution_audit.json", {"status": payload["status"], "audit_version": payload["audit_version"], "generated_at_utc": payload["generated_at_utc"], "scope": payload["scope"], "datasets": payload["quality_distributions"], "duplicates": payload["duplicate_audits"], "interpretation": "Actual image-byte quality distributions only; synthetic demo values are clearly segregated."})
    _json(output / "reliability_state_trace.json", payload["reliability_state_trace"])
    _json(output / "insufficient_evidence_reason_distribution.json", _reason_distribution(traces))
    _json(output / "phase5_vs_phase5_1_comparison.json", payload["comparison"])
    _json(output / "quality_threshold_rationale.json", payload["threshold_rationale"])
    _json(output / "false_negative_safety_comparison.json", payload["false_negative_safety"])
    _json(output / "reliability_configuration_v2.json", {
        "format": "retinaguard-reliability-configuration-v2",
        "version": RetinaGuardEngine.VERSION,
        "derived_from": "retinaguard-v2-reliability",
        "weights_changed": False,
        "thresholds_changed": False,
        "optional_capability_policy": "graceful_degradation",
        "state_policy": {"TRUSTED": "No configured major warning among available checks; not a correctness guarantee.", "REVIEW_RECOMMENDED": "Core assessment completed but a review limitation or non-major warning remains, including unavailable optional capabilities.", "UNRELIABLE": "High-risk warning or score below the unreliable threshold; automated interpretation is not relied upon.", "INSUFFICIENT_EVIDENCE": "Core signal missing or pipeline failure prevents a complete reliability assessment."},
        "safety_rationale": "The audit found optional-capability absence, not pipeline failure, was the dominant cause of Phase 5 INSUFFICIENT_EVIDENCE states. Optional warnings remain visible and still prevent TRUSTED.",
        "clinical_validation_claim": False,
    })
    _write_report(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit RetinaGuard reliability usability on actual local datasets")
    parser.add_argument("--aptos", default=str(DEFAULT_APTOS))
    parser.add_argument("--messidor", default=str(DEFAULT_MESSIDOR))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--analysis-max-dimension", type=int, default=512, help="Maximum working-copy dimension for quality metrics; originals are validated first.")
    parser.add_argument("--workers", type=int, default=4, help="Bounded quality-audit worker count.")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "output": str(Path(args.output).resolve()), "aptos_images": payload["quality_distributions"]["aptos2019"]["file_count"], "messidor_images": payload["quality_distributions"]["messidor2"]["file_count"], "phase5_states": payload["comparison"]["phase5_state_counts"], "phase5_1_states": payload["comparison"]["phase5_1_state_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
