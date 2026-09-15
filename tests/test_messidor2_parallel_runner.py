"""Unit tests for the resumable Messidor-2 execution infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_messidor2_parallel import (
    TERMINAL_STATUSES,
    aggregate,
    cache_key,
    cache_path,
    deduplicate_jobs,
    load_jobs,
    progress_snapshot,
    read_completed,
    write_atomic_json,
)


def _job(image_id: str = "sample.png") -> dict:
    return {"image_id": image_id, "image_path": f"images/{image_id}", "image_sha256": "a" * 64}


def test_duplicate_prevention_and_cache_identity(tmp_path: Path) -> None:
    jobs = deduplicate_jobs([_job(), _job(), _job("other.png")])
    assert [item["image_id"] for item in jobs] == ["sample.png", "other.png"]
    first = cache_key("sample.png", "a" * 64, "frozen")
    second = cache_key("sample.png", "a" * 64, "changed")
    assert first != second
    assert cache_path(tmp_path, _job(), "frozen").name == f"{first}.json"


def test_resume_skips_only_terminal_results(tmp_path: Path) -> None:
    job = _job()
    path = cache_path(tmp_path, job, "frozen")
    write_atomic_json(path, {"job_key": cache_key(job["image_id"], job["image_sha256"], "frozen"), "status": "COMPLETED", "image_id": job["image_id"]})
    cached = read_completed(path, job, "frozen")
    assert cached and cached["status"] in TERMINAL_STATUSES
    write_atomic_json(path, {"job_key": cache_key(job["image_id"], job["image_sha256"], "frozen"), "status": "FAILED", "image_id": job["image_id"]})
    assert read_completed(path, job, "frozen") is None


def test_worker_failure_is_aggregated_without_fake_result() -> None:
    manifest = {"total_jobs": 2, "failed_jobs": 1}
    payload = aggregate(
        [{"image_id": "ok", "status": "COMPLETED"}, {"image_id": "bad", "status": "FAILED", "errors": {"worker": "x"}}],
        manifest,
        output=Path("test_parallel_aggregate.json"),
    )
    try:
        assert payload["status"] == "PARTIAL"
        assert payload["status_counts"] == {"COMPLETED": 1, "FAILED": 1}
        assert next(row for row in payload["results"] if row["image_id"] == "bad")["status"] == "FAILED"
    finally:
        Path("test_parallel_aggregate.json").unlink(missing_ok=True)


def test_limited_run_is_explicitly_marked_as_benchmark_subset() -> None:
    manifest = {"total_jobs": 1, "source_total_jobs": 1744, "failed_jobs": 0, "limited_run": True}
    payload = aggregate([{"image_id": "sample", "status": "COMPLETED"}], manifest, output=Path("test_parallel_subset.json"))
    try:
        assert payload["status"] == "BENCHMARK_PARTIAL"
        assert payload["evaluation_scope"] == "BENCHMARK_SUBSET"
        assert payload["source_total_jobs"] == 1744
    finally:
        Path("test_parallel_subset.json").unlink(missing_ok=True)


def test_progress_snapshot_has_resume_operational_fields() -> None:
    snapshot = progress_snapshot(3, 10, 1, 0.0)
    assert snapshot["completed"] == 3
    assert snapshot["total"] == 10
    assert snapshot["failures"] == 1
    assert "estimated_remaining_seconds" in snapshot


def test_load_jobs_deduplicates_source(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps({**_job(), "extra": 1}) for _ in range(2)) + "\n", encoding="utf-8")
    assert load_jobs(source) == [_job()]
