"""Explicitly gated synthetic product demonstrations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.demo.scenarios import get_scenario, list_scenarios

router = APIRouter(prefix="/demo", tags=["demo"])


def _require_demo_mode() -> None:
    settings = get_settings()
    if settings.environment not in {"development", "test"} or not settings.demo_mode_enabled:
        raise HTTPException(status_code=404, detail="Demo mode is disabled. Set DEMO_MODE_ENABLED=true in a development or test environment.")


@router.get("/scenarios")
async def scenarios() -> dict:
    _require_demo_mode()
    return {"demo_mode": True, "sample_data": True, "scenarios": list_scenarios(), "note": "Synthetic demonstration fixtures only. They are not patient data, validation data, or clinical performance results."}


@router.post("/scenarios/{scenario_id}/run")
async def run_scenario(scenario_id: str) -> dict:
    _require_demo_mode()
    scenario = get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Demo scenario not found")
    return {
        "demo_mode": True,
        "sample_data": True,
        "persisted_to_clinical_records": False,
        "demo_run_id": f"demo-{uuid4()}",
        "executed_at": datetime.now(timezone.utc),
        "scenario": scenario,
        "note": "Synthetic demonstration only. This response does not represent a patient, model validation result, or diagnostic conclusion.",
    }
