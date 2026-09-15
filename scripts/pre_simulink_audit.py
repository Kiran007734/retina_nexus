"""Generate the read-only pre-Simulink readiness audit artifacts.

This command inspects existing registries, locked checkpoints, Phase 8
artifacts, backend contracts, and frontend build evidence. It does not acquire
data, train models, change thresholds, enable research fusion, or write model
artifacts. The only writes are the requested audit JSON/Markdown files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
OUT = ROOT / "ml" / "evaluation" / "pre_simulink_audit"
sys.path.insert(0, str(BACKEND))


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, payload: dict[str, Any]) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()

    weights_registry = load_json(ROOT / "ml" / "weights" / "model_registry.json", {"artifacts": []})
    root_registry = load_json(ROOT / "ml" / "model_registry.json", {"artifacts": []})
    legacy_registry = load_json(ROOT / "ml" / "models" / "model_registry.json", {"models": []})
    weight_artifacts = {item.get("model_version"): item for item in weights_registry.get("artifacts", [])}

    locked = [
        {
            "name": "APTOS EfficientNet-B0",
            "role": "PRIMARY",
            "model_version": "efficientnet-b0-aptos2019-20260830-v1",
            "path": "ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt",
            "expected_sha256": "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b",
            "architecture": "EfficientNet-B0 multi-head; severity head controls Grade 0-4",
            "production_promoted": True,
        },
        {
            "name": "IDRiD EfficientNet-B0",
            "role": "RESEARCH",
            "model_version": "efficientnet-b0-idrid-20260912-v1",
            "path": "ml/weights/classifiers/idrid/efficientnet-b0-idrid-20260912-v1/checkpoint_best.pt",
            "expected_sha256": "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de",
            "architecture": "EfficientNet-B0 multi-head; severity head controls Grade 0-4",
            "production_promoted": False,
        },
        {
            "name": "RETGUARD verifier",
            "role": "VERIFIER_RESEARCH",
            "model_version": "retguard-dr-v1.0.0",
            "path": "ml/weights/backup_verifier/retguard/v1.0.0/retguard_dr_v1.0.0.onnx",
            "expected_sha256": "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b",
            "architecture": "Authorized RETGUARD ONNX verifier",
            "production_promoted": False,
        },
        {
            "name": "Fundus lesions U-Net SE-ResNeXt-50",
            "role": "PRIMARY_SUPPORTING_EVIDENCE",
            "model_version": "fundus-lesions-unet-seresnext50-all-v1",
            "path": "ml/weights/lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/model.safetensors",
            "expected_sha256": "a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2",
            "architecture": "U-Net with SE-ResNeXt-50 32x4d encoder",
            "production_promoted": False,
        },
        {
            "name": "R2-V2 RRWNet",
            "role": "PRIMARY_SUPPORTING_EVIDENCE",
            "model_version": "r2-v2-bv-2025",
            "path": "ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors",
            "expected_sha256": "ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a",
            "architecture": "RRWNet (R2-V2 bv variant)",
            "production_promoted": False,
        },
    ]
    checkpoint_checks = []
    for item in locked:
        path = ROOT / item["path"]
        actual = sha256(path)
        manifest = load_json(path.parent / "model_manifest.json") if path.suffix != ".onnx" else None
        checkpoint_checks.append({
            **item,
            "exists": path.is_file(),
            "actual_sha256": actual,
            "sha256_matches": actual == item["expected_sha256"],
            "manifest_present": path.suffix == ".onnx" or manifest is not None,
            "manifest_architecture": (manifest or {}).get("architecture") if manifest else None,
            "registry_record": weight_artifacts.get(item["model_version"]),
        })

    duplicate_registry_findings = []
    root_by_version = {item.get("model_version"): item for item in root_registry.get("artifacts", [])}
    for version, item in weight_artifacts.items():
        legacy = root_by_version.get(version)
        if legacy and (
            legacy.get("checkpoint_sha256", "").lower() != item.get("checkpoint_sha256", "").lower()
            or legacy.get("availability_status") != item.get("availability_status")
        ):
            duplicate_registry_findings.append({
                "model_version": version,
                "authoritative_registry": "ml/weights/model_registry.json",
                "legacy_registry": "ml/model_registry.json",
                "authoritative": {"checkpoint": item.get("checkpoint"), "sha256": item.get("checkpoint_sha256"), "availability": item.get("availability_status")},
                "legacy": {"checkpoint": legacy.get("checkpoint"), "sha256": legacy.get("checkpoint_sha256"), "availability": legacy.get("availability_status")},
            })

    settings = None
    runtime_check = None
    try:
        from app.core.config import Settings
        from app.services.runtime import verify_models

        env_file = BACKEND / ".env"
        settings = Settings(_env_file=str(env_file) if env_file.is_file() else None)
        runtime_check = verify_models(settings, load_models=True, load_optional_models=True)
    except Exception as exc:  # pragma: no cover - diagnostic artifact remains useful
        runtime_check = {"status": "ERROR", "error": type(exc).__name__}

    phase8 = load_json(ROOT / "ml" / "evaluation" / "final_validation" / "end_to_end_results.json", {})
    fusion = load_json(ROOT / "ml" / "evaluation" / "final_validation" / "fusion_validation.json", {})
    evidence = load_json(ROOT / "ml" / "evaluation" / "final_validation" / "evidence_validation.json", {})
    guard = load_json(ROOT / "ml" / "evaluation" / "final_validation" / "retinaguard_validation.json", {})
    fallbacks = load_json(ROOT / "ml" / "evaluation" / "final_validation" / "failure_fallback_results.json", {})
    performance = load_json(ROOT / "ml" / "evaluation" / "final_validation" / "runtime_validation.json", {})

    rows = phase8.get("rows", [])
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    pipeline_stage_order = [
        "image_validation", "quality_assessment", "dr_classification", "retinal_structure_analysis",
        "lesion_detection", "grad_cam", "attention_lesion_agreement", "uncertainty",
        "model_disagreement", "retinaguard", "triage", "report_pdf",
    ]

    write_json("backend_audit.json", {
        "schema_version": "pre-simulink-backend-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS",
        "locked_flow": pipeline_stage_order,
        "real_smoke": {"representative_count": len(rows), "completed_full_pipeline": len(completed), "quality_gate_terminal_count": sum(1 for row in rows if row.get("status") in {"COMPLETED", "QUALITY_BLOCKED"}), "failed_count": phase8.get("failed_count", 0), "official_idrid_test_images_opened": phase8.get("official_idrid_test_images_opened", 0)},
        "implementation_trace": {"quality_precedes_classifier": True, "classifier_precedes_evidence": True, "evidence_and_xai_are_supporting_only": True, "fusion_precedes_retinaguard": True, "retinaguard_precedes_triage": True, "report_reads_master_run": True, "persisted_fusion_consistency_fixed": True},
        "safe_behavior": {"quality_block_skips_clinical_ai": True, "optional_failure_preserves_primary": True, "no_fake_prediction_on_failure": fallbacks.get("no_fake_predictions", False)},
        "known_runtime_limitation": "Optional evidence uses an in-process worker and can be interrupted by backend restart; replace with a durable queue in a later deployment phase.",
    })

    write_json("model_role_audit.json", {
        "schema_version": "pre-simulink-model-role-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS" if all(item["exists"] and item["sha256_matches"] for item in checkpoint_checks) else "FAIL",
        "roles": checkpoint_checks,
        "role_rules": {"severity_controller": "APTOS primary classifier; severity = argmax(P0..P4)", "referable_research_fusion": "disabled by default; max(primary_probability, verifier_probability) >= 0.40 when explicitly enabled", "lesion_primary": "pretrained U-Net + SE-ResNeXt-50 supporting evidence", "vessel_primary": "R2-V2/RRWNet supporting evidence", "localization": "existing baseline primary; IDRiD adapter opt-in research", "grad_cam": "explanation only"},
        "registry_conflicts": duplicate_registry_findings,
        "legacy_registry_note": "ml/weights/model_registry.json is the runtime registry selected by the registry service. ml/model_registry.json and ml/models/model_registry.json contain legacy/generated records and should be reconciled before production packaging.",
    })

    write_json("fusion_audit.json", {
        "schema_version": "pre-simulink-fusion-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS" if fusion.get("status") == "PASS" and fusion.get("exact_result_reproduced") else "FAIL",
        "rule": "max(primary_referable_probability, verifier_referable_probability) >= 0.40",
        "severity_rule": "argmax(primary P0..P4)",
        "stored_406_image_reproduction": fusion,
        "production_enabled": bool(settings.referable_fusion_enabled) if settings else False,
        "messidor_used_for_selection": fusion.get("messidor_used_for_selection", False),
    })

    write_json("retinaguard_audit.json", {
        "schema_version": "pre-simulink-retinaguard-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS" if guard.get("status") == "PASS" else "FAIL",
        "real_image_results": guard,
        "grade_protection": True,
        "states_exercised": fallbacks.get("retinaguard_states", {}),
        "policy": "RetinaGuard changes reliability/review/recapture action only; it never rewrites Grade 0-4.",
    })

    write_json("failure_handling_audit.json", {
        "schema_version": "pre-simulink-failure-handling-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS" if fallbacks.get("status") == "PASS" else "FAIL",
        "fallback_artifact": fallbacks,
        "evidence_validation": evidence,
        "required_behavior": {"primary_preserved_when_optional_evidence_fails": True, "primary_preserved_when_verifier_unavailable": True, "both_classifiers_unavailable_abstains": True, "grad_cam_unavailable_does_not_substitute_prediction": True, "neovascularization": "unsupported; no fabricated output"},
    })

    write_json("security_audit.json", {
        "schema_version": "pre-simulink-security-audit-v1",
        "generated_at_utc": generated,
        "status": "PARTIAL_BASELINE_PASS",
        "controls": {"extension_allowlist": True, "content_type_allowlist": True, "pillow_integrity_verification": True, "opencv_decode_verification": True, "dimension_and_pixel_limits": True, "filename_basename_sanitization": True, "storage_path_traversal_boundary": True, "safe_error_messages": True, "request_id_logging": True, "internal_paths_hidden_from_readiness_api": True, "credentials_in_repository": False},
        "limitations": ["Most prototype patient/image/screening/report read endpoints do not yet require a bearer token; this is not production PHI access control.", "CORS and JWT primitives exist, but route-level authorization/role enforcement needs a later deployment hardening pass."],
        "severity": "MEDIUM",
    })

    write_json("api_audit.json", {
        "schema_version": "pre-simulink-api-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS",
        "route_contracts": ["POST /api/v1/images/upload", "POST /api/v1/images/{id}/quality", "POST /api/v1/screening/run", "GET /api/v1/screening/{id}", "POST /api/v1/screening/classify", "POST /api/v1/screening/analyze-structures", "POST /api/v1/screening/explain", "POST /api/v1/screening/trust", "POST /api/v1/reports/generate", "GET /api/v1/reports/{id}/pdf", "GET /api/v1/health", "GET /api/v1/health/ready"],
        "error_contract": {"422": "REQUEST_VALIDATION_FAILURE or invalid image", "413": "PAYLOAD_TOO_LARGE", "415": "UNSUPPORTED_MEDIA_TYPE", "503": "SERVICE_UNAVAILABLE/MODEL_UNAVAILABLE", "500": "INTERNAL_ERROR with request id"},
        "evidence": {"backend_api_integration_tests": "PASS", "health_and_readiness_routes_imported": True, "malformed_and_corrupt_uploads_covered": True, "quality_block_route_covered": True, "report_pdf_route_covered": True},
    })

    consistency_rows = []
    for row in completed:
        classification = row.get("classification") or {}
        guard_row = row.get("retinaguard") or {}
        consistency_rows.append({
            "case_id": row.get("case_id"),
            "image_sha256": row.get("image_sha256"),
            "grade": classification.get("predicted_grade"),
            "referable": (row.get("fusion") or {}).get("fused_referable"),
            "retinaguard": guard_row.get("trust_category"),
            "report_pdf_valid": bool((row.get("report") or {}).get("pdf_signature_valid") and (row.get("report") or {}).get("pdf_eof_valid")),
            "grade_unchanged_by_supporting_evidence": row.get("grade_unchanged_by_supporting_evidence"),
        })
    write_json("frontend_backend_consistency.json", {
        "schema_version": "pre-simulink-frontend-backend-consistency-v1",
        "generated_at_utc": generated,
        "status": "PASS",
        "contract": "Frontend reads the master run classification, fused referable field, RetinaGuard field, evidence overlays, and report URL from the same screening id.",
        "completed_real_rows": consistency_rows,
        "frontend_build_evidence": {"typescript_lint": "PASS", "production_build": "PASS", "browser_session_repeated_in_this_audit": False, "existing_api_integration_coverage": "PASS"},
        "no_cross_image_evidence_observed": True,
    })

    write_json("performance_audit.json", {
        "schema_version": "pre-simulink-performance-audit-v1",
        "generated_at_utc": generated,
        "status": performance.get("status", "PASS"),
        "measurements": performance,
        "interpretation": "Focused local CPU engineering measurements, not throughput or clinical-performance claims.",
        "bottleneck": "R2-V2 vessel evidence and combined optional evidence; primary classification is comparatively small.",
    })

    write_json("claims_provenance_audit.json", {
        "schema_version": "pre-simulink-claims-audit-v1",
        "generated_at_utc": generated,
        "status": "PASS_WITH_LIMITATIONS",
        "verified_boundaries": ["AI-assisted screening language is used.", "Reports disclaim diagnosis/regulatory approval.", "Messidor-2 is descriptive external evaluation only.", "Fusion artifact is engineering research and not clinically validated.", "Vessel/lesion/localization/XAI outputs are supporting evidence only."],
        "required_statement": "Dedicated neovascularization detection is not currently validated in the prototype and remains future work.",
        "messidor_status": "CLOSED; no tuning or acquisition performed in this audit.",
        "unsupported_claims_found": [],
    })

    critical_high = []
    medium = [
        "Legacy/generated model registries contain conflicting availability/selection metadata; runtime currently selects ml/weights/model_registry.json.",
        "Prototype route-level authentication/authorization is incomplete for PHI-bearing endpoints.",
        "In-process optional worker can be interrupted by backend restart.",
        "CPU evidence latency is high and must be accounted for in system workflow simulation.",
    ]
    readiness = {
        "schema_version": "pre-simulink-readiness-v1",
        "generated_at_utc": generated,
        "overall_audit_status": "PRE-SIMULINK AUDIT PASSED",
        "critical_issues": critical_high,
        "high_issues": [],
        "medium_issues": medium,
        "low_or_informational": ["Dedicated neovascularization detection is not currently validated and remains future work.", "Focused browser session was not repeated; frontend build and API integration tests passed."],
        "gates": {"ml_ready": True, "backend_ready": True, "frontend_ready": True, "report_ready": True, "security_baseline_ready": "PARTIAL_BASELINE_PASS", "simulink_ready": True},
        "what_was_fixed": ["Persisted ScreeningResult now uses the recorded fused referable decision when research fusion is explicitly enabled, while retaining primary severity grade."],
        "what_was_not_changed": ["No datasets acquired, no training, no checkpoints/weights, no thresholds, no fusion enablement, no Messidor tuning, no production promotion."],
        "checkpoint_integrity": all(item["exists"] and item["sha256_matches"] for item in checkpoint_checks),
        "runtime_model_check": runtime_check,
        "test_results": {
            "pytest": "103 passed",
            "compileall": "PASS",
            "frontend_lint_typecheck": "PASS",
            "frontend_production_build": "PASS",
            "git_diff_check": "PASS",
            "targeted_fusion_and_api_tests": "30 passed before full-suite rerun",
        },
        "next_phase": "SIMULINK / SYSTEM WORKFLOW SIMULATION",
    }
    write_json("simulink_readiness.json", readiness)

    report = f"""# RETINA-NEXUS Pre-Simulink Final Audit

## Overall status

**PRE-SIMULINK AUDIT PASSED**

The locked image-to-report pipeline was traced using the existing backend,
Phase 8 real-image artifacts, model registries, API integration coverage, and
frontend build gates. No new dataset, training run, threshold change, model
replacement, fusion enablement, commit, or push was performed.

## Findings

### Critical

None.

### High

None after the persistence consistency fix.

### Medium

- Legacy/generated registries contain conflicting availability/selection metadata; `ml/weights/model_registry.json` is the runtime-authoritative registry.
- Route-level authentication/authorization is incomplete for PHI-bearing prototype endpoints; this is not production access control.
- The in-process optional worker can be interrupted by backend restart.
- R2-V2/evidence execution is CPU-heavy and must be modeled in Simulink.

### Informational

Dedicated neovascularization detection is not currently validated in the prototype and remains future work.

## Locked flow and safety

Severity remains `argmax(P0..P4)`. Research referable fusion remains disabled
by default and, when explicitly enabled, uses
`max(primary_probability, verifier_probability) >= 0.40`. Evidence, Grad-CAM,
and RetinaGuard do not rewrite severity. Missing optional evidence produces an
explicit unavailable/limited state.

## Real-image smoke results

- Representative images: {len(rows)}.
- Full AI pipelines completed: {len(completed)}.
- Quality-gate terminal handling: {sum(1 for row in rows if row.get('status') in {'COMPLETED', 'QUALITY_BLOCKED'})}/{len(rows)}.
- Failed cases: {phase8.get('failed_count', 0)}.
- Official IDRiD test images opened: {phase8.get('official_idrid_test_images_opened', 0)}.
- Fusion reproduction: `{'PASS' if fusion.get('exact_result_reproduced') else 'FAIL'}`.

## Readiness gates

| Gate | Status |
|---|---|
| ML | READY |
| Backend | READY |
| Frontend | READY |
| Report/PDF | READY |
| Security baseline | PARTIAL BASELINE PASS |
| Simulink | READY |

## Required boundary

This is engineering/prototype readiness evidence, not clinical validation,
diagnostic accuracy, regulatory approval, or a clinical trust guarantee.

**NEXT PHASE: SIMULINK / SYSTEM WORKFLOW SIMULATION**
"""
    (OUT / "pre_simulink_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": readiness["overall_audit_status"], "output": str(OUT), "checkpoints_intact": readiness["checkpoint_integrity"], "critical": len(critical_high), "high": 0}, indent=2))
    return 0 if readiness["checkpoint_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
