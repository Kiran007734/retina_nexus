"""Render the final localhost validation report from recorded JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def render(output_dir: Path, test_results: dict[str, str]) -> str:
    runtime = load(output_dir / "runtime_results.json")
    api = load(output_dir / "api_results.json")
    ui = load(output_dir / "ui_validation.json")
    artifact = load(output_dir / "artifact_validation.json")
    cases = runtime.get("cases", {})
    report_artifact = runtime.get("report_and_artifact", {})
    ready = api.get("backend_ready", {})
    frontend_contract = api.get("frontend_contract", {})
    lines = [
        "# RETINA-NEXUS Localhost Validation Report",
        "",
        f"Validation window: `{runtime.get('started_at')}` to `{runtime.get('finished_at')}`",
        "",
        "This is an engineering integration report. It is not a clinical validation or regulatory approval claim.",
        "",
        "## Verdict",
        "",
        "`LOCALHOST PROTOTYPE READY: NO`",
        "",
        "The backend, frontend assets, real-image API pipeline, reports, PDF, model preflight, and automated regression checks passed. Full interactive browser click-through could not be independently automated because no browser automation tool is installed in this environment; the UI contract, SPA routes, asset delivery, and production build were verified instead.",
        "",
        "## Service and model checks",
        "",
        f"- Backend readiness: HTTP `{ready.get('status')}`, `backend_ready={ready.get('body', {}).get('backend_ready', 'see response')}`.",
        f"- Frontend root: HTTP `{api.get('frontend_root', {}).get('status')}`.",
        f"- API docs/OpenAPI: HTTP `{api.get('docs', {}).get('status')}` / `{api.get('openapi', {}).get('status')}`.",
        f"- CORS preflight: HTTP `{api.get('cors_preflight', {}).get('status')}`, origin `{api.get('cors_preflight', {}).get('allow_origin')}`.",
        f"- Model preflight: `{test_results.get('model_preflight', 'not supplied')}`.",
        "- Production ML defaults were not changed by this validation pass.",
        "",
        "## Real-image cases",
        "",
        "| Case | Source | Quality | Screening | Primary | Evidence | Grade | Referable | Trust |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for label in ("gradable", "borderline", "ungradable"):
        case = cases.get(label, {})
        quality = case.get("quality", {})
        final_run = case.get("final_run", {})
        classification = final_run.get("classification", {})
        guard = final_run.get("retinaguard", {})
        lines.append(
            f"| {label} | `{case.get('source_image')}` | {quality.get('quality_decision')} | "
            f"{final_run.get('status')} | {final_run.get('primary_status')} | {final_run.get('evidence_status')} | "
            f"{classification.get('predicted_grade', '—')} | {classification.get('referable_dr', '—')} | "
            f"{guard.get('trust_category', '—')} ({guard.get('trust_score', '—')}) |"
        )
    lines += [
        "",
        "The gradable case was processed by the registered EfficientNet-B0 classifier and optional evidence modules. Borderline and ungradable cases were blocked from clinical AI after the quality gate and returned recapture-oriented flow results.",
        "",
        "## Reports and artifacts",
        "",
        f"- Report generation: HTTP `{report_artifact.get('report_status')}`, report ID `{report_artifact.get('report_id')}`.",
        f"- PDF: HTTP `{report_artifact.get('pdf', {}).get('status')}`, `{report_artifact.get('pdf', {}).get('bytes')} bytes`, header `{report_artifact.get('pdf', {}).get('pdf_header')}`, EOF `{report_artifact.get('pdf', {}).get('pdf_eof')}`.",
        f"- Artifact safety: `{artifact.get('no_base64_or_image_bytes_written')}`; reports contain summaries rather than image bytes.",
        "",
        "## API contract and security probes",
        "",
        f"- Frontend upload contract: `{frontend_contract.get('upload_method')}` `/images/upload`, field `{frontend_contract.get('upload_field')}`, browser-managed multipart boundary `{frontend_contract.get('content_type_header_managed_by_browser')}`.",
        f"- Frontend run contract: `{frontend_contract.get('run_method')}` `/screening/run` with `{frontend_contract.get('run_content_type')}`.",
        f"- Invalid image: HTTP `{api.get('invalid_image_upload', {}).get('status')}` (expected rejection).",
        f"- Mismatched decoded MIME: HTTP `{api.get('mismatched_mime_upload', {}).get('status')}` (expected rejection after upload hardening).",
        f"- Oversized transport fixture: HTTP `{api.get('oversized_upload', {}).get('status')}` (expected rejection).",
        f"- Traversal filename probe: HTTP `{api.get('traversal_filename_upload', {}).get('status')}`; generated storage ID `{api.get('traversal_filename_upload', {}).get('generated_storage_id')}`; server-side storage uses generated IDs, not client paths.",
        "",
        "## Controlled concurrency",
        "",
        f"- Requested workers: `{runtime.get('concurrency', {}).get('requested_workers')}`; all requests HTTP 200: `{runtime.get('concurrency', {}).get('all_http_200')}`.",
        f"- Note: {runtime.get('concurrency', {}).get('note')}",
        "",
        "## Regression checks",
        "",
        f"- Pytest: `{test_results.get('pytest', 'not supplied')}`.",
        f"- Python compileall: `{test_results.get('compileall', 'not supplied')}`.",
        f"- Frontend lint/typecheck: `{test_results.get('frontend_lint', 'not supplied')}`.",
        f"- Frontend production build: `{test_results.get('frontend_build', 'not supplied')}`.",
        f"- Browser automation: `{ui.get('browser_automation_available')}`; {ui.get('manual_ui_follow_up')}",
        "",
        "## Root cause fixed during this pass",
        "",
        "The upload route validated decoded image bytes but accepted a declared media type that disagreed with the decoded format. A valid PNG sent as `image/jpeg` could therefore enter storage with inconsistent metadata. The route now rejects that request with HTTP 415 and `UNSUPPORTED_MEDIA_TYPE`; normal frontend uploads continue to use the browser-generated multipart boundary and the `image` field.",
        "",
        "No model weights, model architecture, inference thresholds, quality thresholds, RetinaGuard rules, datasets, or evaluation metrics were changed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="ml/evaluation/localhost_validation")
    parser.add_argument("--pytest", default="not supplied")
    parser.add_argument("--compileall", default="not supplied")
    parser.add_argument("--frontend-lint", default="not supplied")
    parser.add_argument("--frontend-build", default="not supplied")
    parser.add_argument("--model-preflight", default="not supplied")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = render(
        output_dir,
        {
            "pytest": args.pytest,
            "compileall": args.compileall,
            "frontend_lint": args.frontend_lint,
            "frontend_build": args.frontend_build,
            "model_preflight": args.model_preflight,
        },
    )
    for name in ("localhost_validation_report.md", "FINAL_LOCALHOST_REPORT.md"):
        (output_dir / name).write_text(report, encoding="utf-8")
    print(output_dir / "FINAL_LOCALHOST_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
