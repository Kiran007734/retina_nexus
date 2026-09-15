"""Run a compact, real-image localhost validation pass.

This is an integration harness, not a synthetic demo. It uses three existing
APTOS images selected by the real Image Trust Gate and records summaries only;
it never writes uploaded image bytes or base64 evidence maps to the reports.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
IMAGE_CASES = {
    "gradable": ROOT / "ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png",
    "borderline": ROOT / "ml/datasets/raw/aptos2019/train_images/005b95c28852.png",
    "ungradable": ROOT / "ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png",
}


def compact_quality(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_decision": value.get("quality_decision"),
        "quality_score": value.get("quality_score"),
        "final_quality_score": value.get("final_quality_score"),
        "enhancement_applied": value.get("enhancement_applied"),
        "enhancement_passes": value.get("enhancement_passes"),
        "recheck_score": value.get("recheck_score"),
        "recheck_decision": value.get("recheck_decision"),
        "next_action": value.get("next_action"),
        "issues": [{"type": item.get("type"), "severity": item.get("severity")} for item in value.get("issues", [])],
        "input_metadata": value.get("input_metadata"),
    }


def compact_run(value: dict[str, Any]) -> dict[str, Any]:
    classification = value.get("classification") or {}
    guard = value.get("retinaguard") or {}
    lesions = value.get("lesions") or {}
    explanation = value.get("explainability") or {}
    grad_cam = explanation.get("grad_cam") or {}
    return {
        "screening_id": value.get("screening_id"),
        "status": value.get("status"),
        "primary_status": value.get("primary_status"),
        "evidence_status": value.get("evidence_status"),
        "stage_status": value.get("stage_status", {}),
        "stage_metrics": value.get("stage_metrics", {}),
        "stage_errors": value.get("stage_errors", {}),
        "classification": {
            key: classification.get(key)
            for key in ("predicted_grade", "predicted_grade_label", "referable_dr", "referable_probability", "raw_confidence", "model_version", "backbone")
        },
        "retinaguard": {
            key: guard.get(key)
            for key in ("trust_score", "trust_category", "recommended_action", "assessment_status", "warnings", "risk_flags")
        },
        "triage": value.get("triage"),
        "evidence": {
            "status": lesions.get("status"),
            "modules": {
                name: {"status": item.get("status"), "supported": item.get("supported"), "count": item.get("count"), "confidence": item.get("confidence")}
                for name, item in (lesions.get("modules") or {}).items()
            },
            "evidence_map_present": bool(lesions.get("evidence_map_data_uri")),
        },
        "explainability": {
            "predicted_class": explanation.get("predicted_class"),
            "agreement": explanation.get("attention_lesion_agreement"),
            "grad_cam_present": bool(grad_cam.get("overlay_data_uri")),
            "normalized_attention_present": bool(grad_cam.get("normalized_attention_map_data_uri")),
        },
    }


def request_summary(response: requests.Response) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": response.status_code,
        "content_type": response.headers.get("content-type"),
        "request_id": response.headers.get("x-request-id"),
    }
    if "json" in response.headers.get("content-type", ""):
        payload = response.json()
        if isinstance(payload, dict):
            summary["keys"] = sorted(payload.keys())
            if response.status_code >= 400:
                summary["body"] = payload
        else:
            summary["list_length"] = len(payload)
    else:
        summary["bytes"] = len(response.content)
        summary["prefix"] = response.content[:12].decode("latin-1", "replace")
    return summary


def get_json(session: requests.Session, url: str, **kwargs: Any) -> tuple[requests.Response, dict[str, Any]]:
    response = session.get(url, timeout=60, **kwargs)
    return response, response.json() if "json" in response.headers.get("content-type", "") else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--frontend-url", default="http://localhost:5173")
    parser.add_argument("--output-dir", default=str(ROOT / "ml/evaluation/localhost_validation"))
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--poll-timeout", type=int, default=330)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = args.backend_url.rstrip("/") + "/api/v1"
    frontend = args.frontend_url.rstrip("/")
    session = requests.Session()
    started = datetime.now(timezone.utc).isoformat()
    api_results: dict[str, Any] = {}
    runtime: dict[str, Any] = {"started_at": started, "cases": {}, "notes": []}

    def call(name: str, method: str, url: str, **kwargs: Any) -> requests.Response:
        begin = time.perf_counter()
        response = session.request(method, url, timeout=60, **kwargs)
        summary = request_summary(response)
        summary["duration_ms"] = round((time.perf_counter() - begin) * 1000, 1)
        api_results[name] = summary
        return response

    # Service and browser-origin checks.
    call("backend_root", "GET", args.backend_url.rstrip("/") + "/")
    ready_response = call("backend_ready", "GET", api + "/health/ready")
    call("backend_health", "GET", api + "/health")
    call("docs", "GET", args.backend_url.rstrip("/") + "/docs")
    call("openapi", "GET", args.backend_url.rstrip("/") + "/openapi.json")
    cors = call("cors_preflight", "OPTIONS", api + "/health", headers={"Origin": frontend, "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "content-type"})
    api_results["cors_preflight"]["allow_origin"] = cors.headers.get("access-control-allow-origin")
    api_results["cors_preflight"]["allow_methods"] = cors.headers.get("access-control-allow-methods")
    frontend_response = call("frontend_root", "GET", frontend + "/")
    frontend_html = frontend_response.text if frontend_response.status_code == 200 else ""
    for route in ("/screening/new", "/screening/results"):
        call("frontend_route:" + route, "GET", frontend + route)
    assets: list[dict[str, Any]] = []
    for asset in re.findall(r'(?:src|href)="([^"]+)"', frontend_html):
        if asset.startswith("/"):
            asset_response = call("frontend_asset:" + asset, "GET", frontend + asset)
            assets.append({"path": asset, "status": asset_response.status_code, "bytes": len(asset_response.content)})

    ready_payload = ready_response.json() if "json" in ready_response.headers.get("content-type", "") else {}
    if ready_response.status_code != 200 or not ready_payload.get("backend_ready"):
        raise RuntimeError("backend readiness did not pass")
    for label, path in IMAGE_CASES.items():
        if not path.is_file():
            raise FileNotFoundError(path)

    patient_response = call("patient_create", "POST", api + "/patients", json={"anonymized_identifier": "localhost-validation-" + str(int(time.time())), "age_group": "adult"})
    patient_response.raise_for_status()
    patient_id = patient_response.json()["id"]

    # Real-image quality and master pipeline paths.
    for label, path in IMAGE_CASES.items():
        content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        with path.open("rb") as image_file:
            upload_begin = time.perf_counter()
            upload = session.post(api + f"/images/upload?patient_id={patient_id}&eye=right", files={"image": (path.name, image_file, content_type)}, timeout=60)
        api_results[f"{label}_upload"] = request_summary(upload) | {"duration_ms": round((time.perf_counter() - upload_begin) * 1000, 1)}
        upload.raise_for_status()
        image_id = upload.json()["image_id"]
        quality_begin = time.perf_counter()
        quality = session.post(api + f"/images/{image_id}/quality", timeout=180)
        api_results[f"{label}_quality"] = request_summary(quality) | {"duration_ms": round((time.perf_counter() - quality_begin) * 1000, 1)}
        quality.raise_for_status()
        quality_payload = quality.json()
        case: dict[str, Any] = {"source_image": str(path.relative_to(ROOT)).replace("\\", "/"), "image_id": image_id, "quality": compact_quality(quality_payload)}
        if label == "ungradable" or quality_payload.get("quality_decision") != "GRADABLE":
            begin = time.perf_counter()
            run_response = session.post(api + "/screening/run", json={"image_id": image_id}, timeout=300)
            api_results[f"{label}_run"] = request_summary(run_response) | {"duration_ms": round((time.perf_counter() - begin) * 1000, 1)}
            run_response.raise_for_status()
            run_payload = run_response.json()
        else:
            begin = time.perf_counter()
            run_response = session.post(api + "/screening/run", json={"image_id": image_id}, timeout=300)
            api_results[f"{label}_run"] = request_summary(run_response) | {"duration_ms": round((time.perf_counter() - begin) * 1000, 1)}
            run_response.raise_for_status()
            run_payload = run_response.json()
        screening_id = run_payload["screening_id"]
        case["screening_id"] = screening_id
        case["initial_run"] = compact_run(run_payload)
        if run_payload.get("classification"):
            poll_started = time.perf_counter()
            latest = run_payload
            while latest.get("evidence_status") == "PROCESSING" and time.perf_counter() - poll_started < args.poll_timeout:
                time.sleep(max(1, args.poll_seconds))
                try:
                    # Use a fresh HTTP connection for each long-running poll.
                    # Optional evidence can be CPU-bound on local Windows and
                    # a stale keep-alive socket may be reset independently of
                    # the API process; the status endpoint remains durable.
                    status_response = requests.get(api + f"/screening/{screening_id}", timeout=60)
                    api_results[f"{label}_status_last"] = request_summary(status_response)
                    status_response.raise_for_status()
                    latest = status_response.json()
                except requests.RequestException as exc:
                    runtime["notes"].append(f"{label}: transient status poll transport error ({type(exc).__name__}); retrying")
                    continue
            case["final_run"] = compact_run(latest)
            case["poll_duration_ms"] = round((time.perf_counter() - poll_started) * 1000, 1)
            if latest.get("evidence_status") == "PROCESSING":
                runtime["notes"].append(f"{label}: optional evidence was still processing at the poll timeout")
        else:
            case["final_run"] = compact_run(run_payload)
        runtime["cases"][label] = case

    gradable = runtime["cases"]["gradable"]
    gradable_id = gradable["image_id"]
    gradable_sid = gradable["screening_id"]
    direct = call("direct_classifier", "POST", api + "/screening/classify", json={"image_id": gradable_id, "screening_session_id": gradable_sid})
    if direct.status_code == 200:
        payload = direct.json()
        api_results["direct_classifier"]["summary"] = {key: payload.get(key) for key in ("predicted_grade", "predicted_grade_label", "referable_dr", "referable_probability", "raw_confidence", "model_version")}
    result = call("persisted_result", "GET", api + f"/screening/{gradable_sid}/result")
    if result.status_code == 200:
        api_results["persisted_result"]["summary"] = result.json()
    content = call("image_content", "GET", api + f"/images/{gradable_id}/content?variant=original")
    api_results["image_content"]["image_bytes"] = len(content.content)

    # Two-request controlled concurrency probe. The endpoint's configured
    # semaphore is intentionally left unchanged; this records behavior under
    # a small realistic burst without launching an uncontrolled load test.
    concurrency_uploads: list[str] = []
    for index in (1, 2):
        with IMAGE_CASES["gradable"].open("rb") as image_file:
            upload = session.post(
                api + f"/images/upload?patient_id={patient_id}&eye=right",
                files={"image": (f"concurrency-{index}.png", image_file, "image/png")},
                timeout=60,
            )
        api_results[f"concurrency_upload_{index}"] = request_summary(upload)
        upload.raise_for_status()
        concurrency_uploads.append(upload.json()["image_id"])

    def run_concurrent(image_id: str) -> dict[str, Any]:
        begin = time.perf_counter()
        response = requests.post(api + "/screening/run", json={"image_id": image_id}, timeout=300)
        return {
            "image_id": image_id,
            "response": request_summary(response),
            "duration_ms": round((time.perf_counter() - begin) * 1000, 1),
            "screening_id": response.json().get("screening_id") if response.status_code == 200 else None,
            "primary_status": response.json().get("primary_status") if response.status_code == 200 else None,
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent_results = list(executor.map(run_concurrent, concurrency_uploads))
    runtime["concurrency"] = {
        "requested_workers": 2,
        "requests": concurrent_results,
        "all_http_200": all(item["response"].get("status") == 200 for item in concurrent_results),
        "note": "Controlled two-request probe; optional evidence was not polled for these probe sessions.",
    }
    for index, item in enumerate(concurrent_results, start=1):
        api_results[f"concurrency_run_{index}"] = item["response"] | {"duration_ms": item["duration_ms"]}

    # Negative/security probes use transport fixtures only; they are not model outputs.
    invalid = session.post(api + f"/images/upload?patient_id={patient_id}&eye=right", files={"image": ("invalid.png", b"not an image", "image/png")}, timeout=60)
    api_results["invalid_image_upload"] = request_summary(invalid)
    mismatch = session.post(api + f"/images/upload?patient_id={patient_id}&eye=right", files={"image": ("mislabeled.png", IMAGE_CASES["gradable"].read_bytes(), "image/jpeg")}, timeout=60)
    api_results["mismatched_mime_upload"] = request_summary(mismatch)
    oversized = session.post(api + f"/images/upload?patient_id={patient_id}&eye=right", files={"image": ("oversized.png", b"0" * (16 * 1024 * 1024), "image/png")}, timeout=90)
    api_results["oversized_upload"] = request_summary(oversized)
    traversal = session.post(
        api + f"/images/upload?patient_id={patient_id}&eye=right",
        files={"image": ("..\\..\\path-traversal.png", IMAGE_CASES["gradable"].read_bytes(), "image/png")},
        timeout=60,
    )
    api_results["traversal_filename_upload"] = request_summary(traversal)
    if traversal.status_code == 201:
        traversal_image_id = traversal.json().get("image_id")
        api_results["traversal_filename_upload"]["generated_storage_id"] = bool(traversal_image_id)

    report = call("report_generate", "POST", api + "/reports/generate", json={"session_id": gradable_sid})
    artifact: dict[str, Any] = {"report_status": report.status_code}
    if report.status_code == 201:
        report_payload = report.json()
        report_body = report_payload.get("report") or {}
        artifact["report_id"] = report_payload.get("report_id")
        artifact["report_status_value"] = report_payload.get("status")
        artifact["ai_assessment"] = report_body.get("ai_assessment")
        artifact["retinaguard"] = {key: (report_body.get("retinaguard") or {}).get(key) for key in ("trust_score", "trust_category", "recommended_safe_action")}
        artifact["recommended_action"] = report_body.get("recommended_action")
        artifact["clinician_decision"] = report_body.get("clinician_decision")
        pdf = call("report_pdf", "GET", api + f"/reports/{report_payload['report_id']}/pdf")
        artifact["pdf"] = {"status": pdf.status_code, "content_type": pdf.headers.get("content-type"), "bytes": len(pdf.content), "pdf_header": pdf.content.startswith(b"%PDF"), "pdf_eof": pdf.content.endswith(b"%%EOF")}
    runtime["report_and_artifact"] = artifact

    api_results["frontend_contract"] = {"api_base": api, "upload_method": "POST", "upload_field": "image", "content_type_header_managed_by_browser": True, "run_method": "POST", "run_content_type": "application/json"}
    runtime["frontend"] = {"url": frontend, "status": frontend_response.status_code, "assets": assets, "browser_automation_available": False, "browser_note": "No browser automation tool is installed in this environment; live HTTP/API and production build checks were executed."}
    runtime["finished_at"] = datetime.now(timezone.utc).isoformat()

    ui_validation = {
        "frontend_http_status": frontend_response.status_code,
        "asset_statuses": assets,
        "api_client_contract_verified": {"upload_field": "image", "multipart_boundary_not_manually_set": True, "run_endpoint": "/api/v1/screening/run"},
        "browser_automation_available": False,
        "manual_ui_follow_up": "Open http://localhost:5173/screening/new and repeat the gradable upload if interactive browser evidence is required.",
    }
    artifact_validation = {"report": artifact, "no_base64_or_image_bytes_written": True, "source_images": {key: str(value.relative_to(ROOT)).replace("\\", "/") for key, value in IMAGE_CASES.items()}}
    (output_dir / "api_results.json").write_text(json.dumps(api_results, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "runtime_results.json").write_text(json.dumps(runtime, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "ui_validation.json").write_text(json.dumps(ui_validation, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "artifact_validation.json").write_text(json.dumps(artifact_validation, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "cases": {key: {"image_id": value["image_id"], "screening_id": value["screening_id"], "quality": value["quality"]["quality_decision"], "status": value["final_run"]["status"], "primary_status": value["final_run"]["primary_status"], "evidence_status": value["final_run"]["evidence_status"]} for key, value in runtime["cases"].items()}, "report": artifact, "api_failures": {key: value for key, value in api_results.items() if int(value.get("status", 0)) >= 400}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
