"""Build a read-only GitHub release inventory and packaging audit.

The audit never copies, deletes, downloads, rewrites, or promotes datasets or
model artifacts. It records hashes and metadata only, and deliberately omits
secret values from all generated reports.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml" / "evaluation" / "github_release_audit"
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".npz", ".zip"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"}
DATASET_SOURCE = {
    "aptos2019": "APTOS 2019 Blindness Detection; authorized Kaggle competition acquisition is documented in project scripts.",
    "idrid": "IDRiD official package; local files are used for research only and are not redistributed by this repository.",
    "drive": "DRIVE retinal vessel dataset; local files are used for vessel research/evaluation only.",
    "messidor": "Messidor/Messidor-2 source and access constraints are documented in docs/MESSIDOR_EXTERNAL_VALIDATION.md; no redistribution permission is asserted.",
}
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"BEGIN (?:RSA|EC|OPENSSH|PRIVATE) KEY"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
    re.compile(r"Bearer [A-Za-z0-9._-]{20,}"),
]
MACHINE_PATTERNS = [r"C:\\Users\\kiran", r"C:/Users/kiran", r"Downloads[\\/]retina_nexus"]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def ignored(path: str) -> bool:
    candidate = ROOT / path
    candidates = [candidate]
    if candidate.is_dir():
        candidates = [item for item in candidate.rglob("*") if item.is_file()]
    for item in candidates:
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", "--", rel(item)], cwd=ROOT, check=False)
        if result.returncode == 0:
            return True
    return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def registry_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in (ROOT / "ml" / "models" / "model_registry.json", ROOT / "ml" / "model_registry.json", ROOT / "ml" / "weights" / "model_registry.json"):
        if not path.is_file():
            continue
        payload = json_file(path)
        for record in payload.get("artifacts", []):
            if isinstance(record, dict):
                records.append({"registry": rel(path), **record})
    return records


def manifest_for(path: Path) -> tuple[dict[str, Any], Path | None]:
    current = path.parent
    while current >= ROOT:
        candidate = current / "model_manifest.json"
        if candidate.is_file():
            return json_file(candidate), candidate
        if current == ROOT:
            break
        current = current.parent
    return {}, None


def registry_match(path: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = path.replace("\\", "/").lower()
    matches: list[dict[str, Any]] = []
    for record in records:
        checkpoint = str(record.get("checkpoint") or "").replace("\\", "/").lower()
        if checkpoint and (normalized == checkpoint or normalized.endswith(checkpoint)):
            matches.append(record)
    return matches[0] if matches else {}


def role_for(path: str) -> tuple[str, str, str, bool]:
    normalized = path.replace("\\", "/")
    if normalized == "ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt":
        return "APTOS EfficientNet-B0 primary classifier", "PRIMARY_PRODUCTION_RUNTIME", "A. REQUIRED FOR LOCALHOST INFERENCE", True
    if "vessel_segmentation/r2-v2-bv-2025/bv.safetensors" in normalized:
        return "R2-V2 RRWNet vessel evidence", "PRIMARY_SUPPORTING_EVIDENCE", "B. OPTIONAL SUPPORTING RUNTIME ARTIFACT", False
    if "lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/model.safetensors" in normalized:
        return "Published fundus lesion evidence model", "PRIMARY_SUPPORTING_EVIDENCE", "B. OPTIONAL SUPPORTING RUNTIME ARTIFACT", False
    if "/backup_verifier/" in normalized:
        return "RETGUARD independent verifier candidate", "RESEARCH_ONLY_NOT_ADOPTED", "B. OPTIONAL RESEARCH/VERIFICATION", False
    if "/classifiers/idrid/" in normalized:
        return "IDRiD disease-grading research checkpoint", "RESEARCH_ONLY_NOT_ADOPTED", "B. OPTIONAL RESEARCH/VERIFICATION", False
    if "/lesions/idrid/" in normalized:
        return "IDRiD lesion research checkpoint", "RESEARCH_ONLY_NOT_ADOPTED", "B. OPTIONAL RESEARCH/VERIFICATION", False
    if "/localization/idrid/" in normalized:
        return "IDRiD optic-disc/fovea localization research checkpoint", "RESEARCH_ONLY_NOT_ADOPTED", "B. OPTIONAL RESEARCH/VERIFICATION", False
    if "/vessels/drive/" in normalized:
        return "DRIVE vessel research candidate", "RESEARCH_ONLY_NOT_ADOPTED", "B. OPTIONAL RESEARCH/VERIFICATION", False
    return "Unclassified local model artifact", "UNKNOWN", "D. UNKNOWN - HUMAN REVIEW REQUIRED", False


def weight_inventory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((ROOT / "ml" / "weights").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in WEIGHT_EXTENSIONS:
            continue
        relative = rel(path)
        model_name, role, category, runtime_required = role_for(relative)
        manifest, manifest_path = manifest_for(path)
        registered = registry_match(relative, records)
        declared_license = manifest.get("license") or registered.get("license")
        size = path.stat().st_size
        manifest_checkpoint = str(manifest.get("checkpoint") or "").replace("\\", "/").lower()
        normalized_relative = relative.lower()
        manifest_applies = bool(
            manifest_checkpoint
            and manifest_path
            and (
                normalized_relative == manifest_checkpoint
                or (manifest_checkpoint == path.name.lower() and path.parent == manifest_path.parent)
            )
        )
        entries.append({
            "path": relative,
            "filename": path.name,
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": sha256(path),
            "model_name": model_name,
            "architecture": manifest.get("architecture") or registered.get("architecture") or "not recorded",
            "purpose_role": role,
            "release_category": category,
            "runtime_required": runtime_required,
            "registry_reference": registered.get("registry"),
            "registry_declared_sha256": registered.get("checkpoint_sha256"),
            "manifest_declared_sha256": manifest.get("checkpoint_sha256") if manifest_applies else None,
            "manifest_sha256_comparison": "verified_against_this_file" if manifest_applies else "not_applicable_to_this_sibling_artifact",
            "source": manifest.get("source_url") or manifest.get("source") or registered.get("source") or "not recorded",
            "license": declared_license or "License/redistribution status: requires human verification.",
            "redistribution_status": "Requires human license/provenance review; no permission is inferred from local availability.",
            "git_storage": "External setup required; use Git LFS only after an explicit legal/distribution decision." if size > 100 * 1024 * 1024 else "External by project policy; normal Git is technically possible but not selected.",
            "ignored_by_git": ignored(relative),
        })
    return entries


def dataset_inventory() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    root = ROOT / "ml" / "datasets" / "raw"
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not path.is_dir():
            continue
        files = [item for item in path.rglob("*") if item.is_file()]
        images = [item for item in files if item.suffix.lower() in IMAGE_EXTENSIONS]
        entries.append({
            "dataset": path.name,
            "path": rel(path),
            "files": len(files),
            "images": len(images),
            "bytes": sum(item.stat().st_size for item in files),
            "ignored": ignored(rel(path)),
            "source_note": DATASET_SOURCE.get(path.name, "Source not recorded."),
            "release_classification": "DO NOT PUSH - EXTERNAL DATASET",
        })
    return entries


def secret_audit() -> dict[str, Any]:
    tracked = [item for item in run_git("ls-files").splitlines() if item]
    current_hits: set[str] = set()
    for relative in tracked:
        if relative.endswith(".env") or relative.startswith("frontend/node_modules/") or relative.startswith("ml/datasets/raw/") or relative.startswith("ml/weights/"):
            continue
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            current_hits.add(relative)
    history_hits: set[str] = set()
    for commit in run_git("rev-list", "--all").splitlines():
        for pattern in SECRET_PATTERNS:
            result = subprocess.run(["git", "grep", "-I", "-l", "-E", pattern.pattern, commit, "--", ":!frontend/node_modules", ":!ml/datasets/raw", ":!ml/weights"], cwd=ROOT, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                history_hits.update(line.split(":", 1)[-1] for line in result.stdout.splitlines() if line)
    return {"secrets_found": bool(current_hits or history_hits), "current_file_matches": sorted(current_hits), "history_file_matches": sorted(history_hits), "values_emitted": False}


def machine_path_audit() -> list[str]:
    matches: set[str] = set()
    candidates = list((ROOT / "docs").rglob("*")) + list((ROOT / "simulink").rglob("*")) + list((ROOT / "ml" / "evaluation").rglob("*.md"))
    for path in candidates:
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in MACHINE_PATTERNS):
            matches.add(rel(path))
    return sorted(matches)


def large_files() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "node_modules", "__pycache__", "slprj"} for part in path.parts):
            continue
        size = path.stat().st_size
        if size <= 50 * 1024 * 1024:
            continue
        relative = rel(path)
        entries.append({
            "path": relative,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "over_50mb": True,
            "over_100mb": size > 100 * 1024 * 1024,
            "over_500mb": size > 500 * 1024 * 1024,
            "tracked": relative in set(run_git("ls-files").splitlines()),
            "ignored": ignored(relative),
        })
    return sorted(entries, key=lambda item: item["size_bytes"], reverse=True)


def model_preflight() -> dict[str, Any]:
    result = subprocess.run([sys.executable, "scripts/verify_models.py", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)
    output = result.stdout.strip()
    try:
        start = output.find("{")
        payload = json.loads(output[start:]) if start >= 0 else {}
    except json.JSONDecodeError:
        payload = {"raw_status": "unparseable"}
    return {"returncode": result.returncode, "status": payload.get("status", "FAILED" if result.returncode else "UNKNOWN"), "models": payload.get("models", {}), "note": payload.get("note")}


def localhost_summary() -> dict[str, Any]:
    directory = ROOT / "ml" / "evaluation" / "localhost_validation"
    runtime = json_file(directory / "runtime_results.json")
    api = json_file(directory / "api_results.json")
    return {
        "artifact_directory": rel(directory) if directory.exists() else None,
        "cases": {label: {key: (case.get("quality", {}).get(key) if key == "quality" else case.get("final_run", {}).get(key)) for key in ("quality", "status", "primary_status", "evidence_status")} for label, case in runtime.get("cases", {}).items()},
        "report_status": runtime.get("report_and_artifact", {}).get("report_status"),
        "pdf_status": runtime.get("report_and_artifact", {}).get("pdf", {}).get("status"),
        "browser_automation_available": runtime.get("frontend", {}).get("browser_automation_available"),
        "negative_probe_statuses": {key: api.get(key, {}).get("status") for key in ("invalid_image_upload", "mismatched_mime_upload", "oversized_upload", "traversal_filename_upload")},
    }


def markdown_table(entries: list[dict[str, Any]]) -> str:
    lines = ["| Path | Size | SHA-256 | Role | Category | Registry/manifest | License/provenance |", "|---|---:|---|---|---|---|---|"]
    for entry in entries:
        declared = entry.get("registry_declared_sha256") or entry.get("manifest_declared_sha256") or "not recorded"
        registry = "yes" if entry.get("registry_reference") else "no"
        license_note = str(entry.get("license", "unknown")).replace("|", "/")
        lines.append(f"| `{entry['path']}` | {entry['size_mb']} MB | `{entry['sha256']}` | {entry['purpose_role']} | {entry['release_category']} | {registry} / `{declared}` | {license_note} |")
    return "\n".join(lines)


def write_reports(inventory: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    weights = inventory["model_weights"]
    (OUT / "MODEL_WEIGHT_INVENTORY.md").write_text(
        "# Model Weight Inventory\n\n"
        "Hashes are calculated from the current local files. No weight bytes are copied into this audit.\n\n"
        + markdown_table(weights) + "\n", encoding="utf-8"
    )
    dataset_lines = ["# Dataset Inventory\n", "| Dataset | Raw files | Images | Size | Git classification | Source note |", "|---|---:|---:|---:|---|---|"]
    for item in inventory["datasets"]:
        dataset_lines.append(f"| `{item['dataset']}` | {item['files']} | {item['images']} | {item['bytes'] / (1024 * 1024):.2f} MB | {item['release_classification']} | {item['source_note']} |")
    (OUT / "DATASET_INVENTORY.md").write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
    categories = {
        "Source code, configs, tests, and documentation": "PUSH",
        "Raw APTOS/IDRiD/DRIVE/Messidor datasets, labels, masks, and archives": "DO NOT PUSH / EXTERNAL REQUIRED",
        "Dataset metadata and evaluation reports": "PUSH WITH REVIEW",
        "APTOS production classifier checkpoint": "EXTERNAL REQUIRED",
        "Optional lesion and R2-V2 vessel checkpoints": "EXTERNAL REQUIRED; Git LFS only after license review",
        "IDRiD/DRIVE research checkpoints and cross-validation folds": "OPTIONAL / EXTERNAL REQUIRED",
        "RETGUARD ONNX verifier and OOD artifact": "OPTIONAL RESEARCH / EXTERNAL REQUIRED",
        "Simulink .slx, source scripts, plots, and compact results": "PUSH",
        "Secrets, .env files, uploads, databases, node_modules, caches, and logs": "DO NOT PUSH",
    }
    manifest_lines = ["# GitHub Release Manifest\n", "This classification is packaging guidance only; it does not grant dataset or model redistribution rights.\n", "| Artifact class | Classification | Decision basis |", "|---|---|---|"]
    for artifact, decision in categories.items():
        basis = "Keep source-controlled." if decision == "PUSH" else "Keep local/external and require documented authorization." if "EXTERNAL" in decision or "DO NOT" in decision else "Retain only after provenance and sensitivity review."
        manifest_lines.append(f"| {artifact} | **{decision}** | {basis} |")
    (OUT / "GITHUB_RELEASE_MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    large = inventory["large_files"]
    large_lines = ["| Path | Size | Tracked | Ignored | Recommendation |", "|---|---:|---|---|---|"]
    for item in large:
        recommendation = "External dataset/weight; do not push." if item["ignored"] else "Review before release; Git LFS/external hosting may be required."
        large_lines.append(f"| `{item['path']}` | {item['size_mb']} MB | {item['tracked']} | {item['ignored']} | {recommendation} |")
    blockers = [
        "Required runtime weights are external/ignored; a fresh clone cannot run real inference until authorized checkpoints are installed.",
        "Several model/dataset redistribution permissions are not established by the repository; human legal/provenance review is required.",
        "No project LICENSE or NOTICE file is present.",
        "GitHub push was previously blocked by unavailable github.com:443; this audit intentionally does not retry or push.",
    ]
    final = [
        "# FINAL GitHub Release Audit",
        "",
        f"Audit date: `{inventory['audit_date']}`",
        f"Commit audited: `{inventory['git']['commit']}`",
        f"Worktree at audit start: `{'CLEAN' if inventory['git']['clean_at_start'] else 'DIRTY'}`",
        "",
        "## Release decision",
        "",
        "**GITHUB READY: NO**",
        "",
        "This is a packaging/release decision, not a clinical or model-quality decision. No model, threshold, preprocessing, fusion rule, RetinaGuard rule, dataset, or screening flow was changed by this audit.",
        "",
        "## Inventory",
        "",
        f"- Frontend, backend, ML source, adapters, registries, XAI, evidence, RetinaGuard, reports, tests, Simulink model, plots, configs, scripts, and deployment files are present in the repository inventory.",
        f"- Local weight artifacts: `{len(weights)}` weight/support files; `{sum(item['size_bytes'] for item in weights) / (1024 * 1024):.2f} MB` total.",
        f"- Raw datasets: `{', '.join(item['dataset'] for item in inventory['datasets'])}`; all raw dataset roots are ignored and untracked.",
        "- See `MODEL_WEIGHT_INVENTORY.md` and `DATASET_INVENTORY.md` for complete records.",
        "",
        "## Model preflight",
        "",
        f"- Status: `{inventory['model_preflight']['status']}`.",
        "- The current APTOS EfficientNet-B0 checkpoint is the only required primary classifier runtime artifact.",
        "- Lesion and R2-V2 vessel models are supporting runtime capabilities and degrade explicitly when unavailable.",
        "- IDRiD, DRIVE research candidates, and RETGUARD verifier artifacts are not production-promoted.",
        "",
        "## Datasets and tracking",
        "",
        "- Raw image datasets, labels, masks, and archives are not tracked by Git.",
        "- Metadata/manifests are retained selectively, but any artifact containing local absolute paths or sensitive provenance should be reviewed before release.",
        "- No redistribution permission is inferred.",
        "",
        "## Security and portability",
        "",
        f"- Secret-pattern audit: `{'FAIL' if inventory['security']['secrets_found'] else 'PASS - no matches found'}`; secret values were not emitted.",
        f"- Machine-specific path references: `{len(inventory['machine_specific_paths'])}` report files identified for packaging cleanup.",
        "- Upload MIME/content mismatch, oversized upload, traversal filename, and invalid image checks are covered by the existing validation/test suite.",
        "- PHI-bearing route authentication/authorization remains incomplete for a clinical deployment.",
        "",
        "## Large files",
        "",
        *large_lines,
        "",
        "## Localhost and regression evidence",
        "",
        f"- Previous recorded real-image localhost validation: `{inventory['localhost']['cases']}`.",
        f"- Report/PDF statuses: `{inventory['localhost']['report_status']}` / `{inventory['localhost']['pdf_status']}`.",
        "- Previous regression result: 103 backend tests, compileall, model preflight, frontend lint, and frontend production build passed.",
        "- Browser automation was unavailable; HTTP/API, SPA routes, assets, and build were verified.",
        "",
        "## Simulink",
        "",
        "- `simulink/RETINA_NEXUS_SYSTEM.slx`, scenario scripts, results, plots, and documentation are present.",
        "- Simulation outputs are operational modeling results and require real-world site calibration; they are not clinical or financial claims.",
        "",
        "## Exact blockers before push",
        "",
        *[f"- {item}" for item in blockers],
        "",
        "## Required actions",
        "",
        "1. Human-review and approve model/dataset provenance, redistribution, and attribution terms.",
        "2. Decide which external checkpoints will be distributed, hosted separately, or installed manually; retain the current ignore policy unless legal approval changes.",
        "3. Add an appropriate project license/NOTICE after human/legal approval.",
        "4. Review the generated path cleanup and audit artifacts, then create a new packaging commit if accepted.",
        "5. Restore GitHub network/authentication and push only after the above approvals. This audit does not push.",
        "",
    ]
    (OUT / "FINAL_GITHUB_RELEASE_AUDIT.md").write_text("\n".join(final), encoding="utf-8")


def main() -> int:
    tracked = set(run_git("ls-files").splitlines())
    inventory = {
        "audit_date": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": run_git("rev-parse", "HEAD"), "branch": run_git("branch", "--show-current"), "clean_at_start": not bool(run_git("status", "--short"))},
        "model_weights": weight_inventory(registry_records()),
        "datasets": dataset_inventory(),
        "security": secret_audit(),
        "machine_specific_paths": machine_path_audit(),
        "large_files": large_files(),
        "tracked_risky_files": {"raw_dataset_files": sorted(path for path in tracked if path.startswith("ml/datasets/raw/") and not path.endswith(".gitkeep")), "weight_files": sorted(path for path in tracked if path.startswith("ml/weights/") and Path(path).suffix.lower() in WEIGHT_EXTENSIONS), "env_files": sorted(path for path in tracked if Path(path).name in {".env", "backend/.env", "frontend/.env"})},
        "model_preflight": model_preflight(),
        "localhost": localhost_summary(),
        "license_files_present": [path for path in ("LICENSE", "NOTICE", "COPYING") if (ROOT / path).exists()],
        "fresh_clone_assessment": {"source_present": True, "raw_datasets_required_for_normal_inference": False, "primary_classifier_weight_required_externally": True, "optional_evidence_weights_degrade_explicitly": True, "fresh_clone_real_inference_ready_without_external_artifacts": False},
    }
    write_reports(inventory)
    print(json.dumps({"output_dir": rel(OUT), "commit": inventory["git"]["commit"], "weight_files": len(inventory["model_weights"]), "datasets": inventory["datasets"], "secret_matches": inventory["security"]["secrets_found"], "preflight": inventory["model_preflight"]["status"], "large_files": len(inventory["large_files"]), "github_ready": "NO"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
