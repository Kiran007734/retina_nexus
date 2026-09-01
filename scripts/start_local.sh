#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_port="${BACKEND_PORT:-8000}"
frontend_port="${FRONTEND_PORT:-5173}"

command -v python >/dev/null || { echo "Python 3.11+ is required." >&2; exit 2; }
command -v npm >/dev/null || { echo "Node.js 20+ and npm are required." >&2; exit 2; }
[[ -f "$repo_root/backend/.env" ]] || { echo "backend/.env is missing. Copy backend/.env.example and configure CLASSIFIER_MODEL_PATH." >&2; exit 2; }
[[ -d "$repo_root/frontend/node_modules" ]] || { echo "frontend/node_modules is missing. Run npm ci in frontend." >&2; exit 2; }

cd "$repo_root"
python scripts/verify_models.py
mkdir -p "${TMPDIR:-/tmp}/retina-nexus-runtime"
log_dir="${TMPDIR:-/tmp}/retina-nexus-runtime"

(cd backend && exec python -m uvicorn app.main:app --host 127.0.0.1 --port "$backend_port") >"$log_dir/backend.out.log" 2>"$log_dir/backend.err.log" &
backend_pid=$!

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$backend_port/api/v1/health/ready" >/dev/null; then break; fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$backend_port/api/v1/health/ready" >/dev/null || { echo "Backend did not become ready; inspect $log_dir/backend.err.log." >&2; exit 1; }

(cd frontend && exec npm run dev -- --host 127.0.0.1 --port "$frontend_port") >"$log_dir/frontend.out.log" 2>"$log_dir/frontend.err.log" &
frontend_pid=$!
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$frontend_port/" >/dev/null; then break; fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$frontend_port/" >/dev/null || { echo "Frontend did not become available; inspect $log_dir/frontend.err.log." >&2; exit 1; }

echo "RETINA-NEXUS is ready for local demonstration."
echo "Backend:  http://127.0.0.1:$backend_port"
echo "Frontend: http://127.0.0.1:$frontend_port"
echo "API docs: http://127.0.0.1:$backend_port/docs"
echo "Backend PID: $backend_pid  Frontend PID: $frontend_pid"
echo "Logs: $log_dir"
