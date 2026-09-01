param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipModelLoad
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

function Stop-IfStarted([System.Diagnostics.Process]$Process) {
    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python was not found on PATH. Install Python 3.11+ and activate the project environment." }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "npm was not found on PATH. Install Node.js 20+ before starting Retina-Nexus." }
if (-not (Test-Path (Join-Path $backendDir ".env"))) { throw "backend/.env is missing. Copy backend/.env.example to backend/.env and configure CLASSIFIER_MODEL_PATH." }
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) { throw "frontend/node_modules is missing. Run npm.cmd ci in frontend before starting Retina-Nexus." }

Push-Location $repoRoot
try {
    $verifyArgs = @("scripts/verify_models.py")
    if ($SkipModelLoad) { $verifyArgs += "--no-load" }
    & python @verifyArgs
    if ($LASTEXITCODE -ne 0) { throw "Model preflight failed. Required classifier availability is needed before the API accepts screening work." }

    $logDir = Join-Path $env:TEMP "retina-nexus-runtime"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $backendProcess = Start-Process -FilePath "python" -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort") -WorkingDirectory $backendDir -RedirectStandardOutput (Join-Path $logDir "backend.out.log") -RedirectStandardError (Join-Path $logDir "backend.err.log") -WindowStyle Hidden -PassThru
    $backendReady = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/v1/health/ready" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $backendReady = $true; break }
        } catch { }
    }
    if (-not $backendReady) { Stop-IfStarted $backendProcess; throw "Backend did not become ready. Inspect $logDir\backend.err.log for the actionable startup error." }

    $frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") -WorkingDirectory $frontendDir -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") -RedirectStandardError (Join-Path $logDir "frontend.err.log") -WindowStyle Hidden -PassThru
    $frontendReady = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort/" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $frontendReady = $true; break }
        } catch { }
    }
    if (-not $frontendReady) { Stop-IfStarted $frontendProcess; Stop-IfStarted $backendProcess; throw "Frontend did not become available. Inspect $logDir\frontend.err.log." }

    Write-Output "RETINA-NEXUS is ready for local demonstration."
    Write-Output "Backend:  http://127.0.0.1:$BackendPort"
    Write-Output "Frontend: http://127.0.0.1:$FrontendPort"
    Write-Output "API docs: http://127.0.0.1:$BackendPort/docs"
    Write-Output "Logs:     $logDir"
    Write-Output "Backend PID: $($backendProcess.Id)  Frontend PID: $($frontendProcess.Id)"
} finally {
    Pop-Location
}
