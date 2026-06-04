# Start Docker Compose deployment stack (Ollama + API + Streamlit + Caddy).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".env")) {
    Write-Host "Copy .env.docker.example to .env and fill in secrets (optional)." -ForegroundColor Yellow
    Copy-Item ".env.docker.example" ".env"
}

$pred = "data\processed\cmapss_FD001_predictions.parquet"
if (-not (Test-Path $pred)) {
    Write-Host "Missing $pred - run: python scripts/import_cmapss_colab_outputs.py --force" -ForegroundColor Red
    exit 1
}

& "$Root\deploy\preflight.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Building and starting services..." -ForegroundColor Cyan
docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker compose failed - is Docker Desktop running?" -ForegroundColor Red
    exit 1
}

Write-Host "Waiting for API health..." -ForegroundColor Cyan
$ok = $false
for ($i = 0; $i - 30; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($r.status -eq "ok") { $ok = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) {
    Write-Host "API not healthy yet - check: docker compose logs api" -ForegroundColor Yellow
} else {
    Write-Host "API healthy. Datasets: $($r.datasets_available -join ', ')" -ForegroundColor Green
}

Write-Host "Checking CMMS -> Databricks (API reads .env)..." -ForegroundColor Cyan
try {
    $cmms = Invoke-RestMethod -Uri "http://localhost:8000/cmms/databricks/status" -TimeoutSec 10
    if ($cmms.configured) {
        Write-Host "CMMS Databricks logging: ON" -ForegroundColor Green
        if ($cmms.table_fqn) { Write-Host "  Table: $($cmms.table_fqn)" }
        if ($cmms.explore_url) { Write-Host "  Explorer: $($cmms.explore_url)" }
    } else {
        Write-Host "CMMS Databricks logging: OFF" -ForegroundColor Yellow
        Write-Host "  Set CMMS_LOG_TO_DATABRICKS=true and DATABRICKS_* in .env, then: docker compose up -d --force-recreate api"
    }
} catch {
    Write-Host "  Could not reach /cmms/databricks/status (API still starting?)" -ForegroundColor Yellow
}

Write-Host "Ensuring llama3.2 in Ollama container..." -ForegroundColor Cyan
docker compose exec -T ollama ollama pull llama3.2

Write-Host ""
Write-Host "Demo ready:" -ForegroundColor Green
Write-Host "  Local app:  http://localhost:8080"
Write-Host "  API docs:   http://localhost:8080/api/docs"
Write-Host "  Tunnel:     .\deploy\start_tunnel.ps1"
Write-Host "  Stop:       docker compose down"
