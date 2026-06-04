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
    Write-Host "Missing $pred — run: python scripts/import_cmapss_colab_outputs.py --force" -ForegroundColor Red
    exit 1
}

Write-Host "Building and starting services…" -ForegroundColor Cyan
docker compose up --build -d

Write-Host "Waiting for API health…" -ForegroundColor Cyan
$ok = $false
for ($i = 0; $i - 30; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($r.status -eq "ok") { $ok = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) {
    Write-Host "API not healthy yet — check: docker compose logs api" -ForegroundColor Yellow
} else {
    Write-Host "API healthy. Datasets: $($r.datasets_available -join ', ')" -ForegroundColor Green
}

Write-Host "Ensuring llama3.2 in Ollama container…" -ForegroundColor Cyan
docker compose exec -T ollama ollama pull llama3.2

Write-Host ""
Write-Host "Demo ready:" -ForegroundColor Green
Write-Host "  Local app:  http://localhost:8080"
Write-Host "  API docs:   http://localhost:8080/api/docs"
Write-Host "  Tunnel:     .\deploy\start_tunnel.ps1"
Write-Host "  Stop:       docker compose down"
