# Checks Docker + optional cloudflared before deployment scripts run.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

$fail = $false

if (-not (Test-DockerDaemon)) {
    $fail = $true
    Write-Host "Docker is not running." -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix:" -ForegroundColor Yellow
    Write-Host "  1. Open Docker Desktop from the Start menu"
    Write-Host "  2. Wait until it says Engine running (whale icon steady)"
    Write-Host "  3. Retry: docker compose up -d --build"
    Write-Host ""
    Write-Host "If Docker Desktop is not installed:"
    Write-Host "  https://www.docker.com/products/docker-desktop/"
} else {
    Write-Host "Docker: OK" -ForegroundColor Green
}

$cf = Get-CloudflaredExe
if (-not $cf) {
    Write-Host "cloudflared: not on PATH yet (needed for public HTTPS URL)" -ForegroundColor Yellow
    Write-Host "  Run: .\deploy\install_cloudflared.ps1"
    Write-Host "  Or open a NEW PowerShell after winget install"
    Write-Host "  Or use ngrok: .\deploy\start_tunnel_ngrok.ps1"
} else {
    Write-Host "cloudflared: OK ($cf)" -ForegroundColor Green
}

if ($fail) { exit 1 }
exit 0
