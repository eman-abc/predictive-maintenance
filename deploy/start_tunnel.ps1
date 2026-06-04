# Expose http://localhost:8080 on a public HTTPS URL (Cloudflare quick tunnel).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

$cloudflared = Get-CloudflaredExe
if (-not $cloudflared) {
    Write-Host "cloudflared not found (winget may have installed it but PATH is stale)." -ForegroundColor Red
    Write-Host "  1. Close this PowerShell window and open a NEW one, then: cloudflared --version" -ForegroundColor Yellow
    Write-Host "  2. Or run: .\deploy\install_cloudflared.ps1" -ForegroundColor Yellow
    Write-Host "  3. Or use ngrok: .\deploy\start_tunnel_ngrok.ps1" -ForegroundColor Yellow
    exit 1
}

& $cloudflared --version
Write-Host ""
Write-Host "Starting tunnel to http://localhost:8080 ..." -ForegroundColor Cyan
Write-Host "Copy the https://....trycloudflare.com URL for interviewers." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor Yellow
& $cloudflared tunnel --url http://localhost:8080
