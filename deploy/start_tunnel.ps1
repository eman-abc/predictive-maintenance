# Expose http://localhost:8080 on a public HTTPS URL (Cloudflare quick tunnel).
$ErrorActionPreference = "Stop"

try {
    cloudflared --version | Out-Null
} catch {
    Write-Host "Install cloudflared: winget install Cloudflare.cloudflared" -ForegroundColor Red
    Write-Host "Or download: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
}

Write-Host "Starting tunnel to http://localhost:8080 …" -ForegroundColor Cyan
Write-Host "Share the https://*.trycloudflare.com URL with interviewers." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor Yellow
cloudflared tunnel --url http://localhost:8080
