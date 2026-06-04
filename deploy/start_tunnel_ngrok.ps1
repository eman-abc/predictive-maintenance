# Alternative public URL if cloudflared/winget fails: ngrok on port 8080.
$ErrorActionPreference = "Stop"

if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Host "Install ngrok: winget install ngrok.ngrok" -ForegroundColor Red
    Write-Host "Or: https://ngrok.com/download" -ForegroundColor Red
    exit 1
}

Write-Host "Starting ngrok -> http://localhost:8080" -ForegroundColor Cyan
Write-Host "Copy the https://....ngrok-free.app URL for interviewers." -ForegroundColor Green
ngrok http 8080
