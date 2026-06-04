# One terminal: start Docker stack, then public HTTPS tunnel (blocks until Ctrl+C).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& "$Root\deploy\run_demo.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Starting public tunnel (this terminal stays open)..." -ForegroundColor Cyan
& "$Root\deploy\start_tunnel.ps1"
