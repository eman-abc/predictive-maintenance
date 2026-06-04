# Install cloudflared on Windows (tunnel for public demo URL).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

$existing = Get-CloudflaredExe
if ($existing) {
    & $existing --version
    Write-Host "cloudflared: $existing" -ForegroundColor Green
    exit 0
}

Write-Host "Installing cloudflared..." -ForegroundColor Cyan

$installed = $false
foreach ($id in @("Cloudflare.cloudflared", "cloudflare.cloudflared")) {
    Write-Host "winget install --id $id -e"
    winget install --id $id -e --accept-package-agreements --accept-source-agreements 2>&1
    if ($LASTEXITCODE -eq 0) {
        $installed = $true
        break
    }
}

Refresh-ShellPath
$existing = Get-CloudflaredExe

if (-not $existing) {
    Write-Host "Downloading cloudflared from GitHub..." -ForegroundColor Yellow
    $dest = "$env:ProgramFiles\cloudflared"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $exe = Join-Path $dest "cloudflared.exe"
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    $existing = $exe
    Write-Host "Installed: $exe" -ForegroundColor Green
    Write-Host "Add to PATH (optional): $dest" -ForegroundColor Yellow
}

if ($existing) {
    & $existing --version
    Write-Host "Use: .\deploy\start_tunnel.ps1  (works without reopening terminal)" -ForegroundColor Green
} else {
    Write-Host "Install failed. Open a NEW PowerShell after winget, or use ngrok." -ForegroundColor Red
    exit 1
}
