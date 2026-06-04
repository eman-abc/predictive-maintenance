# Two-process local deployment (no Docker): API + Streamlit thin client.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:API_BASE_URL = "http://localhost:8000"
$env:OLLAMA_BASE_URL = "http://localhost:11434"

Write-Host "Terminal 1: uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
Write-Host "Terminal 2: `$env:API_BASE_URL='http://localhost:8000'; streamlit run dashboard/app.py"
Write-Host "Tunnel (optional): cloudflared tunnel --url http://localhost:8501"
