# Shared helpers for deploy/*.ps1

function Refresh-ShellPath {
    $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Get-CloudflaredExe {
    Refresh-ShellPath
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $candidates = @(
        "$env:ProgramFiles\cloudflared\cloudflared.exe",
        "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
        "$env:ProgramFiles\Cloudflare\cloudflared\cloudflared.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\cloudflared.exe"
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }
    return $null
}

function Test-DockerDaemon {
    docker info 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}
