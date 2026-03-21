#Requires -Version 5.1
<#
  Stops local dev listeners for this project: Docker LiveKit, TCP 8000 (API), 5173 (Vite).
  Run:  powershell -ExecutionPolicy Bypass -File scripts\stop_stack.ps1
  Or:   STOP_STACK.bat
#>
param(
  [switch]$SkipDocker
)

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

function Get-ListeningPids([int]$Port) {
  $set = New-Object "System.Collections.Generic.HashSet[int]"
  $raw = netstat -ano 2>$null
  foreach ($line in $raw) {
    if ($line -notmatch "LISTENING") { continue }
    if ($line -notmatch ":$Port\s") { continue }
    if ($line -match "\s(\d+)\s*$") {
      $procId = [int]$Matches[1]
      if ($procId -gt 0 -and $procId -ne 4) {
        [void]$set.Add($procId)
      }
    }
  }
  return @($set)
}

function Stop-Pids([int[]]$Pids, [string]$Reason) {
  foreach ($p in ($Pids | Sort-Object -Unique)) {
    try {
      $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
      $name = if ($proc) { $proc.ProcessName } else { "?" }
      Write-Host "Stopping PID $p ($name) - $Reason"
      Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    } catch {
      Write-Host "Could not stop PID $p : $($_.Exception.Message)" -ForegroundColor Yellow
    }
  }
}

Write-Step "Stop Voice AI Banking local stack"
Write-Host "Repo: $RepoRoot"

if (-not $SkipDocker) {
  Write-Step "Docker Compose down (LiveKit)"
  Push-Location $RepoRoot
  try {
    docker compose version | Out-Null 2>$null
    if ($LASTEXITCODE -eq 0) {
      docker compose -f (Join-Path $RepoRoot "docker-compose.yml") down 2>$null
      Write-Host "docker compose down finished." -ForegroundColor Green
    }
  } catch {
    Write-Host "Docker not available or compose failed (ok if Docker was off)." -ForegroundColor Yellow
  }
  Pop-Location
}

foreach ($port in @(8000, 5173)) {
  Write-Step "Port $port - terminating listeners"
  $pids = Get-ListeningPids -Port $port
  if ($pids.Count -eq 0) {
    Write-Host "Nothing listening on $port."
  } else {
    Stop-Pids -Pids $pids -Reason "free port $port"
  }
}

Write-Host "`nDone. Wait a few seconds, then run START_STACK.bat" -ForegroundColor Green
Write-Host "Verify: netstat -ano | findstr `"LISTENING`" | findstr `"8000 5173`"" -ForegroundColor DarkGray
