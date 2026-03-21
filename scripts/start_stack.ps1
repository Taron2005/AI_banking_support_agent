#Requires -Version 5.1
param(
  # Run scripts\stop_stack.ps1 first (docker compose down + kill :8000 and :5173 listeners)
  [switch]$ClearPorts
)
<#
  Starts the full local stack: Docker LiveKit, FastAPI (:8000), voice agent, Vite (:5173).
  Run from repo root:  powershell -ExecutionPolicy Bypass -File scripts\start_stack.ps1
  Fresh start:         powershell -ExecutionPolicy Bypass -File scripts\start_stack.ps1 -ClearPorts
  Or double-click: START_STACK.bat
#>
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ($ClearPorts) {
  Write-Host "`n-ClearPorts: stopping listeners and Docker LiveKit first..." -ForegroundColor Cyan
  & (Join-Path $PSScriptRoot "stop_stack.ps1")
  Start-Sleep -Seconds 3
}

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

function Test-BankingApiReady {
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/livekit/config" -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -ne 200) { return $false }
    $j = $r.Content | ConvertFrom-Json
    return [bool]($j.livekit_url)
  } catch {
    return $false
  }
}

function Wait-BankingApi {
  param([int]$MaxSeconds = 120)
  $deadline = (Get-Date).AddSeconds($MaxSeconds)
  Write-Host "Waiting for http://127.0.0.1:8000/api/livekit/config (max ${MaxSeconds}s)..."
  while ((Get-Date) -lt $deadline) {
    if (Test-BankingApiReady) {
      Write-Host "API is ready." -ForegroundColor Green
      return $true
    }
    Start-Sleep -Seconds 1
  }
  return $false
}

Write-Step "Voice AI Banking - local stack"
Write-Host "Repo: $RepoRoot"

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "ERROR: No .venv found. Run first:" -ForegroundColor Red
  Write-Host "  scripts\setup_env.bat" -ForegroundColor Yellow
  Write-Host "Then copy .env.example to .env and set GROQ_API_KEY (optional but recommended)." -ForegroundColor Yellow
  exit 1
}

Write-Step "Docker LiveKit"
try {
  docker compose version | Out-Null
  docker compose -f (Join-Path $RepoRoot "docker-compose.yml") up -d
  if ($LASTEXITCODE -ne 0) { throw "docker compose failed" }
  Write-Host "LiveKit container started (signaling ws://127.0.0.1:7880)." -ForegroundColor Green
} catch {
  Write-Host "WARNING: Docker step failed. Install/start Docker Desktop, then run: docker compose up -d" -ForegroundColor Yellow
  Write-Host $_.Exception.Message
}

Write-Step "FastAPI backend (:8000)"
$apiAlready = Test-BankingApiReady
if ($apiAlready) {
  Write-Host "Banking API already responding on :8000 - not starting a second server." -ForegroundColor Green
} else {
  $probe = $null
  try {
    $probe = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
  } catch { }
  if ($probe -and $probe.StatusCode -eq 200 -and -not (Test-BankingApiReady)) {
    Write-Host "ERROR: Something on :8000 has /health but NOT /api/livekit/config (wrong app)." -ForegroundColor Red
    Write-Host "Run STOP_STACK.bat (repo root) to free ports 8000/5173 and stop LiveKit, then run START_STACK.bat again." -ForegroundColor Yellow
    Write-Host "Or: netstat -ano | findstr :8000   then  taskkill /PID <pid> /F" -ForegroundColor DarkGray
    exit 1
  }

  Start-Process -FilePath "cmd.exe" -ArgumentList @(
    "/k",
    "cd /d `"$RepoRoot`" && call .venv\Scripts\activate.bat && python run_runtime_api.py"
  ) -WindowStyle Normal

  if (-not (Wait-BankingApi -MaxSeconds 120)) {
    Write-Host "ERROR: API did not become ready in time. Check the 'Banking API' window for errors." -ForegroundColor Red
    exit 1
  }
}

Write-Step "Voice agent (LiveKit participant)"
$env:NO_PAUSE = "1"
Start-Process -FilePath "cmd.exe" -ArgumentList @(
  "/k",
  "cd /d `"$RepoRoot`" && call scripts\run_voice_agent.bat"
) -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Step "React UI (Vite)"
$fe = Join-Path $RepoRoot "frontend-react"
if (-not (Test-Path (Join-Path $fe "package.json"))) {
  Write-Host "ERROR: frontend-react missing." -ForegroundColor Red
  exit 1
}
$npmCmd = if (Test-Path (Join-Path $fe "node_modules")) { "npm run dev" } else { "npm install && npm run dev" }
Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", "cd /d `"$fe`" && $npmCmd") -WindowStyle Normal

Write-Step "Done"
Write-Host "  UI:      http://127.0.0.1:5173" -ForegroundColor White
Write-Host "  API:     http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  LiveKit: ws://127.0.0.1:7880" -ForegroundColor White
Write-Host "`nClose the cmd windows to stop API / voice / UI. Docker: docker compose down" -ForegroundColor DarkGray
