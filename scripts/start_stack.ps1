#Requires -Version 5.1
param(
  # Run scripts\stop_stack.ps1 first (docker compose down + kill :8000 and :5173 listeners)
  [switch]$ClearPorts
)
<#
  Full-stack shortcut (Windows). Canonical manual pipeline: README §7b / §7c.
  Starts: Docker LiveKit, FastAPI (:8000), local STT/TTS (:8088/:8089 if free), voice agent, Vite (:5173).
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

function Test-TcpPortOpen {
  param([string]$HostName = "127.0.0.1", [int]$Port, [int]$TimeoutMs = 400)
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($HostName, $Port, $null, $null)
    $wait = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
    if (-not $wait) {
      try { $client.Close() } catch { }
      return $false
    }
    try {
      $client.EndConnect($iar)
    } catch {
      try { $client.Close() } catch { }
      return $false
    }
    $client.Close()
    return $true
  } catch {
    return $false
  }
}

# PowerShell 5.1: avoid nested `"` inside double-quoted Start-Process args (breaks on `&&`).
function Start-CmdKeepOpen {
  param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$CommandTail
  )
  $q = $WorkDir.Replace('"', '""')
  $line = ('cd /d "{0}" && {1}' -f $q, $CommandTail)
  Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $line) -WindowStyle Normal
}

Write-Step "Voice AI Banking - local stack"
Write-Host "Repo: $RepoRoot"

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "ERROR: No .venv found. Run first:" -ForegroundColor Red
  Write-Host "  scripts\setup_env.bat" -ForegroundColor Yellow
  Write-Host "Then copy .env.example to .env and set GEMINI_API_KEY (optional but recommended)." -ForegroundColor Yellow
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

  Start-CmdKeepOpen -WorkDir $RepoRoot -CommandTail 'call .venv\Scripts\activate.bat && python run_runtime_api.py'

  if (-not (Wait-BankingApi -MaxSeconds 120)) {
    Write-Host "ERROR: API did not become ready in time. Check the 'Banking API' window for errors." -ForegroundColor Red
    exit 1
  }
}

Write-Step "Local Armenian STT / TTS (HTTP)"
$wm = $env:VOICE_WHISPER_MODEL
if ($wm -and $wm.Trim()) {
  $sttModel = $wm.Trim()
} else {
  $sttModel = "medium"
}
if (-not (Test-TcpPortOpen -Port 8088)) {
  Write-Host "Starting STT on :8088 (Whisper model=$sttModel; set VOICE_WHISPER_MODEL to override)..." -ForegroundColor DarkGray
  $sttCmdTail = 'call .venv\Scripts\activate.bat && python scripts\voice_http_stt_server.py --model ' + $sttModel
  Start-CmdKeepOpen -WorkDir $RepoRoot -CommandTail $sttCmdTail
  Start-Sleep -Seconds 2
} else {
  Write-Host "Port 8088 already in use — skipping STT server (reuse existing)." -ForegroundColor DarkGray
}
if (-not (Test-TcpPortOpen -Port 8089)) {
  Write-Host "Starting TTS on :8089..." -ForegroundColor DarkGray
  Start-CmdKeepOpen -WorkDir $RepoRoot -CommandTail 'call .venv\Scripts\activate.bat && python scripts\voice_http_tts_server.py'
  Start-Sleep -Seconds 1
} else {
  Write-Host "Port 8089 already in use — skipping TTS server (reuse existing)." -ForegroundColor DarkGray
}
Write-Host "Ensure .env has VOICE_STT_ENDPOINT=http://127.0.0.1:8088/transcribe and VOICE_TTS_ENDPOINT=http://127.0.0.1:8089/synthesize" -ForegroundColor DarkGray

Write-Step "Voice agent (LiveKit participant)"
$env:NO_PAUSE = "1"
Start-CmdKeepOpen -WorkDir $RepoRoot -CommandTail 'call scripts\run_voice_agent.bat'

Start-Sleep -Seconds 2

Write-Step "React UI (Vite)"
$fe = Join-Path $RepoRoot "frontend-react"
if (-not (Test-Path (Join-Path $fe "package.json"))) {
  Write-Host "ERROR: frontend-react missing." -ForegroundColor Red
  exit 1
}
$npmCmd = if (Test-Path (Join-Path $fe "node_modules")) { 'npm run dev' } else { 'npm install && npm run dev' }
Start-CmdKeepOpen -WorkDir $fe -CommandTail $npmCmd

Write-Step "Done"
Write-Host "  UI:      http://127.0.0.1:5173" -ForegroundColor White
Write-Host "  API:     http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  LiveKit: ws://127.0.0.1:7880" -ForegroundColor White
Write-Host "  STT:     http://127.0.0.1:8088/health (first run downloads Whisper weights)" -ForegroundColor White
Write-Host "  TTS:     http://127.0.0.1:8089/health" -ForegroundColor White
Write-Host "`nClose the cmd windows to stop API / voice / STT / TTS / UI. Docker: docker compose down" -ForegroundColor DarkGray
