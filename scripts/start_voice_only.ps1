#Requires -Version 5.1
param(
  [switch]$SkipVoiceServersHealthCheck
)
<#
  Start STT (:8088), TTS (:8089), and LiveKit voice agent only.
  Use after text chat stack (API + Vite + Docker). Ensures banking API is up for LiveKit token.
  Run: powershell -ExecutionPolicy Bypass -File scripts\start_voice_only.ps1
#>
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

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

function Start-CmdKeepOpen {
  param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$CommandTail
  )
  Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $CommandTail) -WindowStyle Normal -WorkingDirectory $WorkDir
}

function Start-PythonScriptCmdK {
  param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$ArgumentsLine
  )
  $cmdLine = ('"{0}" {1}' -f $PythonExe, $ArgumentsLine)
  Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $cmdLine) -WindowStyle Normal -WorkingDirectory $WorkDir
}

function Test-VoiceUseMockEnv {
  $v = $env:VOICE_USE_MOCK
  if (-not $v) { return $false }
  return $v.Trim().ToLowerInvariant() -match '^(1|true|yes)$'
}

function Wait-VoiceServerHealth {
  param(
    [Parameter(Mandatory = $true)][string]$Uri,
    [int]$MaxSeconds = 120,
    [Parameter(Mandatory = $true)][string]$ServiceName,
    [string]$StillWaitingHint = ""
  )
  $deadline = (Get-Date).AddSeconds($MaxSeconds)
  $tick = 0
  Write-Host "Waiting for $ServiceName at $Uri (max ${MaxSeconds}s)..."
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 8
      if ($r.StatusCode -eq 200) {
        Write-Host "$ServiceName OK." -ForegroundColor Green
        return $true
      }
    } catch {
    }
    $tick++
    if ($tick -eq 1 -or $tick % 8 -eq 0) {
      $extra = if ($StillWaitingHint) { " $StillWaitingHint" } else { "" }
      Write-Host "  ... still waiting ($ServiceName).$extra" -ForegroundColor DarkGray
    }
    Start-Sleep -Seconds 3
  }
  return $false
}

Write-Step "Voice add-on (STT + TTS + LiveKit agent)"
Write-Host "Repo: $RepoRoot"

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "ERROR: No .venv found. Run scripts\setup_env.bat first." -ForegroundColor Red
  exit 1
}

try {
  docker compose version | Out-Null
  docker compose -f (Join-Path $RepoRoot "docker-compose.yml") up -d | Out-Null
} catch { }

if (-not (Test-BankingApiReady)) {
  Write-Host "Banking API not on :8000 - voice agent needs it for LiveKit token. Waiting..." -ForegroundColor Yellow
  if (-not (Wait-BankingApi -MaxSeconds 90)) {
    Write-Host "ERROR: Start the API first (e.g. START_TEXT_CHAT.bat or run_runtime_api.py)." -ForegroundColor Red
    exit 1
  }
} else {
  Write-Host "Banking API already up on :8000." -ForegroundColor Green
}

$wm = $env:VOICE_WHISPER_MODEL
$sttModel = "small"
if ($null -ne $wm -and ($wm.Trim().Length -gt 0)) {
  $sttModel = $wm.Trim()
}

if (-not (Test-TcpPortOpen -Port 8089)) {
  Write-Host "Starting TTS on :8089..." -ForegroundColor DarkGray
  Start-PythonScriptCmdK -WorkDir $RepoRoot -PythonExe $py -ArgumentsLine "scripts\voice_http_tts_server.py"
  Start-Sleep -Seconds 3
} else {
  Write-Host "Port 8089 already in use - skipping TTS (reuse existing)." -ForegroundColor DarkGray
}

if (-not (Test-TcpPortOpen -Port 8088)) {
  Write-Host ('Starting STT on :8088 (Whisper model={0}).' -f $sttModel) -ForegroundColor DarkGray
  $sttArgs = "scripts\voice_http_stt_server.py --model $sttModel"
  Start-PythonScriptCmdK -WorkDir $RepoRoot -PythonExe $py -ArgumentsLine $sttArgs
  Start-Sleep -Seconds 3
} else {
  Write-Host "Port 8088 already in use - skipping STT (reuse existing)." -ForegroundColor DarkGray
}

Write-Host "Ensure .env has VOICE_STT_ENDPOINT and VOICE_TTS_ENDPOINT for :8088 / :8089." -ForegroundColor DarkGray

if (-not $SkipVoiceServersHealthCheck -and -not (Test-VoiceUseMockEnv)) {
  if (-not (Wait-VoiceServerHealth -Uri "http://127.0.0.1:8088/health" -MaxSeconds 360 -ServiceName "STT (Whisper on :8088)" -StillWaitingHint "503 means Whisper still loading.")) {
    Write-Host "ERROR: STT not ready. Check the STT cmd window." -ForegroundColor Red
    exit 1
  }
  if (-not (Wait-VoiceServerHealth -Uri "http://127.0.0.1:8089/health" -MaxSeconds 240 -ServiceName "TTS (Edge on :8089)" -StillWaitingHint "Check TTS window for errors.")) {
    Write-Host "ERROR: TTS not ready. Check the TTS cmd window." -ForegroundColor Red
    exit 1
  }
} else {
  if (Test-VoiceUseMockEnv) {
    Write-Host "VOICE_USE_MOCK: skipping STT and TTS health wait." -ForegroundColor DarkGray
  } else {
    Write-Host "-SkipVoiceServersHealthCheck: verify STT and TTS yourself." -ForegroundColor Yellow
  }
}

Write-Step "Voice agent"
$env:NO_PAUSE = "1"
Start-CmdKeepOpen -WorkDir $RepoRoot -CommandTail 'call scripts\run_voice_agent.bat'

Write-Step "Done"
Write-Host "  STT:   http://127.0.0.1:8088/health" -ForegroundColor White
Write-Host "  TTS:   http://127.0.0.1:8089/health" -ForegroundColor White
Write-Host '  Agent: LiveKit room (see voice agent window)' -ForegroundColor White
Write-Host ""
