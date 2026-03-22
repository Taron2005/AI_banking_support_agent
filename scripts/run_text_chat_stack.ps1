#Requires -Version 5.1
<#
  Stop all project listeners, then start Docker LiveKit + API + Vite only (no STT/TTS/voice agent).
  Run: powershell -ExecutionPolicy Bypass -File scripts\run_text_chat_stack.ps1
#>
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

& (Join-Path $PSScriptRoot "stop_stack.ps1")
Start-Sleep -Seconds 4

& (Join-Path $PSScriptRoot "start_stack.ps1") -TextChatOnly
