#Requires -Version 5.1
<#
  Stop everything, then start the full dev stack (API, STT, TTS health waits, voice agent, Vite).
  Use this file with -File so nested shells do not strip variables when chaining commands.
  Run: powershell -ExecutionPolicy Bypass -File scripts\run_full_pipeline.ps1
#>
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

& (Join-Path $PSScriptRoot "stop_stack.ps1")
Start-Sleep -Seconds 4

& (Join-Path $PSScriptRoot "start_stack.ps1")
