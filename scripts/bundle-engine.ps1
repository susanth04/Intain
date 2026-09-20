# Copy the loan-perf-engine into ./engine before a Render Docker build.
# Usage (PowerShell):
#   .\scripts\bundle-engine.ps1
# Then commit/push, or build with:
#   docker compose up --build

param(
  [string]$EngineRoot = "C:\Users\susan\OneDrive\Desktop\Intain\loan-perf-engine"
)

$dest = Join-Path $PSScriptRoot "..\engine"
if (-not (Test-Path $EngineRoot)) {
  throw "Engine not found at $EngineRoot"
}

Write-Host "Bundling $EngineRoot -> $dest"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Path $dest | Out-Null

robocopy $EngineRoot $dest /E /XD venv .venv __pycache__ site notebooks .git /NFL /NDL /NJH /NJS
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }
Write-Host "Done. Docker can now COPY ./engine"
