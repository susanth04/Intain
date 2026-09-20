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

$required = @(
  "data/raw/loan_monthly_performance_train.csv",
  "data/raw/loan_monthly_performance_test.csv",
  "data/raw/loan_static_attributes.csv",
  "data/processed/imputer.pkl",
  "data/processed/models/next_3m_delinquency_flag_lgbm_cal.pkl",
  "data/processed/models/next_state_lgbm.pkl"
)
foreach ($relativePath in $required) {
  if (-not (Test-Path (Join-Path $dest $relativePath))) {
    throw "Required deployment artifact missing: $relativePath"
  }
}

Write-Host "Done. Docker can now COPY ./engine"
