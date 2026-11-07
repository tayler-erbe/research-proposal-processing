# One-shot local run of the research_intake prototype on the paired sample.
# Everything stays on this machine: local Ollama only, private output folder outside the repo.
#
# Usage (from the repo folder, VPN not needed):
#   powershell -ExecutionPolicy Bypass -File scripts\run_research_intake.ps1 -TextModel qwen2.5:7b
# Optional: -VisionModel qwen2.5vl:7b  -Limit 20  -Private C:\research-intake-private
param(
    [string]$TextModel,
    [string]$VisionModel,
    [int]$Limit = 20,
    [string]$Private = "$env:USERPROFILE\research-intake-private"
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = if (Test-Path .venv\Scripts\python.exe) { '.venv\Scripts\python.exe' } else { 'python' }

if (-not $TextModel) {
    Write-Host "Installed Ollama models:"; ollama list
    Write-Host "`nRe-run with -TextModel <one of the names above> (and optionally -VisionModel for scanned pages)."
    exit 1
}
if (-not (Test-Path storage\linkage\paired_documents_complete.parquet)) {
    throw "storage\linkage\paired_documents_complete.parquet not found. Run this from the repo folder that holds the paired sample."
}

Write-Host "Installing/confirming packages..."
& $py -m pip install -q -r requirements-intake.txt

function Step($label, [string[]]$cmd) {
    Write-Host "`n== $label"
    & $py -m research_intake --root $Private @cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $label" }
}

Step 'Ingest paired sample'   @('ingest-paired', '--linkage', 'storage\linkage')
Step 'Classify documents'     @('classify')
if ($VisionModel) {
    Step 'Plan OCR of unread pages' @('plan', '--stage', 'page_ocr', '--model', $VisionModel)
    Step 'Run OCR'                  @('run',  '--stage', 'page_ocr', '--model', $VisionModel)
}
Step 'Plan field extraction'  @('plan', '--stage', 'extract', '--model', $TextModel)
Step "Run extraction (first $Limit chunks)" @('run', '--stage', 'extract', '--model', $TextModel, '--limit', "$Limit")
Step 'Export tables and review page' @('export')

Write-Host "`nDone. Re-run the same command to continue where it stopped (finished work is skipped)."
Write-Host "Results: $Private\tables\   Review page: $Private\review.html"
Start-Process "$Private\review.html"
