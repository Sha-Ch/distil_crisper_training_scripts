#Requires -Version 5.1
<#
=============================================================================
 Distil-CrisperWhisper - Windows startup helper
=============================================================================
 One command to get going on Windows:
   1) Creates/updates a local Python venv (.venv) and runs the import-light tests
   2) Lets you PICK the storage folder (e.g. your 10 TiB drive) that will hold
      EVERYTHING the pipeline writes -- HF/teacher cache, datasets, pseudo-labels
      + audio, checkpoints, output. It is bind-mounted at /workspace in the
      container, so this one folder is your whole data root.
   3) Writes .env (DATA_DIR + HF_TOKEN/HF_USERNAME) used by docker-compose
   4) Optionally builds + launches the Docker (WSL2) container

 Usage (from this folder, in a PowerShell window):
   .\start.ps1                                  # interactive: tests + folder picker + .env
   .\start.ps1 -DataDir 'D:\distil_data'        # skip the picker, use this folder
   .\start.ps1 -DataDir 'D:\distil_data' -Build -Run
   .\start.ps1 -SkipTests

 IMPORTANT: the GPU pipeline runs INSIDE Docker/WSL2 (Linux). This Windows venv
 is ONLY for the import-light tests + config checks -- it does NOT install torch.
 If PowerShell blocks the script, run once:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
=============================================================================
#>
[CmdletBinding()]
param(
    [string]$DataDir,
    [switch]$SkipTests,
    [switch]$Build,
    [switch]$Run
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

function Write-Section($t) { Write-Host ""; Write-Host ("=== " + $t + " ===") -ForegroundColor Cyan }

function Find-Python {
    foreach ($c in @('py', 'python', 'python3')) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    throw "Python 3.10+ not found. Install it from https://www.python.org/downloads/ and re-run."
}

function Select-DataDir($default) {
    # Try a graphical folder picker; fall back to a text prompt if unavailable.
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
        $dlg.Description = "Pick the storage folder for datasets / teacher cache / pseudo-labels / checkpoints (e.g. your 10 TiB drive)"
        $dlg.ShowNewFolderButton = $true
        if ($default -and (Test-Path $default)) { $dlg.SelectedPath = $default }
        if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dlg.SelectedPath }
    } catch {
        Write-Host "  (folder dialog unavailable, using text input)" -ForegroundColor DarkGray
    }
    $p = Read-Host "Enter storage folder path (e.g. D:\distil_data)"
    return $p.Trim()
}

# --- 1) venv + tests -------------------------------------------------------
Write-Section "Python venv + import-light tests"
$py = Find-Python
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment at .venv ..."
    & $py -m venv (Join-Path $Root ".venv")
}
Write-Host "Installing test dependencies (pytest, pyyaml) ..."
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r (Join-Path $Root "requirements-dev.txt") --quiet
if ($SkipTests) {
    Write-Host "Skipping tests (-SkipTests)."
} else {
    Write-Host "Running tests ..."
    & $venvPy -m pytest tests/ -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed (exit $LASTEXITCODE)." }
    Write-Host "Tests passed." -ForegroundColor Green
}

# --- 2) storage folder -----------------------------------------------------
Write-Section "Storage folder (bind-mounted at /workspace in the container)"
$envPath = Join-Path $Root ".env"
$envMap = [ordered]@{}
if (Test-Path $envPath) {
    foreach ($line in Get-Content $envPath) {
        if ($line -match '^\s*([^#=][^=]*)=(.*)$') { $envMap[$matches[1].Trim()] = $matches[2] }
    }
}
$existing = $null
if ($envMap.Contains('DATA_DIR') -and $envMap['DATA_DIR']) { $existing = ($envMap['DATA_DIR'] -replace '/', '\') }
if (-not $DataDir) { $DataDir = Select-DataDir $existing }
if (-not $DataDir) { throw "No storage folder selected." }
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Write-Host ("Storage folder: " + $DataDir)
Write-Host "  -> holds: hf_cache (teacher+datasets), pseudo_labels (+audio), checkpoints, output"
try {
    $qual = (Split-Path -Qualifier $DataDir).TrimEnd(':')
    $drv = Get-PSDrive -Name $qual -ErrorAction SilentlyContinue
    if ($drv) { Write-Host ("Free space on " + $qual + ": " + [math]::Round($drv.Free / 1GB, 1) + " GB") }
} catch {}

# --- 3) .env ---------------------------------------------------------------
Write-Section "Credentials (.env)"
if (-not $envMap.Contains('HF_TOKEN') -or [string]::IsNullOrWhiteSpace($envMap['HF_TOKEN'])) {
    $tok = Read-Host "HF_TOKEN (required for gated datasets; leave blank to fill in .env later)"
    if ($tok) { $envMap['HF_TOKEN'] = $tok.Trim() } elseif (-not $envMap.Contains('HF_TOKEN')) { $envMap['HF_TOKEN'] = '' }
}
if (-not $envMap.Contains('HF_USERNAME') -or [string]::IsNullOrWhiteSpace($envMap['HF_USERNAME'])) {
    $usr = Read-Host "HF_USERNAME (only needed for push_to_hub; optional)"
    if ($usr) { $envMap['HF_USERNAME'] = $usr.Trim() } elseif (-not $envMap.Contains('HF_USERNAME')) { $envMap['HF_USERNAME'] = '' }
}
if (-not $envMap.Contains('WANDB_API_KEY')) { $envMap['WANDB_API_KEY'] = '' }
$envMap['DATA_DIR'] = ($DataDir -replace '\\', '/')   # forward slashes for docker-compose
$out = foreach ($k in $envMap.Keys) { "$k=$($envMap[$k])" }
Set-Content -Path $envPath -Value $out -Encoding ASCII
Write-Host (".env written (DATA_DIR=" + $envMap['DATA_DIR'] + ")") -ForegroundColor Green
if ([string]::IsNullOrWhiteSpace($envMap['HF_TOKEN'])) {
    Write-Host "  NOTE: HF_TOKEN is empty -- gated datasets will be skipped until you set it in .env." -ForegroundColor Yellow
}

# --- 4) Docker -------------------------------------------------------------
Write-Section "Docker (WSL2)"
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    Write-Host "Docker not found. Install Docker Desktop with the WSL2 backend, then re-run with -Build -Run." -ForegroundColor Yellow
} else {
    if ($Build) {
        Write-Host "Building image (docker compose build) ..."
        docker compose build
        Write-Host "Built one image: distil-crisperwhisper:local (single-stage; + the cached nvidia/cuda base)." -ForegroundColor DarkGray
        Write-Host "A REBUILD untags the previous image, leaving one dangling <none>; remove with: docker image prune -f" -ForegroundColor DarkGray
    }
    if ($Run)   { Write-Host "Launching container (shell at /app/scripts; --rm removes the container on exit) ..."; docker compose run --rm distil }
    if (-not $Build -and -not $Run) {
        Write-Host "Ready. Next steps:" -ForegroundColor Green
        Write-Host "  docker compose build           # one image: distil-crisperwhisper:local (+ cached cuda base)"
        Write-Host "  docker compose run --rm distil # --rm leaves no leftover container"
        Write-Host "Then inside the container (see LOCAL_4090.md):"
        Write-Host "  python3 02_generate_pseudo_labels_multi_gpu.py --config ../config.local.yaml --datasets librispeech --max-samples 50"
        Write-Host "Cleanup after rebuilds: docker image prune -f   (dangling only)  /  docker builder prune (cache)"
    }
}
