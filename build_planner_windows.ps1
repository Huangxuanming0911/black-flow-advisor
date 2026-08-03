param(
    [switch]$SkipInstall,
    [switch]$SkipPlannerBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Spec = Join-Path $ProjectRoot "packaging\BlackFlowRoutePlanner.spec"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found: $Python"
}

Push-Location $ProjectRoot
try {
    if (-not $SkipInstall) {
        & $Python -m pip install -e ".[desktop]"
        if ($LASTEXITCODE -ne 0) { throw "Desktop dependency install failed" }
    }
    if (-not $SkipPlannerBuild) {
        & $Python tools\build_route_planner.py
        if ($LASTEXITCODE -ne 0) { throw "Route planner page build failed" }
    }
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath (Join-Path $ProjectRoot "dist") `
        --workpath (Join-Path $ProjectRoot "build") `
        $Spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Build complete: $(Join-Path $ProjectRoot 'dist\BlackFlowRoutePlanner')"
