# Clean build script for OpenRAI

$ErrorActionPreference = "Stop"

Write-Host "=== OpenRAI Clean Build ===" -ForegroundColor Cyan

# Ensure the script always runs from the project directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Step 1: Remove old build directory
Write-Host ""
Write-Host "[1/4] Cleaning build directory..." -ForegroundColor Yellow

if (Test-Path ".\build") {
    Remove-Item ".\build" -Recurse -Force
    Write-Host "  OK Removed old build directory" -ForegroundColor Green
}
else {
    Write-Host "  OK Build directory does not exist" -ForegroundColor Green
}

# Step 2: Create fresh build directory
Write-Host ""
Write-Host "[2/4] Creating fresh build directory..." -ForegroundColor Yellow

New-Item -ItemType Directory -Path ".\build" -Force | Out-Null

Write-Host "  OK Created build directory" -ForegroundColor Green

# Step 3: Run CMake configuration
Write-Host ""
Write-Host "[3/4] Configuring with CMake..." -ForegroundColor Yellow

Push-Location ".\build"

try {
    & cmake `
        -G "Visual Studio 17 2022" `
        -A x64 `
        ".." 2>&1 | Tee-Object -Variable cmakeOutput

    $cmakeResult = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($cmakeResult -ne 0) {
    Write-Host ""
    Write-Host "  ERROR CMake configuration failed" -ForegroundColor Red
    Write-Host $cmakeOutput
    exit $cmakeResult
}

Write-Host "  OK CMake configuration successful" -ForegroundColor Green

# Step 4: Build Release configuration
Write-Host ""
Write-Host "[4/4] Building Release configuration..." -ForegroundColor Yellow

Push-Location ".\build"

try {
    & cmake `
        --build "." `
        --config Release `
        --parallel 2>&1 | Tee-Object -Variable buildOutput

    $buildResult = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($buildResult -ne 0) {
    Write-Host ""
    Write-Host "  ERROR Build failed" -ForegroundColor Red
    Write-Host $buildOutput
    exit $buildResult
}

Write-Host ""
Write-Host "  OK Build successful" -ForegroundColor Green
Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Cyan

$possibleExecutables = @(
    ".\build\Release\openrai.exe",
    ".\build\openrai.exe"
)

$executable = $possibleExecutables |
    Where-Object { Test-Path $_ } |
    Select-Object -First 1

if ($executable) {
    Write-Host "Executable: $executable" -ForegroundColor White
    Write-Host "Run with: &$executable" -ForegroundColor White
}
else {
    Write-Host "Build completed, but openrai.exe was not found at the expected location." -ForegroundColor Yellow
    Write-Host "Available executables:" -ForegroundColor Yellow

    Get-ChildItem ".\build" -Recurse -Filter "*.exe" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host "  $($_.FullName)" -ForegroundColor White
        }
}

exit 0