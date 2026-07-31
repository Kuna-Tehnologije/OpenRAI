# Clean build script for OpenRAI

Write-Host "=== OpenRAI Clean Build ===" -ForegroundColor Cyan

# Step 1: Remove old build directory
Write-Host "`n[1/4] Cleaning build directory..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item "build" -Recurse -Force
    Write-Host "  ? Removed old build directory" -ForegroundColor Green
}

# Step 2: Create fresh build directory
Write-Host "`n[2/4] Creating fresh build directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "build" | Out-Null
Write-Host "  ? Created build directory" -ForegroundColor Green

# Step 3: Run CMake configuration
Write-Host "`n[3/4] Configuring with CMake..." -ForegroundColor Yellow
Push-Location build
cmake -G Ninja .. 2>&1 | Tee-Object -Variable cmakeOutput
$cmakeResult = $LASTEXITCODE
Pop-Location

if ($cmakeResult -eq 0) {
    Write-Host "  ? CMake configuration successful" -ForegroundColor Green
} else {
    Write-Host "  ? CMake configuration failed!" -ForegroundColor Red
    Write-Host $cmakeOutput
    exit 1
}

# Step 4: Build with Ninja
Write-Host "`n[4/4] Building with Ninja..." -ForegroundColor Yellow
Push-Location build
ninja 2>&1 | Tee-Object -Variable ninjaOutput
$ninjaResult = $LASTEXITCODE
Pop-Location

if ($ninjaResult -eq 0) {
    Write-Host "`n  ? Build successful!" -ForegroundColor Green
    Write-Host "`n=== Build Complete ===" -ForegroundColor Cyan
    Write-Host "Run with: cd build; .\openrai.exe" -ForegroundColor White
} else {
    Write-Host "`n  ? Build failed!" -ForegroundColor Red
    Write-Host $ninjaOutput
    exit 1
}