# Automatic ImGui Setup for OpenRAI Chat
# Installs vcpkg, ImGui, and configures build

param(
    [switch]$SkipVcpkg = $false
)

Write-Host "=== OpenRAI ImGui Setup ===" -ForegroundColor Cyan
Write-Host ""

$VCPKG_PATH = "C:\vcpkg"
$OPENRAI_ROOT = "C:\dev\openRAI"

# Step 1: Install vcpkg if not present
if (-not $SkipVcpkg) {
    Write-Host "[1/5] Checking vcpkg..." -ForegroundColor Yellow

    if (-not (Test-Path $VCPKG_PATH)) {
        Write-Host "  vcpkg not found, installing..." -ForegroundColor White
        git clone https://github.com/Microsoft/vcpkg.git $VCPKG_PATH
        Push-Location $VCPKG_PATH
        .\bootstrap-vcpkg.bat
        Pop-Location
        Write-Host "  ? vcpkg installed" -ForegroundColor Green
    } else {
        Write-Host "  ? vcpkg already installed" -ForegroundColor Green
    }

    # Add to PATH for this session
    $env:PATH += ";$VCPKG_PATH"
} else {
    Write-Host "[1/5] Skipping vcpkg check (--SkipVcpkg)" -ForegroundColor Yellow
}

# Step 2: Install ImGui via vcpkg
Write-Host "[2/5] Installing ImGui..." -ForegroundColor Yellow
& "$VCPKG_PATH\vcpkg.exe" install imgui[core,win32-binding,dx11-binding]:x64-windows

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ? ImGui installed" -ForegroundColor Green
} else {
    Write-Host "  ? ImGui installation failed" -ForegroundColor Red
    exit 1
}

# Step 3: Update CMakeLists.txt
Write-Host "[3/5] Updating CMakeLists.txt..." -ForegroundColor Yellow

$cmakePath = Join-Path $OPENRAI_ROOT "CMakeLists.txt"
$cmakeContent = Get-Content $cmakePath -Raw

if ($cmakeContent -notmatch "find_package\(imgui") {
    Write-Host "  Adding ImGui package..." -ForegroundColor White

    # Add find_package after Threads
    $cmakeContent = $cmakeContent -replace `
        "(find_package\(Threads REQUIRED\))", `
        "`$1`nfind_package(imgui CONFIG REQUIRED)"

    # Update openrai_chat target
    $cmakeContent = $cmakeContent -replace `
        "(target_link_libraries\(openrai_chat PRIVATE Threads::Threads cudart cublas\))", `
        "target_link_libraries(openrai_chat PRIVATE Threads::Threads cudart cublas imgui::imgui d3d11 dxgi)`n    target_compile_definitions(openrai_chat PRIVATE OPENRAI_USE_IMGUI)"

    $cmakeContent = $cmakeContent -replace `
        "(target_link_libraries\(openrai_chat PRIVATE Threads::Threads\))", `
        "target_link_libraries(openrai_chat PRIVATE Threads::Threads imgui::imgui d3d11 dxgi)`n    target_compile_definitions(openrai_chat PRIVATE OPENRAI_USE_IMGUI)"

    Set-Content -Path $cmakePath -Value $cmakeContent
    Write-Host "  ? CMakeLists.txt updated" -ForegroundColor Green
} else {
    Write-Host "  ? CMakeLists.txt already configured" -ForegroundColor Green
}

# Step 4: Regenerate build
Write-Host "[4/5] Regenerating CMake..." -ForegroundColor Yellow

Push-Location $OPENRAI_ROOT

Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

cmake -B build `
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_PATH\scripts\buildsystems\vcpkg.cmake" `
    -DCMAKE_BUILD_TYPE=Debug

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ? CMake configured" -ForegroundColor Green
} else {
    Write-Host "  ? CMake configuration failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Step 5: Build with ImGui
Write-Host "[5/5] Building openrai_chat with ImGui..." -ForegroundColor Yellow

Push-Location $OPENRAI_ROOT

cmake --build build --config Debug --target openrai_chat

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ? Build successful!" -ForegroundColor Green
} else {
    Write-Host "  ? Build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Success!
Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "ImGui visual interface is now active!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run the chat interface:" -ForegroundColor White
Write-Host "  cd C:\dev\openRAI" -ForegroundColor DarkGray
Write-Host "  .\build\Debug\openrai_chat.exe" -ForegroundColor DarkGray
Write-Host ""
Write-Host "You should see a Windows GUI window instead of console!" -ForegroundColor Yellow
Write-Host ""
