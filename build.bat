@echo off
REM Build script za OpenRAI PoC
REM Trebam cl.exe compiler (Visual Studio). CPU-only build; za CUDA streams koristi CMake.

setlocal enabledelayedexpansion

set SOURCES=src\*.cpp
set INCLUDES=/I include
set FLAGS=/std:c++20 /EHsc /W4 /O2
set LINKS=

echo Building OpenRAI PoC...
cl.exe %FLAGS% %INCLUDES% %SOURCES% /Fe:openrai.exe %LINKS% 2>&1

if errorlevel 1 (
    echo Build failed!
    exit /b 1
) else (
    echo Build successful!
    echo Running OpenRAI...
    openrai.exe
)

endlocal
