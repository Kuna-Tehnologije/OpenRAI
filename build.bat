@echo off
REM Build script za OpenRAI PoC
REM Trebam cl.exe compiler (Visual Studio)

setlocal enabledelayedexpansion

set SOURCES=src\main.cpp src\core.cpp src\thread_pool.cpp src\memory.cpp src\gpu_queue.cpp src\reasoning.cpp
set INCLUDES=/I include
set FLAGS=/std:c++17 /EHsc /W4 /O2
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
