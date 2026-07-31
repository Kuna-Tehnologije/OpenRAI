# OpenRAI Build Guide

## Quick Start

### Linux/macOS (Recommended)

```bash
# Make script executable
chmod +x build.sh

# Build and run
./build.sh
```

### Windows (MSVC/Visual Studio)

```batch
# Using build.bat
build.bat

# Or manual compile
cl.exe /std:c++17 /EHsc /I include src\main.cpp src\core.cpp ^
       src\thread_pool.cpp src\memory.cpp src\gpu_queue.cpp ^
       src\reasoning.cpp /Fe:openrai.exe
```

### Using CMake (All Platforms)

```bash
# Configure
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .

# Run
./openrai          # Linux/macOS
# or
openrai.exe        # Windows
```

---

## System Requirements

### Minimum
- **OS**: Linux, macOS, or Windows
- **Compiler**: GCC 7+, Clang 5+, MSVC 19.x
- **C++ Standard**: C++17 or later
- **RAM**: 256 MB
- **Disk**: 50 MB

### Recommended
- **OS**: Ubuntu 20.04+ / macOS 11+ / Windows 10+
- **Compiler**: GCC 11+ or Clang 14+
- **CPU**: Multi-core (4+)
- **RAM**: 2 GB+

---

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install build tools
sudo apt update
sudo apt install -y build-essential cmake git

# Clone repo (if not done)
git clone https://github.com/Kuna-Tehnologije/OpenRAI.git
cd OpenRAI

# Build with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
./openrai

# Or use the shell script
../build.sh
```

### macOS

```bash
# Install Xcode command line tools (if needed)
xcode-select --install

# Using Homebrew (optional)
brew install cmake

# Build
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
./openrai
```

### Windows (Visual Studio 2022)

```batch
REM Open Visual Studio Developer Command Prompt

REM Build with CMake
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
Release\openrai.exe

REM Or direct compile
cd ..
cl.exe /std:c++17 /EHsc /I include src\main.cpp src\core.cpp ^
       src\thread_pool.cpp src\memory.cpp src\gpu_queue.cpp ^
       src\reasoning.cpp /Fe:openrai.exe
openrai.exe
```

### Windows (MinGW/MSYS2)

```bash
# Install MinGW-w64 via MSYS2
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake

# Build
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
openrai.exe
```

---

## Compiler Detection

The build system automatically detects:
- **GCC** (`g++`, `g++-11`, etc.)
- **Clang** (`clang++`)
- **MSVC** (`cl.exe`)
- **MinGW** (`x86_64-w64-mingw32-g++`)

---

## Compilation Options

### Release Build (Optimized)
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### Debug Build (With symbols)
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### CUDA Support (Future)
```bash
cmake -DENABLE_CUDA=ON ..
```

---

## Troubleshooting

### "C++17 not supported"
Update your compiler:
- GCC: `sudo apt install gcc-11 g++-11`
- Clang: `sudo apt install clang-14`

### "pthread not found" (Linux)
```bash
sudo apt install libpthread-stubs0-dev
```

### "CMake not found"
- Linux: `sudo apt install cmake`
- macOS: `brew install cmake`
- Windows: Download from cmake.org

### "Build fails with undefined reference"
Ensure all source files in `src/` are included in the build.

---

## Verification

After successful build, run:

```bash
./openrai
```

Expected output:
```
????????????????????????????????????????
?         OpenRAI - PoC Engine         ?
?   raw / real / reimagined AI         ?
????????????????????????????????????????

[ThreadPool] Started with 10 worker threads
[Memory] Initialized
...
```

---

## Clean Build

```bash
# Using CMake
rm -rf build
mkdir build
cd build
cmake ..
make

# Using shell script
./build.sh clean    # If implemented
```

---

## Performance Testing

Monitor during execution:

**Linux**:
```bash
# In another terminal
htop -p $(pgrep openrai)
```

**macOS**:
```bash
Activity Monitor ? Search "openrai"
```

**Windows**:
```batch
tasklist | find "openrai"
wmic process where name="openrai.exe" get priority
```

---

## Next Steps

1. ? Build successfully
2. Run OpenRAI PoC
3. Observe loop output
4. Move to Phase 2: GPU integration
5. Add persistent vector DB
6. Expand reasoning module
