#!/bin/bash
# Build script for OpenRAI - Linux/macOS
#
#   ./build.sh            CMake build (auto-detects CUDA -> streams/contexts)
#   ./build.sh --direct   plain g++/clang++ CPU-only build without CMake
#   ./build.sh --test     build and run the node scheduler tests

set -e

echo "==================================="
echo "  OpenRAI Build Script (Unix-like)"
echo "==================================="

MODE="cmake"
for a in "$@"; do
    case "$a" in
        --direct) MODE="direct" ;;
        --test)   MODE="test" ;;
    esac
done

if [ "$MODE" = "direct" ]; then
    if command -v clang++ &> /dev/null; then COMPILER="clang++"; else COMPILER="g++"; fi
    echo "Direct CPU-only build with $COMPILER"
    $COMPILER -std=c++20 -Wall -Wextra -O2 -I./include $(ls src/*.cpp) -pthread -o openrai
    echo "Build successful -> ./openrai"
    exit 0
fi

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENRAI_BUILD_TESTS=ON
cmake --build . --config Release -j

echo ""
echo "Build successful -> build/openrai"

if [ "$MODE" = "test" ]; then
    echo "Running node scheduler tests..."
    ./openrai_tests
fi
