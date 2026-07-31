#!/bin/bash
# Build script za OpenRAI PoC - Linux/macOS

set -e

echo "==================================="
echo "  OpenRAI Build Script (Unix-like)"
echo "==================================="

# Detect compiler
if command -v clang++ &> /dev/null; then
    COMPILER="clang++"
    echo "Detected: Clang++"
elif command -v g++ &> /dev/null; then
    COMPILER="g++"
    echo "Detected: GCC"
else
    echo "Error: No C++ compiler found"
    exit 1
fi

# Compiler flags
CXXFLAGS="-std=c++17 -Wall -Wextra -O2"
INCLUDES="-I./include"
LIBS="-pthread"
OUTPUT="openrai"

# Source files
SOURCES="src/main.cpp src/core.cpp src/thread_pool.cpp src/memory.cpp src/gpu_queue.cpp src/reasoning.cpp"

# Build
echo ""
echo "Compiling with: $COMPILER"
echo "Output: $OUTPUT"
echo ""

$COMPILER $CXXFLAGS $INCLUDES $SOURCES $LIBS -o $OUTPUT

if [ $? -eq 0 ]; then
    echo ""
    echo "? Build successful!"
    echo ""
    echo "Running OpenRAI..."
    echo "=================================="
    ./$OUTPUT
else
    echo ""
    echo "? Build failed!"
    exit 1
fi
