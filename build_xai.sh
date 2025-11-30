#!/bin/bash

echo "================================================"
echo "  DIREWOLF XAI - Phase 1 Foundation Build"
echo "================================================"
echo ""

# Create build directory
mkdir -p build_xai
cd build_xai

echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release -S .. -B . -f ../CMakeLists_xai.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    echo ""
    echo "Please ensure:"
    echo "  1. Qt6 is installed"
    echo "  2. CMake 3.20+ is installed"
    echo "  3. C++20 compiler is available"
    echo ""
    exit 1
fi

echo ""
echo "Building project..."
cmake --build . --config Release --parallel

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "  Build Complete!"
echo "================================================"
echo ""
echo "Executable location: build_xai/direwolf_xai"
echo ""
echo "To run the application:"
echo "  cd build_xai"
echo "  ./direwolf_xai"
echo ""
