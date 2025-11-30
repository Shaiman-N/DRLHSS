#!/bin/bash

echo "========================================"
echo "DIREWOLF GUI Dashboard Build Script"
echo "========================================"
echo ""

# Set Qt path (adjust this to your Qt installation)
export QT_PATH="/usr/lib/qt6"
export PATH="$QT_PATH/bin:$PATH"

# Create build directory
mkdir -p build_gui
cd build_gui

echo "Configuring CMake..."
cmake -G "Unix Makefiles" \
    -DCMAKE_PREFIX_PATH=$QT_PATH \
    -DCMAKE_BUILD_TYPE=Release \
    -S .. -B . \
    -C ../CMakeLists_gui.txt

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

echo ""
echo "Building DIREWOLF GUI..."
cmake --build . --config Release -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "Executable: build_gui/bin/direwolf_gui"
echo "========================================"
echo ""
