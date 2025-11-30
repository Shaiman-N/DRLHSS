#!/bin/bash
# DIREWOLF Build and Package Script
# Builds installers for all platforms

set -e

VERSION="1.0.0"
BUILD_DIR="build"
PACKAGE_DIR="packages"

echo "========================================="
echo "DIREWOLF Build & Package System"
echo "Version: $VERSION"
echo "========================================="

# Detect platform
PLATFORM=$(uname -s)
echo "Platform detected: $PLATFORM"

# Create directories
mkdir -p $BUILD_DIR
mkdir -p $PACKAGE_DIR

# Build application
echo ""
echo "Building DIREWOLF..."
cd $BUILD_DIR
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

echo "Build complete!"

# Platform-specific packaging
case "$PLATFORM" in
    Linux)
        echo ""
        echo "Creating Linux packages..."
        
        # DEB package
        echo "Building DEB package..."
        ./scripts/package_deb.sh $VERSION
        
        # RPM package
        echo "Building RPM package..."
        ./scripts/package_rpm.sh $VERSION
        
        # AppImage
        echo "Building AppImage..."
        ./scripts/package_appimage.sh $VERSION
        ;;
        
    Darwin)
        echo ""
        echo "Creating macOS packages..."
        
        # DMG
        echo "Building DMG..."
        ./scripts/package_dmg.sh $VERSION
        
        # PKG
        echo "Building PKG..."
        ./scripts/package_pkg.sh $VERSION
        ;;
        
    MINGW*|MSYS*|CYGWIN*)
        echo ""
        echo "Creating Windows packages..."
        
        # MSI
        echo "Building MSI..."
        ./scripts/package_msi.bat $VERSION
        ;;
        
    *)
        echo "Unknown platform: $PLATFORM"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Packaging complete!"
echo "Packages available in: $PACKAGE_DIR"
echo "========================================="

# List created packages
ls -lh $PACKAGE_DIR/
