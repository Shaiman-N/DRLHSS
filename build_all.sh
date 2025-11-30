#!/bin/bash

# DRLHSS Complete Build Script
# Builds the entire DRLHSS system with all components

set -e  # Exit on error

echo "========================================="
echo "  DRLHSS Complete Build Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
else
    echo -e "${RED}Unsupported platform: $OSTYPE${NC}"
    exit 1
fi

echo -e "${GREEN}Detected platform: $PLATFORM${NC}"
echo ""

# Check for required tools
echo "Checking for required tools..."

command -v cmake >/dev/null 2>&1 || { echo -e "${RED}CMake is required but not installed.${NC}" >&2; exit 1; }
command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo -e "${RED}C++ compiler is required but not installed.${NC}" >&2; exit 1; }

echo -e "${GREEN}✓ CMake found${NC}"
echo -e "${GREEN}✓ C++ compiler found${NC}"

# Check for dependencies
echo ""
echo "Checking for dependencies..."

if [[ "$PLATFORM" == "Linux" ]]; then
    # Check Linux dependencies
    dpkg -s libsqlite3-dev >/dev/null 2>&1 || echo -e "${YELLOW}⚠ libsqlite3-dev not found. Install with: sudo apt-get install libsqlite3-dev${NC}"
    dpkg -s libssl-dev >/dev/null 2>&1 || echo -e "${YELLOW}⚠ libssl-dev not found. Install with: sudo apt-get install libssl-dev${NC}"
    dpkg -s libseccomp-dev >/dev/null 2>&1 || echo -e "${YELLOW}⚠ libseccomp-dev not found. Install with: sudo apt-get install libseccomp-dev${NC}"
    dpkg -s libpcap-dev >/dev/null 2>&1 || echo -e "${YELLOW}⚠ libpcap-dev not found. Install with: sudo apt-get install libpcap-dev${NC}"
elif [[ "$PLATFORM" == "macOS" ]]; then
    # Check macOS dependencies
    brew list sqlite3 >/dev/null 2>&1 || echo -e "${YELLOW}⚠ sqlite3 not found. Install with: brew install sqlite3${NC}"
    brew list openssl >/dev/null 2>&1 || echo -e "${YELLOW}⚠ openssl not found. Install with: brew install openssl${NC}"
    brew list libpcap >/dev/null 2>&1 || echo -e "${YELLOW}⚠ libpcap not found. Install with: brew install libpcap${NC}"
fi

# Check for ONNX Runtime
if [ ! -d "external/onnxruntime" ]; then
    echo -e "${YELLOW}⚠ ONNX Runtime not found in external/onnxruntime${NC}"
    echo "  Download from: https://github.com/microsoft/onnxruntime/releases"
    echo "  Extract to: external/onnxruntime"
fi

echo ""
read -p "Continue with build? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "Configuring with CMake..."
if [[ "$PLATFORM" == "macOS" ]]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl
else
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configuration successful${NC}"

# Build
echo ""
echo "Building project..."
if [[ "$PLATFORM" == "macOS" ]]; then
    make -j$(sysctl -n hw.ncpu)
else
    make -j$(nproc)
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"

# List built executables
echo ""
echo "========================================="
echo "  Build Complete!"
echo "========================================="
echo ""
echo "Built executables:"
echo ""

if [ -f "DRLHSS_main" ]; then
    echo -e "${GREEN}✓ DRLHSS_main${NC} - Legacy main executable"
fi

if [ -f "drl_integration_example" ]; then
    echo -e "${GREEN}✓ drl_integration_example${NC} - DRL integration example"
fi

if [ -f "integrated_system_example" ]; then
    echo -e "${GREEN}✓ integrated_system_example${NC} - Complete integrated system"
fi

if [[ "$PLATFORM" == "Linux" ]] && [ -f "test_linux_sandbox" ]; then
    echo -e "${GREEN}✓ test_linux_sandbox${NC} - Linux sandbox tests"
fi

if [[ "$PLATFORM" == "macOS" ]] && [ -f "test_macos_sandbox" ]; then
    echo -e "${GREEN}✓ test_macos_sandbox${NC} - macOS sandbox tests"
fi

echo ""
echo "To run the integrated system:"
echo "  cd build"
echo "  ./integrated_system_example"
echo ""

if [[ "$PLATFORM" == "Linux" ]]; then
    echo "Note: Sandbox functionality may require elevated privileges:"
    echo "  sudo ./integrated_system_example"
    echo "  or"
    echo "  sudo setcap cap_sys_admin,cap_net_admin+ep ./integrated_system_example"
    echo ""
fi

echo "For more information, see:"
echo "  - docs/NIDPS_INTEGRATION_GUIDE.md"
echo "  - docs/DEPLOYMENT_GUIDE.md"
echo "  - COMPLETE_INTEGRATION_README.md"
echo ""
echo "========================================="

