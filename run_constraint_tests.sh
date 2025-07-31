#!/bin/bash

# Script to build and run the C++ constraint tests for OPTPP

set -e  # Exit on any error

echo "=============================================="
echo "OPTPP C++ Constraint Tests Runner"
echo "=============================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if build directory exists
BUILD_DIR="dakota-packages/OPTPP/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found: $BUILD_DIR"
    echo "Please run the main build first to set up OPTPP"
    exit 1
fi

echo "Building pyopttpp module with constraint tests..."
cd "$BUILD_DIR"

# Build the pybind11 module
if make pyopttpp; then
    echo "✓ Successfully built pyopttpp module"
else
    echo "✗ Failed to build pyopttpp module"
    exit 1
fi

cd "$SCRIPT_DIR"

echo ""
echo "Running C++ constraint tests..."
echo ""

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Running tests via pytest..."
    if pytest tests/test_cpp_constraint_tests.py -v; then
        echo ""
        echo "✓ All available constraint tests completed"
    else
        echo ""
        echo "⚠ Some constraint tests failed or had issues"
        echo "Check the output above for details"
    fi
else
    echo "pytest not found, running tests directly with Python..."
    if python tests/test_cpp_constraint_tests.py; then
        echo ""
        echo "✓ Constraint tests completed"
    else
        echo ""
        echo "⚠ Constraint tests had issues"
        echo "Check the output above for details"
    fi
fi

echo ""
echo "=============================================="
echo "For more details, see: tests/README_CONSTRAINT_TESTS.md"
echo "=============================================="