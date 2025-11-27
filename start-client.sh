#!/bin/bash

# DeepSeek OCR Client Launcher for Linux/macOS

# Try to find a compatible Python version (3.10, 3.11, or 3.12)
PYTHON_CMD=""

for version in 3.10 3.11 3.12; do
    if command -v "python$version" &> /dev/null; then
        PYTHON_CMD="python$version"
        echo "Found Python $version"
        break
    fi
done

# Fallback: check if default python3 is a compatible version
if [ -z "$PYTHON_CMD" ] && command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$PY_VERSION" in
        3.10|3.11|3.12)
            PYTHON_CMD="python3"
            echo "Found Python $PY_VERSION"
            ;;
    esac
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "========================================"
    echo "ERROR: No compatible Python version found"
    echo "========================================"
    echo ""
    echo "This application requires Python 3.10, 3.11, or 3.12"
    echo "PyTorch does not support Python 3.13 or 3.14 yet"
    echo ""
    echo "Please install Python 3.10 (recommended):"
    echo "  Ubuntu/Debian: sudo apt install python3.10"
    echo "  macOS: brew install python@3.10"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

$PYTHON_CMD start.py