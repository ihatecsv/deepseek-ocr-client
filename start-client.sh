#!/bin/bash

# DeepSeek OCR Client Launcher for Linux/macOS

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed"
    echo "Please install Python 3.12+ from https://www.python.org/"
    exit 1
fi

python3 start.py