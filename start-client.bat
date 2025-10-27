@echo off
REM DeepSeek OCR Client Launcher for Windows

REM Set console to UTF-8 encoding to support Unicode characters
chcp 65001 >nul 2>nul

REM Set PYTHONIOENCODING to UTF-8 for Python script
set PYTHONIOENCODING=utf-8

REM Try to find a compatible Python version (3.10, 3.11, or 3.12)
set PYTHON_CMD=

REM Try Python 3.10 first (recommended)
py -3.10 --version >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.10
    goto :run_script
)

REM Try Python 3.11
py -3.11 --version >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.11
    goto :run_script
)

REM Try Python 3.12
py -3.12 --version >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.12
    goto :run_script
)

REM No compatible Python found
echo ========================================
echo ERROR: No compatible Python version found
echo ========================================
echo.
echo This application requires Python 3.10, 3.11, or 3.12
echo PyTorch does not support Python 3.13 or 3.14 yet
echo.
echo Please install Python 3.10 (recommended):
echo https://www.python.org/ftp/python/3.10.14/python-3.10.14-amd64.exe
echo.
echo After installing, run this script again.
echo.
pause
exit /b 1

:run_script
%PYTHON_CMD% start.py
pause