@echo off
REM Open the most recent queue output folder

echo Looking for latest queue results...
echo.

set OUTPUT_DIR=cache\outputs

if not exist "%OUTPUT_DIR%" (
    echo No output folder found!
    echo Process a queue first, then run this script.
    pause
    exit /b 1
)

REM Find the most recent queue folder
for /f "delims=" %%i in ('dir "%OUTPUT_DIR%\queue_*" /b /ad /o-d 2^>nul') do (
    set LATEST=%%i
    goto :found
)

:notfound
echo No queue folders found in %OUTPUT_DIR%
echo Process a queue first, then run this script.
pause
exit /b 1

:found
set FULL_PATH=%OUTPUT_DIR%\%LATEST%
echo Found: %LATEST%
echo.
echo Opening: %FULL_PATH%
echo.

explorer "%FULL_PATH%"

REM Also show summary
if exist "%FULL_PATH%\queue_summary.json" (
    echo Queue Summary:
    type "%FULL_PATH%\queue_summary.json"
    echo.
)

echo.
echo Folder opened in Explorer!
pause

