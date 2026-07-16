@echo off
REM Build the OrthoStitch stand-alone app for Windows.
REM
REM Usage:
REM   build_app.bat            normal build (dist\ folder)
REM   build_app.bat --onefile  single-file .exe variant (slower start-up)
REM
REM Requirements: Python 3.9+ in PATH, requirements.txt already installed.
REM   pip install -r requirements.txt

setlocal
cd /d "%~dp0"

set ONEFILE=0
for %%A in (%*) do if "%%A"=="--onefile" set ONEFILE=1

REM ── install / upgrade PyInstaller ─────────────────────────────────────────
echo ==> Installing / upgrading PyInstaller …
pip install --quiet --upgrade pyinstaller
if errorlevel 1 goto :err

REM ── clean previous build artefacts ────────────────────────────────────────
echo ==> Cleaning previous build …
if exist build\  rmdir /s /q build
if exist dist\   rmdir /s /q dist
if exist __pycache__\ rmdir /s /q __pycache__

REM ── build ─────────────────────────────────────────────────────────────────
if "%ONEFILE%"=="1" (
    echo ==> Building single-file variant …
    pyinstaller --clean --noconfirm ^
        --onefile ^
        --windowed ^
        --name OrthoStitch ^
        --hidden-import stitcher ^
        --hidden-import modality ^
        --hidden-import envi_io ^
        --hidden-import ortho ^
        --collect-all rawpy ^
        --collect-all imagecodecs ^
        --collect-submodules scipy ^
        --exclude-module tkinter ^
        --runtime-hook rthook_cv2.py ^
        gui.py
    if errorlevel 1 goto :err
    echo.
    echo =^=^> Built: dist\OrthoStitch.exe
) else (
    echo ==> Building from orthostitch.spec …
    pyinstaller --clean --noconfirm orthostitch.spec
    if errorlevel 1 goto :err
    echo.
    echo =^=^> Built: dist\OrthoStitch\
    echo     To run:  dist\OrthoStitch\OrthoStitch.exe
)
goto :eof

:err
echo.
echo [ERROR] Build failed. Check output above.
exit /b 1
