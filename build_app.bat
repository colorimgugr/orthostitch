@echo off
REM Build the HSI Stitcher stand-alone app for Windows.
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
        --name HSI_Stitcher ^
        --hidden-import stitcher ^
        --hidden-import modality ^
        --hidden-import envi_io ^
        --hidden-import ortho ^
        --collect-all rawpy ^
        --collect-all imagecodecs ^
        --collect-submodules scipy ^
        --exclude-module tkinter ^
        gui.py
    if errorlevel 1 goto :err
    echo.
    echo =^=^> Built: dist\HSI_Stitcher.exe
) else (
    echo ==> Building from hsi_stitcher.spec …
    pyinstaller --clean --noconfirm hsi_stitcher.spec
    if errorlevel 1 goto :err
    echo.
    echo =^=^> Built: dist\HSI_Stitcher\
    echo     To run:  dist\HSI_Stitcher\HSI_Stitcher.exe
)
goto :eof

:err
echo.
echo [ERROR] Build failed. Check output above.
exit /b 1
