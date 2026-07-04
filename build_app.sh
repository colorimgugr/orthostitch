#!/usr/bin/env bash
# Build the HSI Stitcher stand-alone app for macOS or Linux.
#
# Usage:
#   bash build_app.sh           # normal build
#   bash build_app.sh --onefile # single-file variant (slower start-up)
#
# Output:
#   dist/HSI_Stitcher/          portable directory (all platforms)
#   dist/HSI_Stitcher.app       macOS .app bundle  (macOS only)
#
# Requirements: Python 3.9+, the project's requirements.txt already installed.
#   pip install -r requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ONEFILE=0
for arg in "$@"; do
    [[ "$arg" == "--onefile" ]] && ONEFILE=1
done

# ── install / upgrade PyInstaller ────────────────────────────────────────────
echo "==> Installing / upgrading PyInstaller …"
pip install --quiet --upgrade pyinstaller

# ── clean previous build artefacts ───────────────────────────────────────────
echo "==> Cleaning previous build …"
rm -rf build/ dist/ __pycache__/

# ── build ────────────────────────────────────────────────────────────────────
if [[ $ONEFILE -eq 1 ]]; then
    echo "==> Building single-file variant …"
    pyinstaller --clean --noconfirm \
        --onefile \
        --windowed \
        --name HSI_Stitcher \
        --hidden-import stitcher \
        --hidden-import modality \
        --hidden-import envi_io \
        --hidden-import ortho \
        --collect-all rawpy \
        --collect-all imagecodecs \
        --collect-submodules scipy \
        --exclude-module tkinter \
        gui.py
    echo ""
    echo "==> Built: dist/HSI_Stitcher"
else
    echo "==> Building from hsi_stitcher.spec …"
    pyinstaller --clean --noconfirm hsi_stitcher.spec
    echo ""
    echo "==> Built: dist/HSI_Stitcher/"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    macOS bundle: dist/HSI_Stitcher.app"
        echo ""
        echo "    To run:  open dist/HSI_Stitcher.app"
        echo ""
        echo "    NOTE: macOS Gatekeeper will block unsigned apps. Right-click →"
        echo "    Open → Open to bypass on first launch, or sign with:"
        echo "      codesign --deep --force --sign - dist/HSI_Stitcher.app"
    else
        echo ""
        echo "    To run:  ./dist/HSI_Stitcher/HSI_Stitcher"
    fi
fi
