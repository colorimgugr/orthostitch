#!/usr/bin/env bash
# Build the OrthoStitch stand-alone app for macOS or Linux.
#
# Usage:
#   bash build_app.sh           # normal build (recommended)
#   bash build_app.sh --onefile # single-file variant (slower start-up)
#
# Output:
#   dist/OrthoStitch/          portable directory (all platforms)
#   dist/OrthoStitch.app       macOS .app bundle  (macOS only)
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
        --name OrthoStitch \
        --paths src \
        --hidden-import stitcher \
        --hidden-import modality \
        --hidden-import envi_io \
        --hidden-import ortho \
        --collect-all rawpy \
        --collect-all imagecodecs \
        --collect-submodules scipy \
        --exclude-module tkinter \
        --runtime-hook src/rthook_cv2.py \
        src/gui.py
    echo ""
    echo "==> Built: dist/OrthoStitch"
else
    echo "==> Building from orthostitch.spec …"
    pyinstaller --clean --noconfirm orthostitch.spec
    echo ""
    echo "==> Built: dist/OrthoStitch/"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    macOS bundle: dist/OrthoStitch.app"
        echo ""
        echo "    To run:  open dist/OrthoStitch.app"
        echo ""
        echo "    NOTE: macOS Gatekeeper will block unsigned apps. Right-click →"
        echo "    Open → Open to bypass on first launch, or sign with:"
        echo "      codesign --deep --force --sign - dist/OrthoStitch.app"
    else
        echo ""
        echo "    To run:  ./dist/OrthoStitch/OrthoStitch"
    fi
fi
