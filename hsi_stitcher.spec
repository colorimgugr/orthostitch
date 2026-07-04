# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for HSI Stitcher — cross-platform, single spec.
#
# Supports: macOS (.app bundle), Windows (.exe), Linux (ELF binary)
#
# Build:
#   Mac/Linux:  bash build_app.sh
#   Windows:    build_app.bat
#   or manually: pyinstaller --clean hsi_stitcher.spec
#
# Output:
#   dist/HSI_Stitcher/          portable folder (all platforms)
#   dist/HSI_Stitcher.app       macOS .app bundle (macOS only)
#
# ─── Icon ────────────────────────────────────────────────────────────────────
# Place your icon files at:
#   docs/assets/icon.icns   (macOS)
#   docs/assets/icon.ico    (Windows)
# then set the icon= arguments below.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path

from PyInstaller.building.api import EXE, COLLECT, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_all, collect_submodules

IS_MAC = sys.platform == 'darwin'
IS_WIN = sys.platform == 'win32'

ICON_MAC = 'docs/assets/icon.icns' if Path('docs/assets/icon.icns').exists() else None
ICON_WIN = 'docs/assets/icon.ico'  if Path('docs/assets/icon.ico').exists()  else None
ICON = ICON_MAC if IS_MAC else (ICON_WIN if IS_WIN else None)

# ── collect_all for packages that ship native libraries PyInstaller may miss ─
datas, binaries, hidden = [], [], []

for pkg in ('rawpy', 'imagecodecs'):
    d, b, h = collect_all(pkg)
    datas    += d
    binaries += b
    hidden   += h
# cv2 is intentionally NOT in collect_all: PyInstaller has a built-in hook for
# opencv-python and collect_all causes a double-import recursion crash on macOS.

# scipy hidden submodules (numeric routines loaded at runtime via Cython)
hidden += collect_submodules('scipy')

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    ['gui.py'],
    # pathex makes the sibling modules (stitcher, modality, envi_io, ortho)
    # importable — they live in the same directory as gui.py
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + [
        'stitcher',
        'modality',
        'envi_io',
        'ortho',
        # rawpy optional import guard: if rawpy not installed the GUI still works
        'rawpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_cv2.py'],
    excludes=['tkinter', 'matplotlib', 'pandas', 'IPython', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HSI_Stitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # console=False hides the black terminal window on Windows; progress/log
    # output goes to the Qt GUI instead.  Set to True only when debugging a
    # crash inside the bundle.
    console=False,
    icon=ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HSI_Stitcher',
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
if IS_MAC:
    from PyInstaller.building.osx import BUNDLE
    app = BUNDLE(
        coll,
        name='HSI_Stitcher.app',
        icon=ICON_MAC,
        bundle_identifier='es.ugr.colorimaginglab.hsistitcher',
        info_plist={
            # Enables native Retina rendering
            'NSHighResolutionCapable': True,
            # Allow the app to open files via Finder drag-and-drop
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Image Files',
                    'CFBundleTypeExtensions': ['tif', 'tiff', 'png', 'jpg', 'jpeg'],
                    'CFBundleTypeRole': 'Viewer',
                }
            ],
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            # Camera / microphone usage strings are not needed; remove the
            # entitlement if App Store submission is ever pursued.
        },
    )
