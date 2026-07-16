# OrthoStitch

> Lossless ortho-anchored panorama stitching for hyperspectral and photographic imaging tiles — with a cross-platform desktop GUI.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52)
![License](https://img.shields.io/badge/license-MIT-green)

<!-- Replace with actual screenshots once captured (see docs/assets/.gitkeep for guidance) -->
<!-- ![App screenshot](docs/assets/screenshot_main.png) -->

---

## Overview

OrthoStitch takes a grid (or irregular layout) of imaging tiles — either hyperspectral cubes (ENVI BIL) or photographic captures (PNG / TIFF / JPEG / Nikon NEF RAW) — and assembles them into a single, spatially-consistent panorama using a reference orthoimage of the same subject.

The pipeline is fully **lossless**: every pixel in the output is a bit-exact copy of exactly one source tile pixel. Geometry is estimated entirely on derived grayscale images; spectral data is moved only by integer nearest-neighbor indexing — no interpolation, no blending, no photometric adjustment.

Designed for **multi-modal cultural heritage imaging**: VNIR hyperspectral, visible photography, IR reflectography, and UV fluorescence of paintings and manuscripts.

---

## Features

- **Lossless output** — nearest-neighbor copy only; verified against 1000 random pixels after assembly
- **Any modality** — ENVI BIL hyperspectral, PNG/TIFF/JPEG photography, Nikon NEF RAW
- **Ortho-anchored V3 pipeline** — tiles are jointly registered *and* anchored to the orthoimage in a single bundle adjustment; output orientation always matches the supplied ortho
- **Shared radial distortion** — estimates a single k₁/k₂ lens model from tile overlaps (no calibration target needed)
- **Dark painting support** — tunable SIFT contrast threshold keeps low-contrast keypoints for dim/low-texture subjects
- **High-res photographic tiles** — canvas resolution capped to prevent gigapixel RAM exhaustion while preserving losslessness
- **Interactive GUI** — pan, zoom, rotate result viewers; live progress log; Advanced options dialog for every registration parameter
- **CLI mode** — scriptable from any shell or pipeline

---

## Supported input modalities

| Tile format | Extension(s) | Output mosaic | Notes |
|---|---|---|---|
| ENVI BIL hyperspectral | `*.bil` + `*.bil.hdr` | `*_pano.bil` + `*.hdr` | Wavelength axis preserved |
| Photographic (8/16-bit) | `*.png` `*.tif` `*.tiff` `*.jpg` | `*_pano.tif` | 1- or 3-channel |
| Nikon RAW | `*.nef` | `*_pano.tif` | Requires `rawpy`; demosaiced to RGB |

One modality per folder. Mixed folders are rejected with a clear error.

---

## Installation

### Option A — from source (recommended for development)

```bash
git clone https://github.com/ossama971/orthostitch.git
cd orthostitch
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
|---|---|
| `numpy` | array math |
| `opencv-python` | SIFT, RANSAC, image I/O |
| `scipy` | least-squares bundle adjustment |
| `tifffile` | TIFF read/write |
| `imagecodecs` | TIFF codec support |
| `PyQt6` | GUI (optional — CLI works without it) |
| `rawpy` | Nikon NEF decoding (optional) |

### Option B — pre-built executable (no Python needed)

Download the latest release for your platform from the
[Releases page](../../releases) and unzip:

| Platform | File | How to run |
|---|---|---|
| macOS | `OrthoStitch-mac.zip` → `OrthoStitch.app` | Double-click, or `open OrthoStitch.app` |
| Windows | `OrthoStitch-win.zip` → `OrthoStitch/` | `OrthoStitch\OrthoStitch.exe` |
| Linux | `OrthoStitch-linux.tar.gz` → `OrthoStitch/` | `./OrthoStitch/OrthoStitch` |

> **macOS Gatekeeper note**: The app is unsigned. On first launch, right-click →
> **Open** → **Open** to bypass the "unidentified developer" warning.
> Subsequent launches work normally.

---

## Quick start

### GUI

```bash
python hsi_stitcher/gui.py
# or, from inside the hsi_stitcher/ directory:
python gui.py
```

<!-- ![Inputs card](docs/assets/screenshot_inputs.png) -->

**Step by step:**

1. **Tiles folder** — click *Browse…* or drag-and-drop a folder. The detected modality
   and tile count appear below the field (e.g. `hyperspectral · 12 tiles · ENVI BIL`).
2. **Ortho image** — the reference image the panorama will be aligned and oriented to
   (any TIFF/PNG/JPEG).
3. **Artwork name** — used as the output filename prefix; auto-filled from the folder
   name.
4. **Output folder** — where results are written; auto-filled to `<tiles_folder>/stitched`.
5. Click **Stitch**. Progress appears in the right panel (6 stages).

<!-- ![App after stitching](docs/assets/screenshot_main.png) -->

Once done, the two result viewers show the **pseudo-RGB preview** and the **seam map**.
Each viewer supports:

| Control | Action |
|---|---|
| Mouse wheel | Zoom in / out |
| Click + drag | Pan |
| `+` / `-` buttons | Zoom step |
| **Fit** button | Fit image to window |
| **1:1** button | Actual pixel size |
| **↶ / ↷** buttons | Rotate 90° left / right |

<!-- ![Pseudo-RGB viewer](docs/assets/screenshot_result_rgb.png) -->

Click **Open output folder** to reveal all output files in Finder / Explorer.

---

### CLI

```bash
python hsi_stitcher/stitcher.py \
    --tiles /path/to/tiles_folder \
    --ortho /path/to/ortho.tif \
    --out   /path/to/output \
    --name  MyArtwork
```

<!-- ![CLI output](docs/assets/screenshot_cli.png) -->

All six pipeline stages print progress to stdout:

```
[1/6] registering tiles …      done  (4.3 s)
[2/6] matching pairs …         done  (12.1 s)
[3/6] bundle adjustment …      done  (1.8 s)
[4/6] anchoring to ortho …     done  (3.2 s)
[5/6] assembling mosaic …      done  (28.4 s)
[6/6] writing outputs …        done  (2.1 s)
```

---

## Pipeline architecture

```
  Input: tile folder  +  ortho image
        │                    │
        ▼                    ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     V3 ortho-anchored pipeline                           │
  ├──────────────┬────────────────┬───────────────┬────────────┬────────────┤
  │  Stage 1     │   Stage 2      │   Stage 3     │  Stage 4   │  Stage 5   │
  │  Tile reg.   │  Pairwise      │  Bundle       │  Ortho     │  Assembly  │
  │  images      │  SIFT match    │  adjustment   │  anchoring │            │
  │              │                │               │            │            │
  │  · mean of   │  · all-pairs   │  · spanning   │  · FLANN   │  · Voronoi │
  │    500–900nm │    SIFT feat.  │    tree seed  │    index   │    seams   │
  │    (HSI) or  │  · Lowe ratio  │  · joint LSQ  │    on      │  · band-   │
  │    grayscale │    test        │    of all     │    ortho   │    by-band │
  │    (photo)   │  · RANSAC      │    homograph. │  · ortho-  │    nearest-│
  │  · CLAHE +   │    inlier      │    + shared   │    anchor  │    neighbor│
  │    downscale │    count       │    k₁/k₂      │    weight  │    copy    │
  │    to 1600px │                │               │            │            │
  └──────────────┴────────────────┴───────────────┴────────────┴────────────┘
        │
        ▼
  Stage 6 — Orientation resolved against ortho → outputs written
```

**Why ortho-anchored?**  
Without an absolute reference, a spanning-tree registration drifts over large tile
grids — small per-tile errors accumulate. Anchoring every tile to a known-good ortho
image in the same bundle adjustment bounds the global drift to the ortho's accuracy
(typically < 1 px for a well-captured ortho).

**Losslessness contract**  
After assembly, the pipeline draws 1000 random output pixels, maps each back to its
source tile through the integer seam assignment, and compares the raw channel values.
Any discrepancy aborts with an error. This check verifies end-to-end that no
interpolation crept in.

---

## Advanced options

Click **Advanced…** in the GUI (or edit the module constants in `stitcher.py`) to tune
the registration for your data:

<!-- ![Advanced options dialog](docs/assets/screenshot_advanced.png) -->

| Parameter | Default | Range | When to change |
|---|---|---|---|
| **SIFT contrast threshold** | 0.01 | 0.001 – 0.200 | *Lower* for dark / low-texture paintings to keep more keypoints. OpenCV default is 0.04 — lower it to 0.01 for historical paintings, further for very dark subjects. |
| **SIFT edge threshold** | 20 | 1 – 100 | Higher keeps more edge-like keypoints. Rarely needs changing. |
| **Lowe ratio test** | 0.75 | 0.50 – 0.95 | *Higher* if too few descriptor matches survive; *lower* to suppress ambiguous matches. |
| **Min pair inliers** | 30 | 4 – 1000 | *Lower* if tiles have very small physical overlap. |
| **Min ortho-anchor inliers** | 20 | 4 – 1000 | *Lower* if the ortho is visually dissimilar to the tile modality (e.g. ortho = photo, tiles = UV fluorescence). |
| **RANSAC reproj. threshold** | 3.0 px | 0.5 – 20 | *Higher* for a low-resolution ortho where feature positions are coarser. |
| **Ortho anchor weight** | 0.5 | 0 – 5 | *Higher* forces the panorama to closely follow the ortho at the cost of inter-tile fit; 0 disables anchoring (reverts to free bundle adjustment). |
| **Max output long side** | 12 000 px | 1 000 – 60 000 | **Essential for high-res photographic tiles**: 45 MP NEF tiles at native resolution produce a ~600 MP canvas. Capping to 12 000 px keeps RAM feasible. Output is still bit-exact nearest-neighbor — the grid is simply subsampled. |

The *Reset to defaults* button restores all values; changes only apply to the next run.

---

## Output files

| File | Description |
|---|---|
| `<name>_pano.bil` + `.bil.hdr` | Stitched hyperspectral cube (wavelengths and metadata from source tiles) |
| `<name>_pano.tif` | Stitched photographic panorama (1- or 3-channel TIFF) |
| `<name>_pano_rgb.png` | Pseudo-RGB preview — approx. 640 / 550 / 460 nm (HSI) or captured color (photo) |
| `seam_map.png` | Per-pixel tile ownership map (Voronoi seams, one color per tile) |
| `<name>_rgb_vs_seam.png` | Side-by-side: pseudo-RGB preview and seam map |
| `ortho_overlay.png` | 50 % alpha blend of the panorama warped onto the ortho (registration quality check) |
| `placement.json` | Full registration record: homographies, k₁/k₂, per-pair RMS, anchor inlier counts, per-stage timing, pano-to-ortho homography |

`placement.json` is the primary diagnostic: the `"rms_px"` value per tile pair and
the `"anchor_inliers"` count per tile tell you whether registration was reliable.

---

## Troubleshooting

### "match graph is disconnected — N groups found"

The tile graph split into N disconnected components, meaning some tiles have no accepted
pairwise match. Common causes and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| Dark, low-contrast painting | Default SIFT threshold too strict | Lower **SIFT contrast threshold** to 0.005 – 0.001 |
| Very small tile overlap | Too few features in the overlap zone | Lower **Min pair inliers** (e.g. to 10) |
| Tiles in wrong folder | Non-tile files mixed in | Ensure the folder contains only one modality; exclude `Calibration.*` files |
| Tiles rotated 180° | Descriptor orientation mismatch | Supply an ortho in the correct orientation; the pipeline aligns to it |

The error message lists which tiles belong to each disconnected group, helping you
identify which specific pairs failed.

### High RAM usage or program hangs on large photographic tiles

Photographic tiles (especially Nikon NEF, ~45 MP) at native resolution produce a canvas
of 500 MP or more, which can exceed 64 GB of RAM during assembly.

**Fix**: set **Max output long side** (Advanced dialog) to 12 000 – 24 000 px. At 12 000
px the output is still ~70 MP — sufficient for detailed inspection — and RAM stays under
~4 GB. The output remains bit-exact nearest-neighbor; only the grid spacing changes.

### NEF files fail with "rawpy not found"

`rawpy` is not installed. Install it:

```bash
pip install rawpy
```

If the error is "libraw shared library not found" inside a built executable, `libraw` was
not bundled. Rebuild with the spec file (which includes `--collect-all rawpy`), or add
the library path explicitly:

```bash
pip install rawpy --force-reinstall   # fetches pre-built wheel with libraw
bash build_app.sh
```

### GUI shows blank viewers after stitching

PyQt6 has a default image allocation limit of 256 MB. The pseudo-RGB preview for a
large mosaic can exceed this. The GUI already disables this limit
(`QImageReader.setAllocationLimit(0)`) and downscales previews to ≤ 6000 px for display.
If the viewer still stays blank, check the log panel for a decoding error, and verify
that the output PNG exists in the output folder.

### "RuntimeError: only one modality is allowed per folder"

The tiles folder contains files of more than one modality (e.g. `.bil` and `.tif`).
Keep each modality in its own subdirectory.

---

## Building the executable yourself

PyInstaller cannot cross-compile: you must build on the target OS. On each machine:

```bash
# Install dependencies first
pip install -r requirements.txt

# Build (from the hsi_stitcher/ directory)
bash build_app.sh           # macOS / Linux
build_app.bat               # Windows

# Optional: single-file variant (slower start, easier to share)
bash build_app.sh --onefile
```

Output goes to `dist/OrthoStitch/` (folder) and `dist/OrthoStitch.app` (macOS bundle).

### GitHub Actions automated builds

Add `.github/workflows/build.yml` to build for all three platforms on each tag:

```yaml
name: Build executables

on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        os: [macos-latest, windows-latest, ubuntu-latest]
    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r hsi_stitcher/requirements.txt
          pip install pyinstaller

      - name: Build (macOS / Linux)
        if: runner.os != 'Windows'
        run: bash hsi_stitcher/build_app.sh
        working-directory: hsi_stitcher

      - name: Build (Windows)
        if: runner.os == 'Windows'
        run: .\build_app.bat
        working-directory: hsi_stitcher

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: OrthoStitch-${{ matrix.os }}
          path: hsi_stitcher/dist/
```

### macOS code signing (optional)

To distribute without the Gatekeeper warning:

```bash
# Self-sign (allows local distribution)
codesign --deep --force --sign - dist/OrthoStitch.app

# Developer ID signing (requires Apple Developer account)
codesign --deep --force \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    dist/OrthoStitch.app
xcrun notarytool submit dist/OrthoStitch.zip \
    --apple-id your@email.com --team-id TEAMID --wait
```

---

## Running the test suite

```bash
pip install pytest
python -m pytest hsi_stitcher/tests/ -v
```

Tests are synthetic (no real imaging data required): four overlapping PNG crops of a
random scene serve as tiles, with a downscaled version as the ortho. The suite covers
modality detection, loader correctness, feature rescaling, dtype preservation, and
end-to-end integration (stitches four tiles, checks `placement.json`, runs lossless
check).

```
tests/conftest.py          synthetic_modality fixture (4 PNG tiles + ortho)
tests/test_modality.py     loader: RGB layout, grayscale, reg scale cap, errors
tests/test_engine.py       detect_features rescaling, assemble dtype, pseudo_rgb uint8
tests/test_integration.py  end-to-end: stitch → lossless check → placement.json keys
```

---

## Project structure

```
hsi_stitcher/
├── gui.py              PyQt6 desktop application (entry point for GUI)
├── stitcher.py         Core pipeline: feature detection, matching, bundle adjustment,
│                       assembly, lossless verification
├── modality.py         Format-aware tile loaders: EnviLoader (BIL) + ImageLoader
│                       (PNG/TIFF/JPEG/NEF); auto-detection by extension
├── envi_io.py          Lightweight ENVI BIL reader/writer (header parsing, memmap)
├── ortho.py            Ortho reference loader (caches decoded grayscale at target px)
├── requirements.txt    Python dependencies
├── orthostitch.spec   PyInstaller build spec (all platforms)
├── build_app.sh        macOS / Linux build script
├── build_app.bat       Windows build script
├── tests/
│   ├── conftest.py     Pytest fixtures
│   ├── test_modality.py
│   ├── test_engine.py
│   └── test_integration.py
└── docs/
    └── assets/         Screenshot files (add yours here; see .gitkeep)
```

### Key data flow

```
tiles_dir/         →  modality.make_loader()  →  loader (EnviLoader | ImageLoader)
                                                        │
                        loader.reg(i)  →  grayscale proxy (CLAHE, ≤1600 px)
                        loader.cube(i) →  full-res tile data (lazy memmap / LRU cache)
                        loader.write() →  stitched output file
                                                        │
                   stitcher.stitch()   →  calls loader at each stage
                                       →  placement.json + preview PNGs
```

---

## Algorithm details

### Radial distortion model

A single (k₁, k₂) Brown–Conrady model is shared across all tiles (they come from the
same camera and lens). The distortion center is the image center; the normalization
radius is the half-width. During bundle adjustment both the per-tile homographies *and*
the two distortion coefficients are optimized jointly.

Typical values for heritage imaging setups:

```
k₁ ≈ +0.02  (barrel — most common)
k₂ ≈ −0.003
```

These are reported in `placement.json` as `"k1"` and `"k2"`.

### Seam computation

Voronoi tesselation in image space: each output pixel is assigned to the source tile
whose registration-proxy center is closest (Euclidean distance in the output canvas).
This is equivalent to a nearest-center hard partition — no feathering, no multi-band
blending — which is what keeps the output lossless.

### Panorama-to-ortho homography

After the panorama is assembled, it is registered to the ortho (SIFT + RANSAC at the
ortho's long side ≤ 2000 px). The resulting 3×3 homography is stored in
`placement.json` as `"pano_to_ortho_H"` (row-major). This is used by downstream tools
(e.g. the web-based spectral visualizer) to co-register multiple modalities without
re-running SIFT.

---

## Citation

If you use OrthoStitch in research, please cite:

```bibtex
@software{orthostitch2026,
  title   = {{OrthoStitch}: lossless ortho-anchored panorama stitching
             for multi-modal cultural heritage imaging},
  author  = {Mohamed, Osama and Color Imaging Lab, University of Granada},
  year    = {2026},
  url     = {https://github.com/ossama971/orthostitch}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for the full text.

> Developed at the **Color Imaging Lab**, University of Granada
> ([colorimaginglab.ugr.es](https://colorimaginglab.ugr.es/))
