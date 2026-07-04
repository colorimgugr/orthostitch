# Multi-Modality Tile Input — Design Spec

Date: 2026-06-29
Status: approved (design), pending implementation plan

## Context

`hsi_stitcher` currently stitches **only** ENVI BIL hyperspectral cubes. The
same artwork has other imaging modalities captured as photographic tiles —
VIS (visible), IRR (IR reflectography), UVF (UV fluorescence), UVR (UV
reflectance) — stored as Nikon `.nef` RAW files (~9–11 tiles per modality) in
`Retrato de Pedro José Pérez Valiente/TP/{VIS,IRR,UVF,UVR}/`. These need the
same lossless ortho-anchored stitching the hyperspectral pipeline provides.

Goal: extend the tool to accept 1- and 3-channel photographic tiles (PNG, JPG,
TIFF, NEF) in addition to hyperspectral BIL, reusing the existing geometry
engine unchanged, and adjust the GUI accordingly. The geometry engine is
already format-agnostic (it works on a derived grayscale proxy + arrays shaped
`(lines, bands, samples)`); only **tile reading** and **mosaic writing** are
format-specific.

### Decisions (confirmed with user)
- **Reference:** ortho **always required** — every modality anchors to the
  supplied orthoimage, exactly like HSI today (no ortho-less fallback).
- **Formats:** PNG/JPG/TIFF natively (cv2/PIL); Nikon **NEF** via the optional
  `rawpy` dependency (lazy-imported, clear error if missing).
- **Output (image modalities):** lossless multi-channel **TIFF** + PNG preview
  + seam map + side-by-side + `placement.json`. HSI keeps writing BIL.

## Losslessness contract (reframed)

Every output pixel is a **bit-exact copy of one source tile pixel's channels**
(spectra for HSI; RGB or gray for images). Geometry is estimated only on
derived grayscale proxies; pixel data moves exclusively by integer
(nearest-neighbor) indexing. For already-lossy sources (JPEG, demosaiced NEF)
"lossless" means we never *re-* degrade them: decoded pixels are copied exactly
and written to a lossless container (TIFF). The runtime `lossless_check`
(random sampled pixels asserted equal to source) is retained and is
dtype-agnostic.

## Architecture

New module `hsi_stitcher/modality.py` isolates the only format-specific
concerns: reading tiles and writing the mosaic.

```
make_loader(tiles_dir) -> auto-detect by file extension present in the folder:
    *.bil (+ matching *.bil.hdr)   -> EnviLoader   (wraps existing envi_io)
    *.png/*.jpg/*.jpeg/*.tif/*.tiff/*.nef -> ImageLoader
  (if both present, or neither, raise a clear error)
```

### Loader interface (both `EnviLoader` and `ImageLoader` implement it)

| Member | Meaning |
|---|---|
| `names` | sorted list of tile base names |
| `shape` | common full-resolution `(h, w)`; error if tiles differ |
| `n_bands` | 121 (HSI) / 1 or 3 (images) |
| `dtype` | `float32` (HSI) / `uint8` or `uint16` (images) |
| `cube(i)` | `ndarray (h, n_bands, w)` full-res, for assembly + lossless copy |
| `reg(i)` | `(gray_uint8, inv_scale)` — capped-res proxy + coord scale factor |
| `rgb_bands` | tuple of channel indices used for the preview |
| `write(out_dir, name, pano, desc, report)` | format-appropriate output |

`EnviLoader`:
- `cube(i)` = `envi_io.open_bil` memmap (already `(lines, bands, samples)`).
- `reg(i)` = existing `registration_image` (mean of 500–900 nm bands, stretch,
  CLAHE) at native 512 px → `inv_scale = 1.0`.
- `rgb_bands` from wavelengths (nearest to 640/550/460 nm) via existing
  `select_bands`.
- `write` = existing `envi_io.write_bil` (+ preview/seam already done in core).

`ImageLoader`:
- Decode each tile: PNG/JPG/TIFF via cv2 (`IMREAD_UNCHANGED`, normalize to RGB
  order, keep bit depth); NEF via `rawpy.imread(...).postprocess()` (camera WB,
  EXIF orientation applied) → RGB uint8/uint16.
- `cube(i)` = decoded image transposed `(H, W, C) -> (H, C, W)` so the middle
  axis is "bands" (channels), matching the engine's expected layout.
  Grayscale → `C = 1`.
- `n_bands` = channel count (1 or 3); `dtype` from the decoded image.
- `reg(i)` = luminance/mean of channels, percentile-stretched to uint8, built at
  a **capped long side** (default 1600 px) → `inv_scale = full_long / capped_long`.
- `rgb_bands` = `(0,1,2)` for 3-channel (R,G,B), `(0,0,0)` for 1-channel.
- `write` = `tifffile.imwrite(<name>_pano.tif, ...)` lossless, channels-last.

## Registration proxy with scale factor

NEF tiles are ~45 MP; SIFT on them is slow/heavy. `reg()` builds the proxy at a
capped resolution and returns `inv_scale`. When features are detected, keypoint
coordinates are **immediately multiplied by `inv_scale`** so they live in
**full-resolution tile coordinates**. Therefore everything downstream —
`Geom.from_shape(h, w)` (full res), homographies, ortho anchoring, and the
`build_maps` inverse map that indexes the source tile — operates in full-res
tile space, and the lossless copy reads full-res pixels. HSI uses
`inv_scale = 1.0`, so its behavior is identical to today.

Descriptors are computed on the downscaled proxy (fine — SIFT is scale-
invariant); only the `.pt` coordinates are rescaled.

## Engine changes (small, in `stitcher.py`)

1. **Feature extraction** (`match_pairs`, and the tile side of `match_anchors`):
   accept per-tile `inv_scale` and scale keypoint `.pt` to full-res coords.
2. **`assemble`**: allocate `pano` in the **loader dtype** (was hardcoded
   `float32`) — correctness for integer images + much lower memory.
3. **`select_bands`**: only called for ENVI; `ImageLoader` supplies `rgb_bands`
   directly (no wavelength dependency).
4. **`stitch(...)`** rewired to be loader-driven:
   `loader = make_loader(tiles_dir)`; build `reg`/`inv_scale`; match → place →
   anchor → `build_maps(loader.shape)` → `assemble(dtype=loader.dtype)` →
   `lossless_check` → orientation → `loader.write(...)` + shared preview/seam/
   side-by-side/json. The geometry functions (`spanning_tree_init`,
   `joint_refine`, `match_anchors` global fit, `normalize_scale`, `build_maps`,
   Voronoi) are otherwise unchanged.
5. `pseudo_rgb`/`side_by_side`/`seam_palette` reused as-is (work on any
   `n_bands`/dtype after a light dtype guard in `pseudo_rgb`).
6. **Per-tile lazy access:** `assemble` and `lossless_check` obtain each tile via
   `loader.cube(t)` one tile at a time (not a preloaded list), so only one
   full-res image is resident at once besides the canvas — important for NEF.
   `EnviLoader.cube` returns a memmap (already lazy); `ImageLoader.cube` decodes
   on demand and may cache the most-recently-used tile.

## GUI changes (`gui.py`, minimal)

- Rename input field **"BIL folder" → "Tiles folder"**; placeholder lists
  supported formats (`.bil/.png/.jpg/.tif/.nef`).
- On folder selection, call a lightweight detector and show a **status line**
  under the field, e.g. `VIS · 10 image tiles (.nef, 3-channel)` or
  `12 hyperspectral tiles (.bil, 121 bands)`. Detection errors (mixed/empty
  folder) shown inline; Stitch stays disabled until valid.
- File-dialog browse unchanged (folder picker). Ortho still required.
- Result viewers unchanged — both modalities emit `<name>_pano_rgb.png` and
  `seam_map.png`.

## Dependencies

Add `rawpy` to `requirements.txt` (optional at runtime; only needed for `.nef`).
`tifffile` already present (used for ortho); now also used for TIFF output.

## Known risks (accepted)

- **Cross-modal anchoring:** UVF/IRR look unlike the visible ortho, so SIFT
  anchoring may be weak (needs ≥3 tiles). Failure surfaces as the existing
  clear "only N tiles anchored to ortho; need >= 3" error; the Advanced
  **SIFT contrast** threshold is the mitigation lever.
- **Memory at full resolution:** the final mosaic for ~10 NEF tiles can be a
  multi-GB array. Tractable on a normal workstation; the proxy trick limits
  this to the assembly/output stage, not feature matching.

## Verification

1. **No HSI regression:** re-run PedroJose BIL via CLI; confirm bit-exact
   lossless, RMS ≈ 0.45 px, k1/k2 ≈ +0.0236/−0.0030 (unchanged from current).
2. **Image modality (standard):** convert/point at a PNG or TIFF tile set with
   the ortho; confirm it discovers tiles, builds proxies, stitches, passes the
   lossless check, and writes `<name>_pano.tif` + preview + seam.
3. **NEF:** with `rawpy` installed, run on `TP/VIS` (10 `.nef`) + the
   orthoimage; confirm decode→stitch→TIFF end to end. (If VIS anchors poorly,
   record it; it validates the risk note, not a code defect.)
4. **GUI:** select each folder type; confirm the status line reports the right
   modality/format/tile-count, viewers display results, output folder opens.

## Out of scope

- Ortho-less (tile-to-tile only) stitching.
- Cross-modal registration (stitching one modality against another's result).
- Large-grid (hundreds of tiles) performance work.
- HDR/exposure blending or photometric harmonization (losslessness forbids it).
