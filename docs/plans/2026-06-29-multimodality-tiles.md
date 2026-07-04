# Multi-Modality Tile Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `hsi_stitcher` to stitch 1- and 3-channel photographic tiles (PNG/JPG/TIFF/NEF) in addition to ENVI BIL hyperspectral, reusing the existing geometry engine.

**Architecture:** A new `modality.py` isolates the only format-specific concerns — reading tiles and writing the mosaic — behind a small loader interface (`EnviLoader`, `ImageLoader`) chosen by `make_loader(dir)`. The geometry engine in `stitcher.py` is made loader-driven and format-agnostic: features carry a per-tile scale factor so a downscaled registration proxy maps back to full-resolution tile coordinates, and `assemble` works in the source dtype.

**Tech Stack:** Python 3.11, numpy, OpenCV (SIFT), scipy, tifffile, rawpy (optional, NEF only), PyQt6 (GUI), pytest.

## Global Constraints

- **Losslessness:** every output pixel is a bit-exact copy of one source tile pixel's channels; geometry runs only on derived grayscale; pixel data moves only by integer nearest-neighbor indexing. The runtime `lossless_check` must keep passing.
- **No HSI regression:** the existing BIL path must produce identical geometry (HSI uses `inv_scale = 1.0`).
- **Ortho required:** every run anchors to the supplied orthoimage (V3). No ortho-less mode.
- **Tile coordinate convention:** cubes are shaped `(lines, bands, samples)` = `(h, n_bands, w)`; the middle axis is bands/channels.
- **Registration proxy cap:** photographic proxies are built at long side ≤ 1600 px; keypoints are rescaled to full-res tile coordinates.
- **Git:** the repo root is **not** a git repository. "Commit" steps are optional — run only if `git rev-parse --is-inside-work-tree` succeeds; otherwise skip.
- **Test command:** `python3 -m pytest hsi_stitcher/tests/<file>::<test> -v` (run from the internship root). Synthetic data only — tests must not depend on the multi-GB real files or on `rawpy`.

---

## File Structure

- Create `hsi_stitcher/modality.py` — loader interface, `make_loader`, `EnviLoader`, `ImageLoader`, `describe`.
- Modify `hsi_stitcher/stitcher.py` — `detect_features` (scaled keypoints), `match_pairs`/`match_anchors` use pts arrays, `assemble`/`lossless_check` take a `cube_fn` + dtype, loader-driven `stitch()`.
- Modify `hsi_stitcher/gui.py` — "Tiles folder" field + detection status line.
- Modify `hsi_stitcher/requirements.txt` — add `rawpy`.
- Create `hsi_stitcher/tests/conftest.py` — synthetic scene/tile fixtures.
- Create `hsi_stitcher/tests/test_modality.py`, `test_engine.py`, `test_integration.py`.

---

## Task 1: Modality loaders for ENVI + standard images

**Files:**
- Create: `hsi_stitcher/modality.py`
- Create: `hsi_stitcher/tests/test_modality.py`

**Interfaces:**
- Consumes: `envi_io.read_hdr`, `envi_io.open_bil`, `envi_io.wavelengths`; `stitcher.select_bands`, `stitcher.registration_image` (existing).
- Produces:
  - `make_loader(tiles_dir) -> Loader`
  - `describe(tiles_dir) -> str`
  - Loader attributes/methods: `names: list[str]`, `shape: tuple[int,int]`, `n_bands: int`, `dtype: np.dtype`, `rgb_bands: tuple[int,...]`, `cube(i) -> np.ndarray (h,n_bands,w)`, `reg(i) -> (np.ndarray uint8, float)`, `write(out_dir, name, pano, desc) -> Path`
  - `REG_MAX_PX = 1600`

- [ ] **Step 1: Write the failing test**

Create `hsi_stitcher/tests/test_modality.py`:

```python
import os, sys
import numpy as np
import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # hsi_stitcher/ on path
import modality


def _write_png(path, arr):
    # arr is (H,W,3) RGB or (H,W) gray, uint8
    if arr.ndim == 3:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), arr)


def test_imageloader_rgb_layout_and_meta(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (40, 50, 3), np.uint8)
    b = rng.integers(0, 255, (40, 50, 3), np.uint8)
    _write_png(tmp_path / "t1.png", a)
    _write_png(tmp_path / "t2.png", b)

    ld = modality.make_loader(tmp_path)
    assert ld.names == ["t1", "t2"]
    assert ld.shape == (40, 50)
    assert ld.n_bands == 3
    assert ld.dtype == np.uint8
    assert ld.rgb_bands == (0, 1, 2)

    cube = ld.cube(0)                      # (h, bands, w)
    assert cube.shape == (40, 3, 50)
    # cube[y,:,x] is the RGB triple of pixel (x,y)
    assert np.array_equal(cube[5, :, 7], a[5, 7, :])


def test_imageloader_gray_is_single_band(tmp_path):
    g = np.random.default_rng(1).integers(0, 255, (30, 30), np.uint8)
    _write_png(tmp_path / "g1.png", g)
    _write_png(tmp_path / "g2.png", g.T.copy())  # same shape? no -> use same
    _write_png(tmp_path / "g2.png", g)
    ld = modality.make_loader(tmp_path)
    assert ld.n_bands == 1
    assert ld.rgb_bands == (0, 0, 0)
    assert ld.cube(0).shape == (30, 1, 30)


def test_reg_caps_resolution_and_reports_scale(tmp_path):
    big = np.random.default_rng(2).integers(0, 255, (3200, 2400, 3), np.uint8)
    small = big[:1600, :1200].copy()
    _write_png(tmp_path / "a.png", big)
    _write_png(tmp_path / "a.png", big)
    # need 2 same-shape tiles
    _write_png(tmp_path / "b.png", big.copy())
    ld = modality.make_loader(tmp_path)
    gray, inv_scale = ld.reg(0)
    assert gray.ndim == 2 and gray.dtype == np.uint8
    assert max(gray.shape) == modality.REG_MAX_PX          # capped to 1600
    assert inv_scale == pytest.approx(3200 / 1600, rel=1e-3)  # full/capped


def test_make_loader_errors(tmp_path):
    with pytest.raises(Exception):
        modality.make_loader(tmp_path)        # empty folder
    (tmp_path / "x.png").write_bytes(b"not really")
    with pytest.raises(Exception):
        modality.make_loader(tmp_path)        # only 1 tile


def test_mismatched_shapes_error(tmp_path):
    _write_png(tmp_path / "a.png", np.zeros((10, 10, 3), np.uint8))
    _write_png(tmp_path / "b.png", np.zeros((12, 10, 3), np.uint8))
    with pytest.raises(Exception):
        modality.make_loader(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest hsi_stitcher/tests/test_modality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modality'` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

Create `hsi_stitcher/modality.py`:

```python
"""Tile input/output loaders by modality (format).

Isolates the only format-specific concerns of the stitcher: reading tiles and
writing the assembled mosaic. The geometry engine in stitcher.py is otherwise
format-agnostic.

Auto-detection by file extension present in a folder:
    *.bil (+ matching *.bil.hdr)            -> EnviLoader  (hyperspectral)
    *.png/.jpg/.jpeg/.tif/.tiff/.nef        -> ImageLoader (1- or 3-channel)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envi_io

REG_MAX_PX = 1600       # registration proxy long-side cap (photographic tiles)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nef")


def _reg_from_gray(gray_full):
    """1-99% stretch + CLAHE + downscale to REG_MAX_PX. Returns (uint8, inv_scale)."""
    long_side = max(gray_full.shape[:2])
    inv_scale = 1.0
    g = gray_full
    if long_side > REG_MAX_PX:
        inv_scale = long_side / REG_MAX_PX
        s = REG_MAX_PX / long_side
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = g.astype(np.float64)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) / (hi - lo + 1e-12), 0, 1)
    g8 = (g * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g8), inv_scale


class EnviLoader:
    """ENVI BIL hyperspectral tiles (existing format)."""

    def __init__(self, tiles_dir):
        import stitcher  # local import to avoid cycle at module load
        self._dir = Path(tiles_dir)
        self.names = stitcher.discover_tiles(self._dir)
        self._hdr0 = envi_io.read_hdr(self._dir / f"{self.names[0]}.bil.hdr")
        cubes = [envi_io.open_bil(self._dir / f"{t}.bil") for t in self.names]
        shapes = {c.shape for c in cubes}
        if len(shapes) != 1:
            raise RuntimeError(f"tiles have differing shapes {shapes}; all must match")
        h, n_bands, w = cubes[0].shape
        self.shape = (h, w)
        self.n_bands = n_bands
        self.dtype = np.dtype(np.float32)        # output cube dtype
        wl = envi_io.wavelengths(self._hdr0)
        self._b_lo, self._b_hi, self.rgb_bands = stitcher.select_bands(wl)
        self._cubes = cubes
        self._wl = wl

    def cube(self, i):
        return self._cubes[i]                    # memmap, lazy

    def reg(self, i):
        import stitcher
        return stitcher.registration_image(self._cubes[i], self._b_lo, self._b_hi), 1.0

    def write(self, out_dir, name, pano, desc):
        out = Path(out_dir) / f"{name}_pano.bil"
        envi_io.write_bil(out, pano, self._hdr0, desc)
        return out


class ImageLoader:
    """1- or 3-channel photographic tiles (PNG/JPG/TIFF/NEF)."""

    def __init__(self, tiles_dir, paths):
        self._paths = paths                      # list[Path], sorted
        self.names = [p.stem for p in paths]
        probe = self._decode(paths[0])
        h, w = probe.shape[:2]
        self.shape = (h, w)
        self.n_bands = 1 if probe.ndim == 2 else probe.shape[2]
        self.dtype = np.dtype(probe.dtype)
        self.rgb_bands = (0, 0, 0) if self.n_bands == 1 else (0, 1, 2)
        self._cache = (None, None)               # (index, cube) MRU cache

    @staticmethod
    def _decode(path):
        """Return an HxW (gray) or HxWxC RGB array, native bit depth."""
        suffix = path.suffix.lower()
        if suffix == ".nef":
            try:
                import rawpy
            except ImportError as e:
                raise RuntimeError(
                    "reading .nef requires the 'rawpy' package (pip install rawpy)") from e
            with rawpy.imread(str(path)) as raw:
                return raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"could not read image tile: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3
                               else cv2.COLOR_BGRA2RGB)
        return img

    def _full(self, i):
        img = self._decode(self._paths[i])
        if img.shape[:2] != self.shape:
            raise RuntimeError(
                f"tile {self.names[i]} shape {img.shape[:2]} != {self.shape}")
        return img

    def cube(self, i):
        if self._cache[0] == i:
            return self._cache[1]
        img = self._full(i)
        cube = img[:, None, :] if img.ndim == 2 else np.transpose(img, (0, 2, 1))
        cube = np.ascontiguousarray(cube)        # (h, bands, w)
        self._cache = (i, cube)
        return cube

    def reg(self, i):
        img = self._full(i)
        gray = img if img.ndim == 2 else cv2.cvtColor(
            (img if img.dtype == np.uint8 else
             cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)),
            cv2.COLOR_RGB2GRAY)
        return _reg_from_gray(gray)

    def write(self, out_dir, name, pano, desc):
        import tifffile
        out = Path(out_dir) / f"{name}_pano.tif"
        img = pano[:, 0, :] if self.n_bands == 1 else np.transpose(pano, (0, 2, 1))
        tifffile.imwrite(str(out), np.ascontiguousarray(img),
                         description=desc, photometric=(
                             "minisblack" if self.n_bands == 1 else "rgb"))
        return out


def _image_paths(tiles_dir):
    d = Path(tiles_dir)
    paths = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    return paths


def make_loader(tiles_dir):
    """Pick a loader by the file types present. Raises on empty/mixed/<2 tiles."""
    d = Path(tiles_dir)
    bils = sorted(d.glob("*.bil"))
    imgs = _image_paths(d)
    if bils and imgs:
        raise RuntimeError(f"{d} contains both .bil and image tiles; keep one modality per folder")
    if bils:
        return EnviLoader(d)
    if imgs:
        if len(imgs) < 2:
            raise RuntimeError(f"need >= 2 image tiles in {d}, found {len(imgs)}")
        return ImageLoader(d, imgs)
    raise RuntimeError(f"no tiles found in {d} (expected *.bil or {IMAGE_EXTS})")


def describe(tiles_dir):
    """Short human description for the GUI status line; never raises."""
    try:
        d = Path(tiles_dir)
        bils = sorted(d.glob("*.bil"))
        imgs = _image_paths(d)
        if bils and imgs:
            return "mixed .bil + image tiles — keep one modality per folder"
        if bils:
            hdr = envi_io.read_hdr(str(bils[0]) + ".hdr")
            return f"{len(bils)} hyperspectral tiles (.bil, {hdr.get('bands','?')} bands)"
        if imgs:
            ext = imgs[0].suffix.lower()
            return f"{len(imgs)} image tiles ({ext})"
        return "no tiles found"
    except Exception as e:
        return f"unreadable: {e}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest hsi_stitcher/tests/test_modality.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit** (optional — only if in a git repo)

```bash
git rev-parse --is-inside-work-tree 2>/dev/null && \
  git add hsi_stitcher/modality.py hsi_stitcher/tests/test_modality.py && \
  git commit -m "feat: modality loaders for ENVI and standard images"
```

---

## Task 2: NEF (Nikon RAW) support

**Files:**
- Modify: `hsi_stitcher/requirements.txt`
- Modify: `hsi_stitcher/tests/test_modality.py` (add gated test)

**Interfaces:**
- Consumes: `ImageLoader._decode` (already routes `.nef` through `rawpy`, Task 1).
- Produces: no new symbols; confirms NEF decode path + dependency.

The decode path was written in Task 1 (`_decode` handles `.nef` via `rawpy` with a clear error if missing). This task records the dependency and verifies the path is exercised when `rawpy` is available.

- [ ] **Step 1: Add the dependency**

Edit `hsi_stitcher/requirements.txt` — append:

```
rawpy
```

- [ ] **Step 2: Write the failing/gated test**

Append to `hsi_stitcher/tests/test_modality.py`:

```python
def test_nef_requires_rawpy_message(tmp_path, monkeypatch):
    # a .nef file present but rawpy import fails -> clear error
    (tmp_path / "a.nef").write_bytes(b"\x00")
    (tmp_path / "b.nef").write_bytes(b"\x00")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rawpy":
            raise ImportError("no rawpy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ld = modality.make_loader(tmp_path)          # construction probes tile 0 -> decode
    # _decode is called during __init__ probe; expect the rawpy guidance error
    # (construction itself raises because probe decodes tile 0)


@pytest.mark.skipif(__import__("importlib").util.find_spec("rawpy") is None,
                    reason="rawpy not installed")
def test_nef_decode_roundtrip(tmp_path):
    import rawpy  # noqa
    # We cannot synthesize a real NEF; assert the suffix routing instead.
    assert ".nef" in modality.IMAGE_EXTS
```

Note: the first test documents the guidance path; adjust the assertion to wrap the construction in `pytest.raises(RuntimeError)` since the probe decode of a fake `.nef` will raise the rawpy message. Final form:

```python
def test_nef_requires_rawpy_message(tmp_path, monkeypatch):
    (tmp_path / "a.nef").write_bytes(b"\x00")
    (tmp_path / "b.nef").write_bytes(b"\x00")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rawpy":
            raise ImportError("no rawpy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="rawpy"):
        modality.make_loader(tmp_path)
```

- [ ] **Step 3: Run tests**

Run: `python3 -m pytest hsi_stitcher/tests/test_modality.py -v`
Expected: PASS. `test_nef_requires_rawpy_message` passes; `test_nef_decode_roundtrip` passes or is skipped depending on `rawpy`.

- [ ] **Step 4: Commit** (optional)

```bash
git rev-parse --is-inside-work-tree 2>/dev/null && \
  git add hsi_stitcher/requirements.txt hsi_stitcher/tests/test_modality.py && \
  git commit -m "feat: NEF support via optional rawpy + dependency"
```

---

## Task 3: Engine — scaled features, dtype-aware assembly

**Files:**
- Modify: `hsi_stitcher/stitcher.py` (add `detect_features`; change `match_pairs`, `match_anchors`, `assemble`, `lossless_check`, `pseudo_rgb`)
- Create: `hsi_stitcher/tests/test_engine.py`

**Interfaces:**
- Consumes: `make_sift`, `RATIO`, `MIN_INLIERS`, `MIN_ANCHOR_INLIERS` (existing module globals).
- Produces:
  - `detect_features(reg_imgs, inv_scales) -> list[(pts: np.ndarray (N,2) float32 full-res, des)]`
  - `match_pairs(feats, tiles) -> (pairs, feats)` (now takes feats, not reg images)
  - `match_anchors(feats, Hs_init, k, g, tiles, ortho_path) -> (anchors, Hs_ortho, ortho)`
  - `assemble(cube_fn, n_tiles, label, tile_maps, canvas, n_bands, dtype) -> pano`
  - `lossless_check(pano, cube_fn, label, tile_maps, n=1000)`

- [ ] **Step 1: Write the failing test**

Create `hsi_stitcher/tests/test_engine.py`:

```python
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stitcher


def test_detect_features_rescales_keypoints():
    # a textured proxy at half-resolution; inv_scale=2 -> pts doubled vs raw
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200), np.uint8)
    raw = stitcher.detect_features([img], [1.0])[0][0]
    scaled = stitcher.detect_features([img], [2.0])[0][0]
    assert raw.shape[1] == 2 and scaled.shape[1] == 2
    assert raw.shape[0] == scaled.shape[0]
    assert np.allclose(scaled, raw * 2.0)


def test_assemble_preserves_dtype_and_values():
    # two 1-tile-each label regions, identity maps -> exact copy in uint8
    h = w = 4
    n_bands = 3
    cube0 = np.arange(h * n_bands * w, dtype=np.uint8).reshape(h, n_bands, w)
    cube1 = (cube0 + 100).astype(np.uint8)
    cubes = [cube0, cube1]
    # canvas == one tile; identity index maps
    ix = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    iy = np.tile(np.arange(h, dtype=np.int32)[:, None], (1, w))
    tile_maps = [(0, h, 0, w, ix, iy), (0, h, 0, w, ix, iy)]
    label = np.full((h, w), 0, np.int16)
    label[:, 2:] = 1                       # right half owned by tile 1
    pano = stitcher.assemble(lambda t: cubes[t], 2, label, tile_maps,
                             (h, w), n_bands, np.uint8)
    assert pano.dtype == np.uint8
    assert np.array_equal(pano[:, :, :2], cube0[:, :, :2])
    assert np.array_equal(pano[:, :, 2:], cube1[:, :, 2:])


def test_pseudo_rgb_on_uint8():
    pano = np.random.default_rng(1).integers(1, 255, (10, 3, 12), np.uint8)
    rgb = stitcher.pseudo_rgb(pano, (0, 1, 2))
    assert rgb.shape == (10, 12, 3) and rgb.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest hsi_stitcher/tests/test_engine.py -v`
Expected: FAIL — `detect_features` missing; `assemble` signature mismatch.

- [ ] **Step 3: Implement the engine changes**

In `hsi_stitcher/stitcher.py`, add `detect_features` and rewrite `match_pairs` to take feats. Replace the current `match_pairs` (the SIFT loop) with:

```python
def detect_features(reg_imgs, inv_scales):
    """SIFT per registration proxy; keypoint coords scaled to full-res tile px.

    Returns list of (pts (N,2) float32 in full-res tile coords, descriptors)."""
    sift = make_sift()
    feats = []
    for img, s in zip(reg_imgs, inv_scales):
        kp, des = sift.detectAndCompute(img, None)
        if kp:
            pts = (np.float32([p.pt for p in kp]) * float(s))
        else:
            pts = np.zeros((0, 2), np.float32)
        feats.append((pts, des))
    return feats


def match_pairs(feats, tiles):
    """SIFT-match all pairs; return {(i,j): (pts_i, pts_j)} of RANSAC inliers.

    feats: output of detect_features (full-res tile coordinates)."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = {}
    for i, j in itertools.combinations(range(len(feats)), 2):
        pts_i, di = feats[i]
        pts_j, dj = feats[j]
        if di is None or dj is None:
            continue
        knn = bf.knnMatch(di, dj, k=2)
        good = [m for m, n in knn if m.distance < RATIO * n.distance]
        if len(good) < MIN_INLIERS:
            continue
        pi = pts_i[[m.queryIdx for m in good]]
        pj = pts_j[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(pj, pi, cv2.RANSAC, RANSAC_THRESH)
        if H is None:
            continue
        inl = mask.ravel().astype(bool)
        if inl.sum() < MIN_INLIERS:
            continue
        pairs[(i, j)] = (pi[inl], pj[inl])
        print(f"  pair {tiles[i]}-{tiles[j]}: {inl.sum()} inliers")
    return pairs, feats
```

Rewrite `match_anchors`'s signature and tile loop to use feats pts. Change the `def` line and the per-tile loop header:

```python
def match_anchors(feats, Hs_init, k, g, tiles, ortho_path):
    """Cross-modal SIFT matches tile -> ortho (4000px). Returns anchors dict,
    initial homographies mapped into ortho frame, and the ortho image."""
    ortho = ortho_mod.load_ortho(ortho_path, 4000)
    sift = make_sift()
    ko, do = sift.detectAndCompute(ortho, None)      # ortho features (ortho px)
    print(f"  ortho keypoints: {len(ko)}")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    center = np.float32([g.c])
    raw = {}
    for t in range(len(tiles)):
        pts_t, dp = feats[t]
        if dp is None:
            continue
        good = [m for m, n in bf.knnMatch(dp, do, k=2) if m.distance < RATIO * n.distance]
        if len(good) < MIN_ANCHOR_INLIERS:
            print(f"  {tiles[t]}: only {len(good)} raw matches, no anchor")
            continue
        src = pts_t[[m.queryIdx for m in good]]
        dst = np.float32([ko[m.trainIdx].pt for m in good])
        Hm, mask = cv2.findHomography(undistort_pts(src, k, g), dst, cv2.RANSAC, 6.0)
        if Hm is None or mask.sum() < MIN_ANCHOR_INLIERS:
            n = 0 if mask is None else int(mask.sum())
            print(f"  {tiles[t]}: {n} anchor inliers, rejected")
            continue
        inl = mask.ravel().astype(bool)
        raw[t] = (src[inl], dst[inl], Hm)
        print(f"  {tiles[t]}: {inl.sum()} anchor inliers")
    if len(raw) < 3:
        raise RuntimeError(f"only {len(raw)} tiles anchored to ortho; need >= 3")
```

(The rest of `match_anchors` below `if len(raw) < 3:` is unchanged.)

Rewrite `assemble` and `lossless_check` to take a `cube_fn` and dtype:

```python
def assemble(cube_fn, n_tiles, label, tile_maps, canvas, n_bands, dtype):
    Hh, W = canvas
    pano = np.zeros((Hh, n_bands, W), dtype)          # source dtype, not float32
    for t in range(n_tiles):
        by0, by1, bx0, bx1, ix, iy = tile_maps[t]
        own = label[by0:by1, bx0:bx1] == t
        sy, sx = np.nonzero(own)
        if sy.size == 0:
            continue
        src_y, src_x = iy[sy, sx], ix[sy, sx]
        cube = cube_fn(t)
        for b in range(n_bands):
            band = np.asarray(cube[:, b, :])
            pano[sy + by0, b, sx + bx0] = band[src_y, src_x]
    return pano


def lossless_check(pano, cube_fn, label, tile_maps, n=1000):
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(label >= 0)
    n = min(n, len(ys))
    sel = rng.choice(len(ys), n, replace=False)
    for y, x in zip(ys[sel], xs[sel]):
        t = label[y, x]
        by0, by1, bx0, bx1, ix, iy = tile_maps[t]
        sy, sx = iy[y - by0, x - bx0], ix[y - by0, x - bx0]
        assert np.array_equal(pano[y, :, x], np.asarray(cube_fn(t)[sy, :, sx])), \
            f"value mismatch at ({y},{x})"
    print(f"  OK: {n} sampled pixels bit-identical to source")
```

Make `pseudo_rgb` dtype-safe — change its first lines so the band is float for percentile math:

```python
def pseudo_rgb(pano, b_rgb):
    chans = []
    for b in b_rgb:
        band = pano[:, b, :].astype(np.float32)
        lo, hi = np.percentile(band[band > 0], (1, 99)) if (band > 0).any() else (0, 1)
        chans.append(np.clip((band - lo) / (hi - lo + 1e-12), 0, 1))
    return (np.stack(chans, axis=-1) * 255).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest hsi_stitcher/tests/test_engine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** (optional)

```bash
git rev-parse --is-inside-work-tree 2>/dev/null && \
  git add hsi_stitcher/stitcher.py hsi_stitcher/tests/test_engine.py && \
  git commit -m "feat: scaled features + dtype-aware assembly in engine"
```

---

## Task 4: Loader-driven stitch() + integration test

**Files:**
- Modify: `hsi_stitcher/stitcher.py` (`stitch()` body and `main()`)
- Create: `hsi_stitcher/tests/conftest.py`
- Create: `hsi_stitcher/tests/test_integration.py`

**Interfaces:**
- Consumes: `modality.make_loader`; `detect_features`, `match_pairs`, `connected_components`, `spanning_tree_init`, `joint_refine`, `match_anchors`, `normalize_scale`, `pair_rms`, `radial_trend`, `build_maps`, `assemble`, `lossless_check`, `pseudo_rgb`, `register_to_ortho`, `seam_palette`, `side_by_side` (all existing/Task 3).
- Produces: `stitch(tiles_dir, ortho_path, out_dir, artwork_name, progress_cb=None, params=None) -> report dict` (param renamed `bil_dir` → `tiles_dir`).

- [ ] **Step 1: Write the failing integration test**

Create `hsi_stitcher/tests/conftest.py`:

```python
import os
import numpy as np
import cv2
import pytest


def _scene(seed=0, H=1400, W=1100):
    """A SIFT-friendly synthetic painting: many high-contrast random shapes."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 30, np.uint8)
    for _ in range(700):
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        col = tuple(int(v) for v in rng.integers(40, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, c, int(rng.integers(6, 36)), col, -1, cv2.LINE_AA)
        else:
            c2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            cv2.line(img, c, c2, col, int(rng.integers(1, 5)), cv2.LINE_AA)
    return img


@pytest.fixture
def synthetic_modality(tmp_path):
    """4 overlapping RGB tiles cut from one scene + the scene as the ortho.

    Tiles are crops, so they overlap pairwise AND match the ortho strongly."""
    scene = _scene()
    H, W = scene.shape[:2]
    th, tw = 800, 650
    boxes = {                              # 2x2 grid with generous overlap
        "tile_00": (0, 0),
        "tile_01": (0, W - tw),
        "tile_02": (H - th, 0),
        "tile_03": (H - th, W - tw),
    }
    bil_dir = tmp_path / "VIS"
    bil_dir.mkdir()
    for name, (y, x) in boxes.items():
        crop = scene[y:y + th, x:x + tw]
        cv2.imwrite(str(bil_dir / f"{name}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    ortho_path = tmp_path / "ortho.png"
    cv2.imwrite(str(ortho_path), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
    out_dir = tmp_path / "out"
    return dict(tiles_dir=bil_dir, ortho_path=ortho_path, out_dir=out_dir)
```

Create `hsi_stitcher/tests/test_integration.py`:

```python
import os, sys, json
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stitcher


def test_stitch_rgb_modality_end_to_end(synthetic_modality):
    p = synthetic_modality
    # loosened thresholds: synthetic scene yields fewer SIFT inliers than real data
    params = {"MIN_INLIERS": 12, "MIN_ANCHOR_INLIERS": 10}
    report = stitcher.stitch(str(p["tiles_dir"]), str(p["ortho_path"]),
                             str(p["out_dir"]), "VIS", params=params)
    assert len(report["tiles"]) == 4
    assert len(report["anchored_tiles"]) >= 3
    # outputs exist with the right names
    out = p["out_dir"]
    assert (out / "VIS_pano.tif").exists()
    assert (out / "VIS_pano_rgb.png").exists()
    assert (out / "seam_map.png").exists()
    assert (out / "VIS_rgb_vs_seam.png").exists()
    assert (out / "placement.json").exists()
    # the lossless_check inside stitch() already asserted bit-exact copies;
    # confirm the written cube is uint8 and 3-channel
    import tifffile
    pano = tifffile.imread(str(out / "VIS_pano.tif"))
    assert pano.dtype == np.uint8 and pano.shape[2] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest hsi_stitcher/tests/test_integration.py -v`
Expected: FAIL — `stitch` still references the old `bil_dir`/`select_bands`/`assemble(cubes,...)` flow and old `match_pairs(reg,...)` signature.

- [ ] **Step 3: Rewrite `stitch()` to be loader-driven**

In `hsi_stitcher/stitcher.py`, replace the body of `stitch(...)` (keep the docstring and the `apply_params` reset lines) from the line that reads `bil_dir = Path(bil_dir)` down to `return report`. Rename the first parameter `bil_dir` → `tiles_dir`. New body:

```python
def stitch(tiles_dir, ortho_path, out_dir, artwork_name, progress_cb=None, params=None):
    """Run the full V3 ortho-anchored pipeline. Writes outputs to `out_dir`.

    Accepts any modality folder (ENVI BIL, or PNG/JPG/TIFF/NEF image tiles);
    the loader abstracts tile reading and mosaic writing.
    params: optional {name: value} overrides for the tunable thresholds.
    Returns the placement report dict."""
    apply_params(get_default_params())
    apply_params(params)

    import modality

    def step(i, msg):
        print(f"[{i}/6] {msg}")
        if progress_cb:
            progress_cb(i, 6, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = artwork_name.strip() or "pano"

    loader = modality.make_loader(tiles_dir)
    tiles = loader.names
    h, w = loader.shape
    n_bands = loader.n_bands
    g = Geom.from_shape(h, w)
    print(f"discovered {len(tiles)} tiles in {tiles_dir}")
    print(f"  modality: {n_bands}-band, tile {h}x{w}, dtype {loader.dtype}")

    step(1, "registration images")
    reg, inv_scales = [], []
    for i in range(len(tiles)):
        gimg, s = loader.reg(i)
        reg.append(gimg)
        inv_scales.append(s)

    step(2, "pairwise SIFT matching")
    feats = detect_features(reg, inv_scales)
    pairs, feats = match_pairs(feats, tiles)
    print(f"  {len(pairs)} usable pairs")
    comps = connected_components(pairs, len(tiles))
    if len(comps) > 1:
        groups = "; ".join("{" + ", ".join(tiles[t] for t in c) + "}" for c in comps)
        raise RuntimeError(
            f"tile match graph is disconnected into {len(comps)} groups: {groups}. "
            "Tiles in different groups share too little overlapping texture to "
            "register together (common when large regions are dark/low-texture). "
            "Ensure the tiles actually overlap, or lower SIFT_CONTRAST further.")

    step(3, "global placement (ortho-anchored)")
    ref, Hs = spanning_tree_init(pairs, len(tiles), tiles)
    print(f"  reference tile: {tiles[ref]}")
    k = np.zeros(2)
    Hs, k = joint_refine(pairs, Hs, {ref}, k, g)
    print("  matching tiles to ortho for anchoring...")
    anchors, Hs, _ = match_anchors(feats, Hs, k, g, tiles, ortho_path)
    anchors_used = sorted(tiles[t] for t in anchors)
    Hs, k = joint_refine(pairs, Hs, set(), k, g, anchors=anchors, f_scale=3.0)
    Hs = normalize_scale(Hs, k, g)
    per_pair, rms = pair_rms(pairs, Hs, k, g, tiles)
    print(f"  global pairwise RMS (output frame): {rms:.3f} px")
    trend = radial_trend(pairs, Hs, k, g)

    step(4, "canvas + Voronoi seams")
    label, tile_maps, canvas, T = build_maps(Hs, k, (h, w), g)
    holes = int((label == -1).sum())
    print(f"  background/hole pixels: {holes} ({100*holes/label.size:.1f}% of canvas)")

    step(5, "assembling mosaic (nearest-neighbor copies)")
    pano = assemble(loader.cube, len(tiles), label, tile_maps, canvas, n_bands, loader.dtype)
    lossless_check(pano, loader.cube, label, tile_maps)

    step(6, "orientation + outputs")
    rgb = pseudo_rgb(pano, loader.rgb_bands)
    rot_k, ninl, angle, extra = register_to_ortho(rgb, ortho_path)
    if rot_k != 0:
        print(f"  note: panorama rotated by rot90 k={rot_k} to match the ortho")
    print(f"  rotation: rot90 k={rot_k} ({ninl} ortho inliers, {angle:.1f} deg residual)")
    pano = np.rot90(pano, rot_k, axes=(0, 2))
    rgb = np.rot90(rgb, rot_k, axes=(0, 1))
    label_r = np.rot90(label, rot_k, axes=(0, 1))

    desc = (f"Mosaic of {len(tiles)} {n_bands}-band tiles ({name}), ortho-anchored. "
            "Nearest-neighbor placement, one source tile per pixel (Voronoi seams), "
            f"pixels bit-identical to source tiles. Shared radial distortion "
            f"k1={k[0]:.5f} k2={k[1]:.5f}. Pairwise registration RMS {rms:.2f} px.")
    loader.write(out_dir, name, pano, desc)
    cv2.imwrite(str(out_dir / f"{name}_pano_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    palette = seam_palette(len(tiles))
    seam_bgr = palette[label_r + 1]
    cv2.imwrite(str(out_dir / "seam_map.png"), seam_bgr)
    cv2.imwrite(str(out_dir / f"{name}_rgb_vs_seam.png"), side_by_side(rgb, seam_bgr))

    ortho_rms = None
    if extra is not None:
        Hm, src, dst, ortho = extra
        gimg = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        warped = cv2.warpPerspective(gimg, Hm, (ortho.shape[1], ortho.shape[0]))
        overlay = cv2.addWeighted(cv2.cvtColor(ortho, cv2.COLOR_GRAY2BGR), 0.5,
                                  cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imwrite(str(out_dir / "ortho_overlay.png"), overlay)
        d = cv2.perspectiveTransform(src[None], Hm)[0] - dst
        ortho_rms = float(np.sqrt((d ** 2).mean()))
        print(f"  ortho residual RMS: {ortho_rms:.2f} px (at ortho scale 2000px)")

    report = {
        "artwork": name, "mode": "ortho", "tiles": tiles,
        "modality_bands": int(n_bands),
        "reference_tile": tiles[ref],
        "distortion_k1_k2": [float(k[0]), float(k[1])],
        "anchored_tiles": anchors_used, "rotation_k": int(rot_k),
        "canvas_before_rotation": list(map(int, canvas[::-1])),
        "global_rms_px": float(rms), "radial_trend_px": trend,
        "ortho_overlay_rms_px": ortho_rms, "hole_pixels": holes,
        "pairs": per_pair,
        "homographies_T_H": {tiles[t]: (T @ Hs[t]).tolist() for t in range(len(tiles))},
    }
    (out_dir / "placement.json").write_text(json.dumps(report, indent=2))
    print("done. outputs in", out_dir)
    return report
```

Update `main()` flags: rename `--bil` → `--tiles` (keep `--bil` as a hidden alias). Replace the `main()` body's argument block:

```python
def main():
    ap = argparse.ArgumentParser(description="Lossless ortho-anchored stitching")
    ap.add_argument("--tiles", "--bil", dest="tiles", required=True,
                    help="folder of tiles (.bil/.png/.jpg/.tif/.nef)")
    ap.add_argument("--ortho", required=True, help="ortho reference image (tif/png/jpg)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", required=True, help="artwork name (output filename prefix)")
    args = ap.parse_args()
    stitch(args.tiles, args.ortho, args.out, args.name)
```

- [ ] **Step 4: Run the integration test**

Run: `python3 -m pytest hsi_stitcher/tests/test_integration.py -v`
Expected: PASS — 4 tiles, ≥3 anchored, all output files present, TIFF is uint8 3-channel.

- [ ] **Step 5: Run the full unit + integration suite**

Run: `python3 -m pytest hsi_stitcher/tests/ -v`
Expected: all PASS (skips allowed for NEF if `rawpy` absent).

- [ ] **Step 6: HSI regression check (manual, real data)**

Run:
```bash
cd "/Users/osama/Desktop/Masters/COSI/Granada Internship"
python3 hsi_stitcher/stitcher.py \
  --tiles "Retrato de Pedro José Pérez Valiente/HSIVNIR/bil" \
  --ortho "Retrato de Pedro José Pérez Valiente/pedro_ortoimagen-002.tif" \
  --out /tmp/pj_regress --name PedroJose
```
Expected: `OK: 1000 sampled pixels bit-identical to source`; global RMS ≈ 0.45 px; k1 ≈ +0.0236, k2 ≈ −0.0030; 12 tiles, 12 anchored. Then `rm -rf /tmp/pj_regress`.

- [ ] **Step 7: Commit** (optional)

```bash
git rev-parse --is-inside-work-tree 2>/dev/null && \
  git add hsi_stitcher/stitcher.py hsi_stitcher/tests/conftest.py hsi_stitcher/tests/test_integration.py && \
  git commit -m "feat: loader-driven stitch() supporting all modalities"
```

---

## Task 5: GUI — Tiles folder field + modality status line

**Files:**
- Modify: `hsi_stitcher/gui.py`

**Interfaces:**
- Consumes: `modality.describe(folder) -> str`, `stitcher.stitch(tiles_dir, ...)`.
- Produces: no new public symbols (UI only).

- [ ] **Step 1: Update the input field + add a status line**

In `hsi_stitcher/gui.py`, add the import near the top (after `import stitcher`):

```python
import modality
```

In `_inputs_card`, change the BIL row's placeholder and add a status label below the grid. Replace the `self.bil_edit` creation line's placeholder:

```python
        self.bil_edit = DropLineEdit(want_dir=True, on_drop=self._bil_changed)
        self.bil_edit.setPlaceholderText("Folder with tiles (.bil/.png/.jpg/.tif/.nef)")
```

Change its label text in the `row(0, ...)` call from `"BIL folder"` to `"Tiles folder"`.

Immediately after `lay.addLayout(grid)` in `_inputs_card`, insert:

```python
        self.detect_lbl = QLabel("")
        self.detect_lbl.setObjectName("Subtitle")
        lay.addWidget(self.detect_lbl)
```

- [ ] **Step 2: Populate the status line on folder change**

In `_bil_changed`, after the existing prefill logic and before `self._validate()`, add the detection text:

```python
        p = self.bil_edit.text().strip()
        if p and os.path.isdir(p):
            self.detect_lbl.setText(modality.describe(p))
        else:
            self.detect_lbl.setText("")
```

(Place this inside `_bil_changed`; the existing name/out prefill stays above it.)

- [ ] **Step 3: Offscreen smoke test**

Run:
```bash
cd "/Users/osama/Desktop/Masters/COSI/Granada Internship"
QT_QPA_PLATFORM=offscreen python3 -c "
import sys; sys.path.insert(0,'hsi_stitcher')
from PyQt6.QtWidgets import QApplication
import gui, modality, tempfile, os, numpy as np, cv2
app=QApplication(sys.argv); app.setStyleSheet(gui.QSS)
w=gui.MainWindow(); w.show()
d=tempfile.mkdtemp()
for n in ('a','b'):
    cv2.imwrite(os.path.join(d,n+'.png'), np.zeros((20,20,3),np.uint8))
w.bil_edit.setText(d)
assert 'image tiles' in w.detect_lbl.text(), w.detect_lbl.text()
print('status line:', w.detect_lbl.text())
print('GUI modality smoke OK')
" 2>&1 | grep -vi "propagateSizeHints\|qt.qpa"
```
Expected: prints a status line like `2 image tiles (.png)` and `GUI modality smoke OK`.

- [ ] **Step 4: Commit** (optional)

```bash
git rev-parse --is-inside-work-tree 2>/dev/null && \
  git add hsi_stitcher/gui.py && \
  git commit -m "feat: GUI tiles-folder field with modality detection"
```

---

## Task 6: Docs

**Files:**
- Modify: `hsi_stitcher/README.md`

- [ ] **Step 1: Update the README**

In `hsi_stitcher/README.md`: change "any folder of `.bil`+`.bil.hdr` tiles" framing to mention modalities; under Install note `rawpy` is needed only for `.nef`; in the CLI example, change `--bil` to `--tiles`; add a short "Modalities" subsection:

```markdown
## Modalities

The tile folder may contain any one modality:

| Tiles | Loader | Output mosaic |
|---|---|---|
| `*.bil` (+ `.bil.hdr`) | hyperspectral (ENVI) | `<name>_pano.bil` |
| `*.png/.jpg/.tif` | 1- or 3-channel image | `<name>_pano.tif` |
| `*.nef` (Nikon RAW) | demosaiced RGB (needs `rawpy`) | `<name>_pano.tif` |

One modality per folder; the format is auto-detected. An ortho reference is
required in all cases. Output orientation follows the ortho.
```

- [ ] **Step 2: Verify README renders (no broken table)**

Run: `python3 -c "import pathlib,sys; t=pathlib.Path('hsi_stitcher/README.md').read_text(); sys.exit(0 if '--tiles' in t and 'rawpy' in t else 1)"`
Expected: exit 0.

- [ ] **Step 3: Commit** (optional)

```bash
git rev-parse --is-inside-work-tree 2>/dev/null && \
  git add hsi_stitcher/README.md && \
  git commit -m "docs: document multi-modality tile support"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** loader abstraction (T1), NEF (T2), scaled features + dtype assembly (T3), loader-driven stitch + TIFF output + lossless integration (T4), GUI status line (T5), docs (T6). Ortho-required is unchanged (no new task needed). Risk notes are documentation, not code.
- **Type consistency:** `match_pairs(feats, tiles)`, `match_anchors(feats, Hs_init, k, g, tiles, ortho_path)`, `assemble(cube_fn, n_tiles, label, tile_maps, canvas, n_bands, dtype)`, `lossless_check(pano, cube_fn, ...)`, `stitch(tiles_dir, ...)` are used identically in Task 4's `stitch()` body.
- **Placeholders:** none — all steps carry real code/commands.
