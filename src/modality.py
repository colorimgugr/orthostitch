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
    """1-99% stretch + CLAHE + downscale to REG_MAX_PX. Returns the uint8 proxy."""
    long_side = max(gray_full.shape[:2])
    g = gray_full
    if long_side > REG_MAX_PX:
        s = REG_MAX_PX / long_side
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = g.astype(np.float64)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) / (hi - lo + 1e-12), 0, 1)
    g8 = (g * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g8)


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

    def preview(self, pano):
        # hyperspectral: synthesize a contrast-stretched pseudo-RGB (radiance
        # bands are not display-ready)
        import stitcher
        return stitcher.pseudo_rgb(pano, self.rgb_bands)

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
        # LRU cube cache: decoding (esp. NEF) is expensive, so each tile is
        # decoded once and reused across reg()/assemble()/lossless_check().
        # Bounded by a memory budget so large tile sets don't blow up RAM.
        tile_bytes = h * w * self.n_bands * self.dtype.itemsize
        self._cache_max = max(2, int(1.5e9 / max(tile_bytes, 1)))
        self._cache = {}                         # idx -> cube
        self._lru = []                           # idx order, oldest first
        # eagerly reject mismatched shapes where we can read dims cheaply
        # (NEF dims are validated lazily at decode time in _full)
        for p in paths[1:]:
            dims = self._dims(p)
            if dims is not None and dims != self.shape:
                raise RuntimeError(
                    f"tile {p.stem} shape {dims} != {self.shape}; all tiles must match")

    @staticmethod
    def _dims(path):
        """Cheap (h, w) without full decode; None for formats we can't probe."""
        if path.suffix.lower() == ".nef":
            return None
        try:
            from PIL import Image
            with Image.open(str(path)) as im:
                w, h = im.size
            return (h, w)
        except Exception:
            return None

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
        if i in self._cache:                     # LRU hit
            self._lru.remove(i)
            self._lru.append(i)
            return self._cache[i]
        img = self._full(i)
        cube = img[:, None, :] if img.ndim == 2 else np.transpose(img, (0, 2, 1))
        cube = np.ascontiguousarray(cube)        # (h, bands, w)
        self._cache[i] = cube
        self._lru.append(i)
        while len(self._lru) > self._cache_max:  # evict oldest
            self._cache.pop(self._lru.pop(0), None)
        return cube

    def reg(self, i):
        # derive the proxy from the cached full decode (no extra decode), then
        # downscale; inv_scale maps proxy coords back to full-res tile px
        cube = self.cube(i)
        if self.n_bands == 1:
            gray_full = cube[:, 0, :]
        else:
            img = np.transpose(cube, (0, 2, 1))  # (h, w, bands) RGB
            rgb8 = img if img.dtype == np.uint8 else \
                cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gray_full = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
        gray = _reg_from_gray(gray_full)
        inv_scale = max(self.shape) / max(gray.shape)
        return gray, inv_scale

    def preview(self, pano):
        # photographic: show the captured channels as-is (no contrast stretch),
        # so the preview matches the real snapshot look. Only converts bit depth.
        if self.n_bands == 1:
            ch = pano[:, 0, :]
            rgb = np.stack([ch, ch, ch], axis=-1)
        else:
            rgb = np.transpose(pano[:, :3, :], (0, 2, 1))   # (H,W,3) in R,G,B
        if rgb.dtype != np.uint8:                            # 16-bit -> 8-bit, linear
            maxv = np.iinfo(rgb.dtype).max if np.issubdtype(rgb.dtype, np.integer) \
                else (float(rgb.max()) or 1.0)
            rgb = np.clip(rgb.astype(np.float32) * (255.0 / maxv), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

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
