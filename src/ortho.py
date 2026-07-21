"""Ortho reference image loader.

Returns an upright uint8 grayscale image at roughly a requested long-side size.

Unlike the per-artwork prototype, this does NOT force any 90-degree rotation:
the ortho is loaded exactly as supplied, so the stitched panorama comes out in
the same orientation as the ortho it is anchored to. Large pyramidal/tiled TIFs
are decoded from an appropriate pyramid level so the full-resolution image is
never materialized.
"""

from pathlib import Path

import cv2
import numpy as np

_CACHE = {}  # (resolved_path, max_px) -> grayscale uint8 image


def _to_uint8_gray(img):
    """Coerce a decoded array to a contiguous uint8 grayscale image."""
    if img.ndim == 3:                         # drop alpha, convert color -> gray
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:                 # normalize non-8-bit data to 0-255
        img = img.astype(np.float64)
        lo, hi = np.percentile(img, (1, 99))
        img = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
        img = (img * 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _load_tif(path, max_px):
    """Decode the smallest TIF pyramid level whose long side >= max_px."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        pages = sorted(tf.pages, key=lambda p: p.shape[1])      # smallest first
        page = next((p for p in pages if max(p.shape[:2]) >= max_px), pages[-1])
        img = page.asarray()
    return img


def _load_other(path):
    """Decode a regular image (PNG/JPG/...) as RGB via OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)               # BGR or None
    if img is None:
        raise ValueError(f"could not read ortho image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                 # match TIF RGB order


def load_ortho(path, max_px):
    """Upright uint8 grayscale ortho at roughly `max_px` on the long side.

    TIFs are decoded from a pyramid level; other formats are decoded fully then
    downscaled. Results are cached by (path, max_px)."""
    path = Path(path)
    key = (str(path.resolve()), int(max_px))
    if key in _CACHE:
        return _CACHE[key]

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        img = _load_tif(path, max_px)
    else:
        img = _load_other(path)

    gray = _to_uint8_gray(img)

    # downscale toward the requested size (only if meaningfully larger), using
    # area interpolation so SIFT sees a clean, non-aliased reference
    long_side = max(gray.shape[:2])
    if long_side > max_px * 1.25:
        s = max_px / long_side
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = np.ascontiguousarray(gray)

    _CACHE[key] = gray
    return gray
