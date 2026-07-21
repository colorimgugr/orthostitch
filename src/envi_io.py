"""Minimal ENVI BIL reader/writer for the PedroJose VNIR cubes.

BIL layout on disk: line-major, i.e. shape (lines, bands, samples).
"""

import re
from pathlib import Path

import numpy as np

DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
}


def read_hdr(path):
    """Parse an ENVI .hdr file into a dict (keys lowercased).

    Values in {...} (possibly spanning lines) are returned as lists of
    strings; everything else as a stripped string.
    """
    text = Path(path).read_text()
    if not text.lstrip().startswith("ENVI"):
        raise ValueError(f"{path} is not an ENVI header")
    # Strip the leading "ENVI" magic, then join continuation lines of {...}
    body = text.lstrip()[4:]
    hdr = {}
    # Match "key = value" where value is either a {...} block or a single line
    pattern = re.compile(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", re.MULTILINE | re.DOTALL)
    for m in pattern.finditer(body):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if val.startswith("{"):
            items = val.strip("{}").replace("\n", " ")
            hdr[key] = [s.strip() for s in items.split(",") if s.strip()]
        else:
            hdr[key] = val
    return hdr


def open_bil(bil_path, hdr=None):
    """Memmap a BIL cube as (lines, bands, samples). Read-only."""
    bil_path = Path(bil_path)
    if hdr is None:
        hdr = read_hdr(str(bil_path) + ".hdr")
    if hdr.get("interleave", "bil").lower() != "bil":
        raise ValueError(f"expected BIL interleave, got {hdr.get('interleave')}")
    lines = int(hdr["lines"])
    samples = int(hdr["samples"])
    bands = int(hdr["bands"])
    dtype = DTYPE_MAP[int(hdr["data type"])]
    if int(hdr.get("byte order", 0)) != 0:
        dtype = np.dtype(dtype).newbyteorder(">")
    return np.memmap(bil_path, dtype=dtype, mode="r", shape=(lines, bands, samples))


def write_bil(out_path, cube, template_hdr, description=None):
    """Write (lines, bands, samples) float32 cube as BIL + .hdr.

    Copies spectral metadata (wavelengths, units, reflectance scale factor)
    from template_hdr.
    """
    out_path = Path(out_path)
    cube = np.ascontiguousarray(cube, dtype=np.float32)
    lines, bands, samples = cube.shape
    cube.tofile(out_path)

    parts = ["ENVI"]
    if description:
        parts.append("description = {%s}" % description)
    parts += [
        "interleave = bil",
        "data type = 4",
        f"lines = {lines}",
        f"samples = {samples}",
        f"bands = {bands}",
        "byte order = 0",
    ]
    for key in ("wavelength units", "reflectance scale factor"):
        if key in template_hdr:
            parts.append(f"{key} = {template_hdr[key]}")
    if "wavelength" in template_hdr:
        parts.append("wavelength = {%s}" % ", ".join(template_hdr["wavelength"]))
    Path(str(out_path) + ".hdr").write_text("\n".join(parts) + "\n")


def wavelengths(hdr):
    return np.array([float(w) for w in hdr["wavelength"]])
