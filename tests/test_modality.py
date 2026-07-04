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
    rng = np.random.default_rng(1)
    g1 = rng.integers(0, 255, (30, 30), np.uint8)
    g2 = rng.integers(0, 255, (30, 30), np.uint8)
    _write_png(tmp_path / "g1.png", g1)
    _write_png(tmp_path / "g2.png", g2)
    ld = modality.make_loader(tmp_path)
    assert ld.n_bands == 1
    assert ld.rgb_bands == (0, 0, 0)
    assert ld.cube(0).shape == (30, 1, 30)


def test_imageloader_preview_no_stretch(tmp_path):
    # photographic preview must reproduce the captured channels exactly (no
    # contrast stretch), unlike the hyperspectral pseudo-RGB path
    rng = np.random.default_rng(7)
    a = rng.integers(30, 200, (16, 20, 3), np.uint8)  # mid-range, not full 0-255
    _write_png(tmp_path / "t1.png", a)
    _write_png(tmp_path / "t2.png", a.copy())
    ld = modality.make_loader(tmp_path)
    # build a pano-shaped array (h, bands, w) from tile 0 and preview it
    pano = ld.cube(0)
    prev = ld.preview(pano)
    assert prev.shape == (16, 20, 3) and prev.dtype == np.uint8
    assert np.array_equal(prev, a)                    # identical, not stretched


def test_reg_caps_resolution_and_reports_scale(tmp_path):
    rng = np.random.default_rng(2)
    big1 = rng.integers(0, 255, (3200, 2400, 3), np.uint8)
    big2 = rng.integers(0, 255, (3200, 2400, 3), np.uint8)
    _write_png(tmp_path / "a.png", big1)
    _write_png(tmp_path / "b.png", big2)
    ld = modality.make_loader(tmp_path)
    gray, inv_scale = ld.reg(0)
    assert gray.ndim == 2 and gray.dtype == np.uint8
    assert max(gray.shape) == modality.REG_MAX_PX          # capped to 1600
    assert inv_scale == pytest.approx(3200 / 1600, rel=1e-3)  # full/capped


def test_make_loader_errors(tmp_path):
    with pytest.raises(Exception):
        modality.make_loader(tmp_path)        # empty folder
    _write_png(tmp_path / "x.png", np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(Exception):
        modality.make_loader(tmp_path)        # only 1 tile


def test_mismatched_shapes_error(tmp_path):
    _write_png(tmp_path / "a.png", np.zeros((10, 10, 3), np.uint8))
    _write_png(tmp_path / "b.png", np.zeros((12, 10, 3), np.uint8))
    with pytest.raises(Exception):
        modality.make_loader(tmp_path)


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
        modality.make_loader(tmp_path)        # probe decodes tile 0 -> rawpy guidance


@pytest.mark.skipif(__import__("importlib").util.find_spec("rawpy") is None,
                    reason="rawpy not installed")
def test_nef_suffix_routed():
    assert ".nef" in modality.IMAGE_EXTS
