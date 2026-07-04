import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stitcher


def test_detect_features_rescales_keypoints():
    # a textured proxy; inv_scale=2 -> pts doubled vs raw (inv_scale=1)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200), np.uint8)
    raw = stitcher.detect_features([img], [1.0])[0][0]
    scaled = stitcher.detect_features([img], [2.0])[0][0]
    assert raw.shape[1] == 2 and scaled.shape[1] == 2
    assert raw.shape[0] == scaled.shape[0]
    assert np.allclose(scaled, raw * 2.0)


def test_assemble_preserves_dtype_and_values():
    # two tiles, identity maps; left half owned by tile 0, right half by tile 1
    h = w = 4
    n_bands = 3
    cube0 = np.arange(h * n_bands * w, dtype=np.uint8).reshape(h, n_bands, w)
    cube1 = (cube0 + 100).astype(np.uint8)
    cubes = [cube0, cube1]
    ix = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    iy = np.tile(np.arange(h, dtype=np.int32)[:, None], (1, w))
    tile_maps = [(0, h, 0, w, ix, iy), (0, h, 0, w, ix, iy)]
    label = np.full((h, w), 0, np.int16)
    label[:, 2:] = 1
    pano = stitcher.assemble(lambda t: cubes[t], 2, label, tile_maps,
                             (h, w), n_bands, np.uint8)
    assert pano.dtype == np.uint8
    assert np.array_equal(pano[:, :, :2], cube0[:, :, :2])
    assert np.array_equal(pano[:, :, 2:], cube1[:, :, 2:])


def test_pseudo_rgb_on_uint8():
    pano = np.random.default_rng(1).integers(1, 255, (10, 3, 12), np.uint8)
    rgb = stitcher.pseudo_rgb(pano, (0, 1, 2))
    assert rgb.shape == (10, 12, 3) and rgb.dtype == np.uint8
