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
