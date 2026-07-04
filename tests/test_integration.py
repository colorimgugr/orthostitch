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
    # pano->ortho homography persisted for cross-modality co-registration
    H_po = report["pano_to_ortho_H"]
    assert H_po is not None and np.asarray(H_po).shape == (3, 3)
