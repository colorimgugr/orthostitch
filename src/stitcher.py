"""Generalized lossless ortho-anchored hyperspectral panorama stitching.

Generalized from the per-artwork PedroJose prototype. Works on any folder of
ENVI BIL tiles (each `*.bil` with a matching `*.bil.hdr`) plus any ortho image.
Everything artwork-specific is derived from the data at runtime:

  - the tile set is discovered from the folder (no hardcoded names),
  - registration / pseudo-RGB bands are chosen by wavelength (from the headers),
  - the radial-distortion center/scale come from the tile dimensions,
  - the ortho is used exactly as supplied (no forced rotation).

Losslessness contract (unchanged from the prototype): every output pixel is a
bit-exact copy of one input spectrum. Geometry is estimated on derived
grayscale images only; spectral data is moved exclusively by integer
(nearest-neighbor) indexing.

Always runs the V3 "ortho-anchored" pipeline:
  selfcal (shared radial distortion estimated from tile overlaps) + each tile
  additionally anchored to the orthorectified reference image.
"""

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# allow running both as a script (`python hsi_stitcher/stitcher.py`) and as a
# module (`python -m hsi_stitcher.stitcher`) by ensuring this dir is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ortho as ortho_mod

# --- algorithmic thresholds (not artwork-specific) ------------------------
MIN_INLIERS = 30            # min RANSAC inliers to trust a tile-to-tile pair
MIN_ANCHOR_INLIERS = 20     # min inliers to trust a tile-to-ortho anchor
RANSAC_THRESH = 3.0         # reprojection error (px) for pair homography RANSAC
RATIO = 0.75                # Lowe ratio-test threshold for SIFT matches
ANCHOR_WEIGHT = 0.5         # down-weight of anchor residuals vs pair residuals

# SIFT sensitivity. OpenCV's default contrast threshold (0.04) discards too many
# low-contrast keypoints on dark paintings, which fragments the match graph into
# disconnected clusters and starves the ortho anchoring. A lower threshold
# recovers enough genuine features in dark/low-texture regions to register the
# full mosaic; RANSAC still rejects the spurious matches it lets through.
SIFT_CONTRAST = 0.01
SIFT_EDGE = 20
# keep at most this many SIFT keypoints per image (strongest by response). Only
# bites texture-rich photographic tiles (which yield tens of thousands and make
# matching quadratically slow); HSI/dark tiles have far fewer, so unaffected.
SIFT_MAX_FEATURES = 10000
# cap correspondences per pair/anchor fed to bundle adjustment. A homography has
# 8 DOF; a few hundred well-spread inliers fit it as well as thousands, but the
# numeric-Jacobian solve cost scales with the residual count.
MAX_PAIR_CORR = 300

# cap the output mosaic long side (px). The pipeline normally renders at native
# tile resolution; for high-resolution photographic tiles (e.g. ~45 MP NEF) that
# is a gigapixel canvas that exhausts RAM. Capping keeps each output pixel a
# bit-exact nearest-neighbor copy of a source pixel (losslessness preserved) but
# subsamples the grid. HSI native canvases are far below this, so unaffected.
MAX_CANVAS_PX = 12000


def make_sift(nfeatures=None):
    """SIFT detector tuned for low-contrast (dark painting) tiles.

    nfeatures defaults to SIFT_MAX_FEATURES (cap for tiles, which are matched
    all-pairs); pass 0 for the ortho reference, which is matched only once per
    tile and benefits from its full, rich keypoint set for anchoring."""
    n = SIFT_MAX_FEATURES if nfeatures is None else nfeatures
    return cv2.SIFT_create(nfeatures=n,
                           contrastThreshold=SIFT_CONTRAST, edgeThreshold=SIFT_EDGE)


def cap_corr(pi, pj, n=None, rng=None):
    """Subsample paired correspondences to at most n (deterministic)."""
    n = MAX_PAIR_CORR if n is None else n
    if n and len(pi) > n:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(pi), n, replace=False)
        return pi[idx], pj[idx]
    return pi, pj


def connected_components(pairs, n):
    """Connected components of tile indices over the pair (edge) keys."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j) in pairs:
        parent[find(i)] = find(j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=len, reverse=True)


# ----------------------------------------------------------------------------
# run-time tunable parameters (exposed by the GUI's advanced-options dialog)
# ----------------------------------------------------------------------------
TUNABLE_PARAMS = (
    "SIFT_CONTRAST", "SIFT_EDGE", "RATIO", "MIN_INLIERS",
    "MIN_ANCHOR_INLIERS", "RANSAC_THRESH", "ANCHOR_WEIGHT", "MAX_CANVAS_PX",
)
# captured once, on pristine globals, so "reset to defaults" always works
_DEFAULT_PARAMS = {k: globals()[k] for k in TUNABLE_PARAMS}


def get_default_params():
    """Default values of the tunable parameters (a fresh copy)."""
    return dict(_DEFAULT_PARAMS)


def apply_params(params):
    """Override tunable module constants for this run, casting to their type."""
    if not params:
        return
    g = globals()
    for k, v in params.items():
        if k in TUNABLE_PARAMS:
            g[k] = type(_DEFAULT_PARAMS[k])(v)

# wavelengths (nm) used to pick bands; resolved to nearest available indices
REG_BAND_RANGE = (500.0, 900.0)   # registration image = mean over this range
RGB_WAVELENGTHS = (640.0, 550.0, 460.0)  # pseudo-RGB channels


@dataclass
class Geom:
    """Shared radial-distortion geometry, derived from the tile size."""
    c: np.ndarray   # distortion center (image center), shape (2,)
    f: float        # normalization radius

    @classmethod
    def from_shape(cls, h, w):
        # center at the pixel midpoint; f = half-width (reproduces the
        # prototype's 255.5 / 256 for 512 px tiles)
        return cls(c=np.array([(w - 1) / 2.0, (h - 1) / 2.0]), f=w / 2.0)


# ----------------------------------------------------------------------------
# tile discovery and band selection
# ----------------------------------------------------------------------------
def discover_tiles(bil_dir):
    """Return sorted tile base names for every `*.bil` with a matching `.hdr`."""
    bil_dir = Path(bil_dir)
    names = []
    for bil in sorted(bil_dir.glob("*.bil")):
        if (bil_dir / (bil.name + ".hdr")).exists():
            names.append(bil.stem)            # e.g. "PedroJose_001"
    if len(names) < 2:
        raise RuntimeError(
            f"need >= 2 .bil/.bil.hdr tile pairs in {bil_dir}, found {len(names)}")
    return names


def select_bands(wl):
    """Pick band indices from a wavelength axis (nm, ascending).

    Returns (reg_lo, reg_hi, b_rgb): a half-open band range for the registration
    image and a (r, g, b) index triple for the pseudo-RGB preview."""
    wl = np.asarray(wl, float)
    lo = int(np.searchsorted(wl, REG_BAND_RANGE[0], side="left"))
    hi = int(np.searchsorted(wl, REG_BAND_RANGE[1], side="right"))
    if hi - lo < 5:                           # range not covered -> use all bands
        lo, hi = 0, len(wl)
    b_rgb = tuple(int(np.argmin(np.abs(wl - t))) for t in RGB_WAVELENGTHS)
    return lo, hi, b_rgb


# ----------------------------------------------------------------------------
# distortion model (parameterized by Geom instead of module globals)
# ----------------------------------------------------------------------------
def undistort_pts(pts, k, g):
    """Distorted sensor coords -> ideal coords (closed form)."""
    n = (pts - g.c) / g.f                            # normalized coords about center
    r2 = (n ** 2).sum(axis=-1, keepdims=True)        # squared radius per point
    return g.c + g.f * n * (1 + k[0] * r2 + k[1] * r2 ** 2)  # radial poly, de-normalize


def distort_pts(pts_u, k, g, iters=12):
    """Ideal coords -> distorted sensor coords (fixed-point inversion)."""
    nu = (pts_u - g.c) / g.f                         # ideal points, normalized
    nd = nu.copy()                                   # initial guess
    for _ in range(iters):                           # iterate toward the inverse
        r2 = (nd ** 2).sum(axis=-1, keepdims=True)
        nd = nu / (1 + k[0] * r2 + k[1] * r2 ** 2)
    return g.c + g.f * nd                            # de-normalize to sensor px


def project(H, pts):
    p = pts @ H[:, :2].T + H[:, 2]                   # apply homography (homogeneous)
    return p[:, :2] / p[:, 2:3]                      # perspective divide


# ----------------------------------------------------------------------------
# feature work (all on derived grayscale; never touches output spectra)
# ----------------------------------------------------------------------------
def registration_image(cube, b_lo, b_hi):
    """Grayscale image for feature matching only (never touches output)."""
    acc = np.zeros((cube.shape[0], cube.shape[2]), np.float64)  # (rows, cols)
    for b in range(b_lo, b_hi):                      # average the chosen bands
        acc += cube[:, b, :]
    img = acc / (b_hi - b_lo)
    lo, hi = np.percentile(img, (1, 99))             # robust intensity range
    img = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
    img8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # local contrast
    return clahe.apply(img8)


def detect_features(reg_imgs, inv_scales):
    """SIFT per registration proxy; keypoint coords scaled to full-res tile px.

    Returns list of (pts (N,2) float32 in full-res tile coords, descriptors)."""
    sift = make_sift()
    feats = []
    for img, s in zip(reg_imgs, inv_scales):
        kp, des = sift.detectAndCompute(img, None)
        if kp:
            pts = np.float32([p.pt for p in kp]) * float(s)  # -> full-res coords
        else:
            pts = np.zeros((0, 2), np.float32)
        feats.append((pts, des))
    return feats


def match_pairs(feats, tiles):
    """SIFT-match all pairs; return {(i,j): (pts_i, pts_j)} of RANSAC inliers.

    feats: output of detect_features (coordinates already in full-res tile px)."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = {}
    for i, j in itertools.combinations(range(len(feats)), 2):  # every tile pair
        pts_i, di = feats[i]
        pts_j, dj = feats[j]
        if di is None or dj is None:
            continue
        knn = bf.knnMatch(di, dj, k=2)               # 2 NN per descriptor
        good = [m for m, n in knn if m.distance < RATIO * n.distance]  # ratio test
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
        pairs[(i, j)] = (pi[inl], pj[inl])           # keep inlier correspondences
        print(f"  pair {tiles[i]}-{tiles[j]}: {inl.sum()} inliers")
    return pairs, feats


def spanning_tree_init(pairs, n, tiles):
    """Compose per-tile homographies to a reference along a max-inlier tree."""
    Hp = {}
    for (i, j), (pi, pj) in pairs.items():           # relative homography per pair
        H, _ = cv2.findHomography(pj, pi, 0)
        Hp[(i, j)] = H
        Hp[(j, i)] = np.linalg.inv(H)
    weight = {k: len(v[0]) for k, v in pairs.items()}            # edge weight = inliers
    conn = [sum(w for (i, j), w in weight.items() if ref in (i, j)) for ref in range(n)]
    ref = int(np.argmax(conn))                       # best-connected tile = reference
    H_glob = {ref: np.eye(3)}                        # reference at identity
    while len(H_glob) < n:                           # grow tree over heaviest edges
        best = None
        for (i, j), w in weight.items():
            for a, b in ((i, j), (j, i)):
                if a in H_glob and b not in H_glob:
                    if best is None or w > best[0]:
                        best = (w, a, b)
        if best is None:                             # graph split -> cannot place all
            placed = sorted(tiles[k] for k in H_glob)
            raise RuntimeError(f"match graph is disconnected; placed only {placed}")
        _, a, b = best
        H_glob[b] = H_glob[a] @ Hp[(a, b)]           # chain through placed neighbor
    return ref, [H_glob[k] / H_glob[k][2, 2] for k in range(n)]  # normalize h33=1


def joint_refine(pairs, Hs, fixed, k0, g, anchors=None, estimate_k=True, f_scale=1.0):
    """Jointly refine homographies (those not in `fixed`) and shared (k1,k2)."""
    n = len(Hs)
    free = [t for t in range(n) if t not in fixed]   # tiles we optimize
    Hs = [H / H[2, 2] for H in Hs]                   # normalize before packing 8 params
    x0 = np.concatenate([Hs[t].ravel()[:8] for t in free]
                        + ([np.asarray(k0, float)] if estimate_k else []))

    def unpack(x):                                   # flat vector -> matrices + k
        out = [Hs[t] for t in range(n)]
        for idx, t in enumerate(free):
            h = np.append(x[idx * 8:(idx + 1) * 8], 1.0)  # re-append fixed h33=1
            out[t] = h.reshape(3, 3)
        k = x[-2:] if estimate_k else np.asarray(k0, float)
        return out, k

    def residuals(x):                                # reprojection errors to minimize
        Hx, k = unpack(x)
        res = []
        for (i, j), (pi, pj) in pairs.items():       # same point via both tiles must agree
            res.append(project(Hx[i], undistort_pts(pi, k, g))
                       - project(Hx[j], undistort_pts(pj, k, g)))
        if anchors:                                  # optional tile->ortho pull
            for t, (pt, q) in anchors.items():
                res.append(ANCHOR_WEIGHT
                           * (project(Hx[t], undistort_pts(pt, k, g)) - q))
        return np.concatenate(res).ravel()

    r0 = residuals(x0)                               # before optimization
    sol = least_squares(residuals, x0, method="trf", loss="soft_l1",
                        f_scale=f_scale, max_nfev=3000)
    Hs_out, k_out = unpack(sol.x)
    rms0 = np.sqrt(np.mean(r0 ** 2))
    rms1 = np.sqrt(np.mean(sol.fun ** 2))
    print(f"  joint refinement RMS: {rms0:.3f} -> {rms1:.3f}  "
          f"k1={k_out[0]:+.5f} k2={k_out[1]:+.5f}")
    return Hs_out, np.asarray(k_out)


def pair_rms(pairs, Hs, k, g, tiles):
    """Per-pair and global RMS in the (possibly normalized) output frame."""
    per_pair = {}
    sq, n = 0.0, 0
    for (i, j), (pi, pj) in pairs.items():
        d = (project(Hs[i], undistort_pts(pi, k, g))
             - project(Hs[j], undistort_pts(pj, k, g)))
        per_pair[f"{tiles[i]}-{tiles[j]}"] = {
            "inliers": int(len(pi)),
            "rms_px": float(np.sqrt(np.mean(d ** 2))),
        }
        sq += float((d ** 2).sum())
        n += d.size
    return per_pair, float(np.sqrt(sq / n))


def radial_trend(pairs, Hs, k, g):
    """Mean radial residual binned by distance from tile center (diagnostic).

    Bins are fractions of the tile half-diagonal so the diagnostic scales with
    any tile size."""
    rs, rad = [], []
    for (i, j), (pi, pj) in pairs.items():
        d = (project(Hs[i], undistort_pts(pi, k, g))
             - project(Hs[j], undistort_pts(pj, k, g)))
        r = np.linalg.norm(pi - g.c, axis=1)         # distance from center per match
        u = (pi - g.c) / (r[:, None] + 1e-9)         # unit radial direction
        rs.append(r)
        rad.append((d * u).sum(axis=1))              # radial component of residual
    rs, rad = np.concatenate(rs), np.concatenate(rad)
    halfdiag = g.f * np.sqrt(2.0)                     # ~max radius for a square tile
    out = {}
    fracs = [0.0, 0.28, 0.5, 0.72, 1.0]              # bin edges as fractions
    for fl, fh in zip(fracs[:-1], fracs[1:]):
        lo, hi = fl * halfdiag, fh * halfdiag
        m = (rs >= lo) & (rs < hi)
        if m.sum() > 10:
            out[f"r[{lo:.0f},{hi:.0f})"] = round(float(rad[m].mean()), 3)
    print("  radial residual trend:", out)
    return out


def match_anchors(feats, Hs_init, k, g, tiles, ortho_path):
    """Cross-modal SIFT matches tile -> ortho (4000px). Returns anchors dict,
    initial homographies mapped into ortho frame, and the ortho image.

    feats: output of detect_features (tile coords already in full-res px)."""
    ortho = ortho_mod.load_ortho(ortho_path, 4000)
    sift = make_sift(nfeatures=0)                     # ortho: keep full keypoint set
    ko, do = sift.detectAndCompute(ortho, None)      # ortho features (ortho px)
    print(f"  ortho keypoints: {len(ko)}")
    # FLANN index built once on the ~10^5 ortho descriptors; brute force would be
    # O(tiles x ortho) and dominate runtime. Approximate NN + ratio test is the
    # standard fast path for large SIFT sets.
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    flann.add([do])
    flann.train()
    center = np.float32([g.c])                        # tile center for sanity checks
    raw = {}
    for t in range(len(tiles)):                       # anchor each tile to the ortho
        pts_t, dp = feats[t]
        if dp is None:
            continue
        good = [m for m, n in (pr for pr in flann.knnMatch(dp, k=2) if len(pr) == 2)
                if m.distance < RATIO * n.distance]
        if len(good) < MIN_ANCHOR_INLIERS:
            print(f"  {tiles[t]}: only {len(good)} raw matches, no anchor")
            continue
        src = pts_t[[m.queryIdx for m in good]]                  # tile-side points
        dst = np.float32([ko[m.trainIdx].pt for m in good])      # ortho-side points
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

    # global homography ortho <- selfcal frame from all anchor inliers
    src_all = np.concatenate([project(Hs_init[t], undistort_pts(s, k, g))
                              for t, (s, d, _) in raw.items()])
    dst_all = np.concatenate([d for (_, d, _) in raw.values()])
    Ho, mask = cv2.findHomography(src_all, dst_all, cv2.RANSAC, 8.0)
    print(f"  global selfcal->ortho fit: {int(mask.sum())}/{len(src_all)} inliers")
    Hs_ortho = [Ho @ H for H in Hs_init]              # push every tile into ortho frame

    # sanity-verify each tile anchor against the global placement
    anchors = {}
    for t, (s, d, Hm) in raw.items():
        pred = project(Hs_ortho[t], undistort_pts(center, k, g))[0]  # center via global
        own = project(Hm, undistort_pts(center, k, g))[0]            # center via own anchor
        if np.linalg.norm(pred - own) > 150:
            print(f"  {tiles[t]}: anchor disagrees with neighbors "
                  f"({np.linalg.norm(pred - own):.0f}px), dropped")
            continue
        anchors[t] = (s, d)
    print(f"  anchored tiles: {len(anchors)}/{len(tiles)}")
    return anchors, Hs_ortho, ortho


def normalize_scale(Hs, k, g, shape):
    """Compose a global isotropic scale so mean tile scale at center = 1, then
    cap the resulting canvas long side to MAX_CANVAS_PX (subsamples but keeps
    each output pixel a bit-exact source-pixel copy)."""
    scales = []
    center = np.float32([g.c])
    for H in Hs:
        eps = 1.0                                     # finite-difference step
        p0 = project(H, center)[0]
        px = project(H, center + [[eps, 0]])[0]
        py = project(H, center + [[0, eps]])[0]
        A = np.stack([(px - p0) / eps, (py - p0) / eps], axis=1)  # local Jacobian
        scales.append(np.sqrt(abs(np.linalg.det(A))))            # local area scale
    s = 1.0 / np.mean(scales)
    print(f"  scale normalization: mean tile scale {np.mean(scales):.3f}, s={s:.3f}")
    Hs = [np.diag([s, s, 1.0]) @ H for H in Hs]       # -> native tile resolution

    # estimate the resulting canvas long side from warped undistorted corners
    h, w = shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    corners_u = undistort_pts(corners, k, g).astype(np.float32)
    allc = np.concatenate([cv2.perspectiveTransform(corners_u[None], np.float32(H))[0]
                           for H in Hs])
    extent = float(max(np.ptp(allc[:, 0]), np.ptp(allc[:, 1])))
    if MAX_CANVAS_PX and extent > MAX_CANVAS_PX:
        cap = MAX_CANVAS_PX / extent
        print(f"  canvas cap: native long side {extent:.0f}px > {MAX_CANVAS_PX} "
              f"-> additional scale {cap:.3f} (output subsampled, still bit-exact)")
        Hs = [np.diag([cap, cap, 1.0]) @ H for H in Hs]
    return Hs


# ----------------------------------------------------------------------------
# rasterization and assembly
# ----------------------------------------------------------------------------
def build_maps(Hs, k, shape, g):
    """Canvas bounds, per-tile rounded (distortion-aware) inverse maps,
    Voronoi label map."""
    h, w = shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    corners_u = undistort_pts(corners, k, g).astype(np.float32)
    warped = [cv2.perspectiveTransform(corners_u[None], np.float32(H))[0] for H in Hs]
    allc = np.concatenate(warped)
    x0, y0 = np.floor(allc.min(axis=0)).astype(int) - 2          # canvas TL (margin)
    x1, y1 = np.ceil(allc.max(axis=0)).astype(int) + 2           # canvas BR (margin)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)   # origin -> (0,0)
    W, Hh = x1 - x0 + 1, y1 - y0 + 1
    print(f"  canvas: {W} x {Hh}")

    label = np.full((Hh, W), -1, np.int16)            # tile owner per pixel (-1 = none)
    bestd = np.full((Hh, W), np.inf, np.float32)      # nearest tile-center distance
    tile_maps = []
    for t, H in enumerate(Hs):
        Ht = T @ H                                    # include canvas translation
        wc = project(Ht, undistort_pts(np.float32([[w / 2, h / 2]]), k, g))[0]  # center on canvas
        tc = cv2.perspectiveTransform(corners_u[None], np.float32(Ht))[0]       # corners on canvas
        bx0 = max(int(np.floor(tc[:, 0].min())) - 4, 0)          # tile bbox (clamped)
        by0 = max(int(np.floor(tc[:, 1].min())) - 4, 0)
        bx1 = min(int(np.ceil(tc[:, 0].max())) + 5, W)
        by1 = min(int(np.ceil(tc[:, 1].max())) + 5, Hh)
        ys, xs = np.mgrid[by0:by1, bx0:bx1]           # canvas grid for this tile box
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0)
        inv = np.linalg.inv(Ht) @ pts                 # canvas -> ideal tile coords
        und = np.stack([inv[0] / inv[2], inv[1] / inv[2]], axis=-1)
        sen = distort_pts(und, k, g)                  # ideal -> actual sensor px
        ix = np.rint(sen[:, 0]).reshape(ys.shape).astype(np.int32)  # nearest source col
        iy = np.rint(sen[:, 1]).reshape(ys.shape).astype(np.int32)  # nearest source row
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)         # inside this tile?
        d = (xs - wc[0]) ** 2 + (ys - wc[1]) ** 2     # squared dist to tile center
        sub_l = label[by0:by1, bx0:bx1]
        sub_d = bestd[by0:by1, bx0:bx1]
        upd = valid & (d < sub_d)                     # claim pixel if nearest center
        sub_l[upd] = t
        sub_d[upd] = d[upd]
        tile_maps.append((by0, by1, bx0, bx1, ix, iy))
    return label, tile_maps, (Hh, W), T


def assemble(cube_fn, n_tiles, label, tile_maps, canvas, n_bands, dtype):
    """Build the mosaic by pure nearest-neighbor copies, in the source dtype.

    cube_fn(t) -> (h, n_bands, w) source tile, fetched one tile at a time so
    only one full-res tile is resident at once (important for large photos)."""
    Hh, W = canvas
    pano = np.zeros((Hh, n_bands, W), dtype)          # output cube (rows, bands, cols)
    for t in range(n_tiles):
        by0, by1, bx0, bx1, ix, iy = tile_maps[t]
        own = label[by0:by1, bx0:bx1] == t            # canvas px owned by this tile
        sy, sx = np.nonzero(own)
        if sy.size == 0:
            continue
        src_y, src_x = iy[sy, sx], ix[sy, sx]         # corresponding source px
        cube = cube_fn(t)
        for b in range(n_bands):                      # lossless nearest-neighbor copy
            band = np.asarray(cube[:, b, :])
            pano[sy + by0, b, sx + bx0] = band[src_y, src_x]
    return pano


def lossless_check(pano, cube_fn, label, tile_maps, n=1000):
    rng = np.random.default_rng(0)                    # deterministic sampling
    ys, xs = np.nonzero(label >= 0)                   # filled canvas pixels
    n = min(n, len(ys))
    sel = rng.choice(len(ys), n, replace=False)
    ysel, xsel = ys[sel], xs[sel]
    tsel = label[ysel, xsel]
    # visit samples grouped by tile so cube_fn(t) is fetched at most once per
    # tile (a tile may be an expensive on-demand decode, e.g. NEF) rather than
    # thrashing a small cache with random tile order
    for idx in np.argsort(tsel, kind="stable"):
        y, x, t = int(ysel[idx]), int(xsel[idx]), int(tsel[idx])
        by0, by1, bx0, bx1, ix, iy = tile_maps[t]
        sy, sx = iy[y - by0, x - bx0], ix[y - by0, x - bx0]
        assert np.array_equal(pano[y, :, x], np.asarray(cube_fn(t)[sy, :, sx])), \
            f"value mismatch at ({y},{x})"            # output must equal source exactly
    print(f"  OK: {n} sampled pixels bit-identical to source")


def pseudo_rgb(pano, b_rgb):
    chans = []
    for b in b_rgb:                                   # three pseudo-RGB bands
        band = pano[:, b, :].astype(np.float32)       # float for percentile math
        lo, hi = np.percentile(band[band > 0], (1, 99)) if (band > 0).any() else (0, 1)
        chans.append(np.clip((band - lo) / (hi - lo + 1e-12), 0, 1))
    return (np.stack(chans, axis=-1) * 255).astype(np.uint8)


def register_to_ortho(rgb, ortho_path):
    """Resolve the final orientation by matching the panorama to the 2000px ortho.

    Operates on a downscaled copy of the preview (orientation/overlay don't need
    full resolution; SIFT on a gigapixel panorama would be very slow). Builds the
    overlay at that scale too. Returns (rot_k, inliers, residual_angle,
    overlay_bgr | None, ortho_rms | None, H_pano_to_ortho | None) where the
    homography maps full-resolution post-rotation panorama pixels to ortho
    pixels at ortho long side 2000 px."""
    ortho = ortho_mod.load_ortho(ortho_path, 2000)
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    long_side = max(gray_full.shape[:2])
    if long_side > 2000:                              # downscale to ortho working scale
        s = 2000 / long_side
        gray = cv2.resize(gray_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
        gray = gray_full
    sift = make_sift(nfeatures=0)                     # ortho/orientation: full keypoints
    ko, do = sift.detectAndCompute(ortho, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    best = (361.0, -1, 0, None)                       # (residual angle, inliers, k, extras)
    for k in range(4):                                # try all four 90-degree rotations
        gg = np.ascontiguousarray(np.rot90(gray, k))
        kp, dp = sift.detectAndCompute(gg, None)
        if dp is None:
            continue
        good = [m for m, n in bf.knnMatch(dp, do, k=2) if m.distance < RATIO * n.distance]
        if len(good) < 8:
            continue
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([ko[m.trainIdx].pt for m in good])
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if Hm is None:
            continue
        ninl = int(mask.sum())
        angle = abs(np.degrees(np.arctan2(Hm[1, 0], Hm[0, 0])))  # leftover rotation
        print(f"  rot k={k}: {ninl} RANSAC inliers, residual rotation {angle:.1f} deg")
        if ninl >= MIN_INLIERS and angle < best[0]:
            inl = mask.ravel().astype(bool)
            best = (angle, ninl, k, (Hm, src[inl], dst[inl], gg))
    angle, ninl, k, extra = best
    if k < 0:
        raise RuntimeError("could not register panorama to orthoimage")

    overlay, ortho_rms, H_po = None, None, None
    if extra is not None:                             # 50% blend + residual, at 2000px scale
        Hm, src, dst, gg = extra
        warped = cv2.warpPerspective(gg, Hm, (ortho.shape[1], ortho.shape[0]))
        overlay = cv2.addWeighted(cv2.cvtColor(ortho, cv2.COLOR_GRAY2BGR), 0.5,
                                  cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), 0.5, 0)
        d = cv2.perspectiveTransform(src[None], Hm)[0] - dst
        ortho_rms = float(np.sqrt((d ** 2).mean()))
        # Hm maps rotated+downscaled preview px -> ortho px; the shipped panorama
        # is rotated by the same k, so full-res pano px -> ortho is Hm . diag(s)
        H_po = Hm @ np.diag([s, s, 1.0])
    return k, ninl, angle, overlay, ortho_rms, H_po


# ----------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------
def _label_strip(text, w, h=22):
    """A dark caption strip with white text, w px wide."""
    img = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(img, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def side_by_side(rgb, seam_bgr):
    """RGB preview and colorized seam map at equal height, captioned, hstacked.

    Inputs are BGR (OpenCV order); returns BGR for cv2.imwrite."""
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h = min(rgb_bgr.shape[0], seam_bgr.shape[0])      # match heights
    def fit(img):
        s = h / img.shape[0]
        return cv2.resize(img, (int(round(img.shape[1] * s)), h),
                          interpolation=cv2.INTER_AREA)
    a, b = fit(rgb_bgr), fit(seam_bgr)
    left = np.vstack([_label_strip("pseudo-RGB", a.shape[1]), a])
    right = np.vstack([_label_strip("seam map", b.shape[1]), b])
    gap = np.full((left.shape[0], 8, 3), 30, np.uint8)  # thin separator
    return np.hstack([left, gap, right])


def seam_palette(n):
    """Distinct color per tile; index 0 is background (black)."""
    return np.array([[0, 0, 0]] + [[(37 * (t + 3)) % 200 + 55,
                                    (91 * (t + 7)) % 200 + 55,
                                    (53 * (t + 11)) % 200 + 55]
                                   for t in range(n)], np.uint8)


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def stitch(tiles_dir, ortho_path, out_dir, artwork_name, progress_cb=None, params=None):
    """Run the full V3 ortho-anchored pipeline. Writes outputs to `out_dir`.

    Accepts any modality folder (ENVI BIL, or PNG/JPG/TIFF/NEF image tiles);
    the loader abstracts tile reading and mosaic writing.
    progress_cb(stage:int, total:int, msg:str) is called at each stage (1..6).
    params: optional {name: value} overrides for the tunable thresholds
            (see TUNABLE_PARAMS); missing keys keep their defaults.
    Returns the placement report dict."""
    apply_params(get_default_params())              # reset any previous overrides
    apply_params(params)                            # then apply this run's overrides

    import modality

    clk = {"start": time.time(), "last": time.time()}

    def step(i, msg):
        now = time.time()
        if i > 1:
            print(f"  [stage {i - 1} took {now - clk['last']:.1f}s]")
        clk["last"] = now
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
    if len(comps) > 1:                               # cannot place tiles in one frame
        groups = "; ".join("{" + ", ".join(tiles[t] for t in c) + "}" for c in comps)
        raise RuntimeError(
            f"tile match graph is disconnected into {len(comps)} groups: {groups}. "
            "Tiles in different groups share too little overlapping texture to "
            "register together (common when large regions are dark/low-texture). "
            "Ensure the tiles actually overlap, or lower SIFT_CONTRAST further.")

    step(3, "global placement (ortho-anchored)")
    ref, Hs = spanning_tree_init(pairs, len(tiles), tiles)       # uses full inlier counts
    print(f"  reference tile: {tiles[ref]}")
    rng = np.random.default_rng(0)
    pairs = {ij: cap_corr(pi, pj, rng=rng) for ij, (pi, pj) in pairs.items()}  # bound BA cost
    k = np.zeros(2)
    Hs, k = joint_refine(pairs, Hs, {ref}, k, g)                 # selfcal
    print("  matching tiles to ortho for anchoring...")
    anchors, Hs, _ = match_anchors(feats, Hs, k, g, tiles, ortho_path)
    anchors = {t: cap_corr(s, d, rng=rng) for t, (s, d) in anchors.items()}
    anchors_used = sorted(tiles[t] for t in anchors)
    Hs, k = joint_refine(pairs, Hs, set(), k, g, anchors=anchors, f_scale=3.0)
    Hs = normalize_scale(Hs, k, g, (h, w))
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
    rgb = loader.preview(pano)                       # modality-aware (stretch only for HSI)
    rot_k, ninl, angle, overlay, ortho_rms, H_po = register_to_ortho(rgb, ortho_path)
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
    seam_bgr = palette[label_r + 1]                              # +1 so -1 -> black
    cv2.imwrite(str(out_dir / "seam_map.png"), seam_bgr)
    cv2.imwrite(str(out_dir / f"{name}_rgb_vs_seam.png"), side_by_side(rgb, seam_bgr))

    if overlay is not None:                                      # ortho alignment check
        cv2.imwrite(str(out_dir / "ortho_overlay.png"), overlay)
        print(f"  ortho residual RMS: {ortho_rms:.2f} px (at ortho scale 2000px)")

    report = {
        "artwork": name,
        "mode": "ortho",
        "tiles": tiles,
        "modality_bands": int(n_bands),
        "reference_tile": tiles[ref],
        "distortion_k1_k2": [float(k[0]), float(k[1])],
        "anchored_tiles": anchors_used,
        "rotation_k": int(rot_k),
        "canvas_before_rotation": list(map(int, canvas[::-1])),
        "global_rms_px": float(rms),
        "radial_trend_px": trend,
        "ortho_overlay_rms_px": ortho_rms,
        # full-res post-rotation panorama px -> ortho px (ortho long side 2000).
        # Lets two modalities of the same artwork be co-registered without SIFT:
        # H_AB = inv(H_B_ortho) @ H_A_ortho
        "pano_to_ortho_H": None if H_po is None else H_po.tolist(),
        "hole_pixels": holes,
        "pairs": per_pair,
        "homographies_T_H": {tiles[t]: (T @ Hs[t]).tolist() for t in range(len(tiles))},
    }
    (out_dir / "placement.json").write_text(json.dumps(report, indent=2))
    print(f"  [stage 6 took {time.time() - clk['last']:.1f}s]")
    print(f"done in {time.time() - clk['start']:.1f}s. outputs in {out_dir}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Lossless ortho-anchored stitching")
    ap.add_argument("--tiles", "--bil", dest="tiles", required=True,
                    help="folder of tiles (.bil/.png/.jpg/.tif/.nef)")
    ap.add_argument("--ortho", required=True, help="ortho reference image (tif/png/jpg)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", required=True, help="artwork name (output filename prefix)")
    args = ap.parse_args()
    stitch(args.tiles, args.ortho, args.out, args.name)


if __name__ == "__main__":
    sys.exit(main())
