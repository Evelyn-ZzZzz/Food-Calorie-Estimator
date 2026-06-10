"""Stereo disparity -> depth (SGBM demo; replace with calibrated rectification later)."""
from __future__ import annotations

import cv2
import numpy as np

# Table-top scene: clamp depth to avoid Z explosion when d is tiny
Z_MIN_M = 0.15
Z_MAX_M = 1.50


def compute_disparity_sgbm(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    num_disparities: int = 160,
) -> np.ndarray:
    g0 = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    if g0.shape != g1.shape:
        g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]))

    block = 7
    nd = max(16, (num_disparities // 16) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=block,
        P1=8 * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=80,
        speckleRange=16,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(g0, g1).astype(np.float32) / 16.0
    return disp


def stereo_quality_score(disparity: np.ndarray) -> float:
    """0..1 — fraction of pixels with plausible disparity."""
    valid = disparity > 1.0
    if not valid.any():
        return 0.0
    return float(valid.mean())


def disparity_to_depth_m(
    disparity: np.ndarray,
    fx_px: float,
    baseline_mm: float,
    min_disp: float = 1.0,
) -> np.ndarray:
    b_m = baseline_mm / 1000.0
    z = np.full(disparity.shape, np.nan, dtype=np.float32)
    ok = disparity >= min_disp
    z[ok] = (fx_px * b_m) / disparity[ok]
    z = np.clip(z, Z_MIN_M, Z_MAX_M)
    return z


def depth_postprocess(depth_m: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    """Median filter on valid depths only; do not fill invalid with global median."""
    z = depth_m.copy()
    valid = np.isfinite(z)
    if not valid.any():
        return z
    z8 = np.clip(z * 1000, 0, 65535).astype(np.uint16)
    z8[~valid] = 0
    z8 = cv2.medianBlur(z8, 5)
    out = z8.astype(np.float32) / 1000.0
    out[z8 == 0] = np.nan
    out = np.clip(out, Z_MIN_M, Z_MAX_M)
    # Where disparity was OK but depth nan, keep nan
    bad = disparity < 1.0
    out[bad] = np.nan
    return out


def depth_colormap_bgr(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m)
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    z = depth_m.copy()
    z[~valid] = np.nanmedian(z[valid])
    z_norm = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(255 - z_norm, cv2.COLORMAP_TURBO)
