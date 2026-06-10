"""Mask + depth -> volume -> mass -> kcal (with sanity caps and fallbacks)."""
from __future__ import annotations

import numpy as np

from core.nutrition_tables import kcal_from_mass_g, lookup_food, mass_g_from_volume_cm3

# Per-item caps for tabletop food (cm³ / kcal)
MAX_VOLUME_CM3 = 2500.0
MAX_KCAL_PER_ITEM = 2500.0
DEFAULT_THICKNESS_M = 0.025  # bbox fallback ~2.5 cm


def pixel_extent_to_meters(pixel_extent: float, z_m: float, fx_px: float) -> float:
    """
    Pinhole model: physical size along image axis (meters).
    real_size ≈ pixel_size × Z / fx
    """
    if fx_px <= 0 or z_m <= 0:
        return 0.0
    return pixel_extent * z_m / fx_px


def pixel_area_m2(z_m: float, fx_px: float) -> float:
    """Footprint of one pixel on a plane at depth Z (m²). side = Z/fx → area = (Z/fx)²."""
    if fx_px <= 0 or z_m <= 0:
        return 0.0
    s = z_m / fx_px
    return s * s


def suggest_fx_px(image_width: int, equiv_35mm: float = 24.0) -> int:
    return int(round((equiv_35mm / 36.0) * image_width))


def mask_from_yolo_result(result) -> np.ndarray | None:
    if result.masks is None or len(result.masks) == 0:
        if result.boxes is None or len(result.boxes) == 0:
            return None
        h, w = result.orig_shape
        m = np.zeros((h, w), dtype=np.uint8)
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            m[y1:y2, x1:x2] = 1
        return m.astype(bool)
    h, w = result.orig_shape
    combined = np.zeros((h, w), dtype=np.float32)
    for mask_tensor in result.masks.data:
        m = mask_tensor.cpu().numpy()
        if m.shape != (h, w):
            import cv2

            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        combined = np.maximum(combined, m)
    return combined > 0.5


def _resize_z(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if z.shape == mask.shape:
        return z
    import cv2

    return cv2.resize(z, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)


def table_depth_m(z: np.ndarray, mask: np.ndarray) -> float:
    """Reference plane depth from background (outside mask)."""
    bg = (~mask) & np.isfinite(z)
    if bg.sum() > 200:
        return float(np.median(z[bg]))
    valid = np.isfinite(z)
    if valid.any():
        return float(np.percentile(z[valid], 80))
    return 0.45


def volume_cm3_from_mask_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    fx_px: float,
    stereo_ok: bool = True,
) -> tuple[float, str]:
    """
    Height integration: h = Z_table - Z_food (closer = smaller Z).
    Returns (volume_cm3, mode) where mode is 'stereo' or 'bbox_fallback'.
    """
    if mask is None or not mask.any() or fx_px <= 0:
        return 0.0, "none"

    z = _resize_z(depth_m, mask)
    valid = mask & np.isfinite(z)
    if not valid.any() or not stereo_ok:
        return 0.0, "bbox_fallback"

    z_table = table_depth_m(z, mask)
    z_pixels = z[valid]
    heights_m = np.maximum(0.0, z_table - z_pixels)
    # Drop outliers (bad stereo spikes)
    if heights_m.size > 20:
        cap = float(np.percentile(heights_m, 95))
        heights_m = np.minimum(heights_m, cap)

    # Volume = Σ h_i × area_i,  area_i = (Z_i/fx)²  (real ≈ pixel × Z/fx per axis)
    areas_m2 = (z_pixels / fx_px) ** 2
    volume_m3 = float(np.sum(heights_m * areas_m2))
    vol_cm3 = min(volume_m3 * 1e6, MAX_VOLUME_CM3)
    if vol_cm3 < 0.5:
        return 0.0, "bbox_fallback"
    return vol_cm3, "stereo"


def volume_cm3_bbox_fallback(
    xyxy: np.ndarray,
    fx_px: float,
    depth_m: np.ndarray | None = None,
    thickness_m: float = DEFAULT_THICKNESS_M,
) -> float:
    x1, y1, x2, y2 = xyxy
    w_px = max(1, int(x2 - x1))
    h_px = max(1, int(y2 - y1))
    z_est = 0.4
    if depth_m is not None and np.isfinite(depth_m).any():
        z_est = float(np.nanmedian(depth_m[np.isfinite(depth_m)]))
    # footprint: (w_px × Z/fx) × (h_px × Z/fx)
    w_m = pixel_extent_to_meters(w_px, z_est, fx_px)
    h_m = pixel_extent_to_meters(h_px, z_est, fx_px)
    footprint_m2 = w_m * h_m
    vol_cm3 = footprint_m2 * thickness_m * 1e6
    return min(vol_cm3, MAX_VOLUME_CM3)


def nutrition_for_detection(class_name: str, volume_cm3: float, mode: str = "stereo") -> dict:
    info = lookup_food(class_name)
    mass_g = mass_g_from_volume_cm3(volume_cm3, info["density_g_cm3"])
    kcal = kcal_from_mass_g(mass_g, info["kcal_per_100g"])
    kcal = min(kcal, MAX_KCAL_PER_ITEM)
    return {
        "class_name": class_name,
        "volume_cm3": round(volume_cm3, 1),
        "mass_g": round(mass_g, 1),
        "kcal": round(kcal, 1),
        "density_g_cm3": info["density_g_cm3"],
        "kcal_per_100g": info["kcal_per_100g"],
        "cook": info["cook"],
        "volume_mode": mode,
    }
