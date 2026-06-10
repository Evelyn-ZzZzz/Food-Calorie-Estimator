"""End-to-end: stereo + YOLO + volume/kcal + timing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import DEFAULT_BASELINE_MM, DEFAULT_FX_PX
from core.llm_narrator import narrate_meal
from core.metrics_report import LatencyMs, power_report_markdown
from core.stereo_depth import (
    compute_disparity_sgbm,
    depth_colormap_bgr,
    depth_postprocess,
    disparity_to_depth_m,
    stereo_quality_score,
)
from core.volume_nutrition import (
    nutrition_for_detection,
    suggest_fx_px,
    volume_cm3_bbox_fallback,
    volume_cm3_from_mask_depth,
)
from core.yolo_detect import draw_detections_outline, predict_left


@dataclass
class FoodItemResult:
    class_name: str
    confidence: float
    volume_cm3: float
    mass_g: float
    kcal: float
    cook: str
    volume_mode: str = "stereo"


@dataclass
class AnalysisResult:
    left_vis: np.ndarray
    right_vis: np.ndarray
    depth_vis: np.ndarray
    items: list[FoodItemResult] = field(default_factory=list)
    total_kcal: float = 0.0
    latency: LatencyMs = field(default_factory=LatencyMs)
    llm_text: str = ""
    latency_md: str = ""
    power_md: str = ""
    hologram_html: str = ""
    status_html: str = ""


def _hologram_html(items: list[FoodItemResult], total_kcal: float, llm_text: str) -> str:
    cards = ""
    for it in items:
        mode_note = f"vol. via {it.volume_mode}"
        cards += f"""
        <div class="holo-card">
          <div class="holo-title">{it.class_name}</div>
          <div class="holo-row"><span class="holo-label">Confidence</span><span class="holo-value">{it.confidence:.0%}</span></div>
          <div class="holo-row"><span class="holo-label">Volume</span><span class="holo-value">{it.volume_cm3:.1f} cm³</span></div>
          <div class="holo-row"><span class="holo-label">Mass</span><span class="holo-value">{it.mass_g:.1f} g</span></div>
          <div class="holo-row"><span class="holo-label">Calories</span><span class="holo-value">{it.kcal:.0f} kcal</span></div>
          <div class="holo-sub">{mode_note} · density ref: {it.cook}</div>
        </div>"""
    if not cards:
        cards = '<div class="holo-card"><div class="holo-title">No food detected</div></div>'
    return f"""
    <div class="holo-panel">
      <div class="holo-header">◈ AR Nutrition Overlay</div>
      <div class="holo-total">Total ≈ <b>{total_kcal:.0f}</b> kcal</div>
      {cards}
      <div class="holo-llm">{llm_text}</div>
    </div>
    """


def _status_html(
    stereo_q: float,
    fx_px: float,
    img_w: int,
    n_det: int,
    used_fallback: bool,
) -> str:
    fx_suggest = suggest_fx_px(img_w)
    lines = [
        "<div class='status-panel'>",
        f"<b>Stereo quality:</b> {stereo_q:.0%} valid disparity pixels",
    ]
    if stereo_q < 0.08:
        lines.append(
            " · <span style='color:#ff8a80'>Poor — check left/right alignment, B, and fx. "
            "Volume uses bbox fallback (approx.).</span>"
        )
    if abs(fx_px - fx_suggest) > fx_suggest * 0.35:
        lines.append(
            f"<br><b>fx hint:</b> for this image width ({img_w}px), try fx ≈ <b>{fx_suggest}</b> "
            f"(24mm-equiv). Current fx={fx_px:.0f} may skew depth."
        )
    if n_det == 0:
        lines.append(
            "<br><b>YOLO:</b> no boxes — try lower lighting, center the dish, or use Nutrition5k-like views. "
            "Class must be one of 113 training names (e.g. <i>fried chicken</i>)."
        )
    elif used_fallback:
        lines.append("<br><b>Volume:</b> some items used bbox thickness estimate (stereo unreliable).")
    lines.append("</div>")
    return "\n".join(lines)


def run_analysis(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    baseline_mm: float = DEFAULT_BASELINE_MM,
    fx_px: float = DEFAULT_FX_PX,
    enable_llm: bool = True,
    speak: bool = False,
) -> AnalysisResult:
    lat = LatencyMs()
    t0 = time.perf_counter()

    left = left_bgr.copy()
    right = right_bgr.copy()
    if right.shape != left.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    h, w = left.shape[:2]
    lat.capture_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    disp = compute_disparity_sgbm(left, right)
    stereo_q = stereo_quality_score(disp)
    stereo_ok = stereo_q >= 0.08
    depth = disparity_to_depth_m(disp, fx_px=fx_px, baseline_mm=baseline_mm)
    depth = depth_postprocess(depth, disp)
    depth_vis = depth_colormap_bgr(depth)
    lat.stereo_depth_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    res = predict_left(left)
    left_vis = draw_detections_outline(left, res, show_mask_fill=False)
    lat.yolo_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    items: list[FoodItemResult] = []
    names = res.names
    used_fallback = False

    if res.boxes is not None and len(res.boxes) > 0:
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            cname = names.get(cls_id, str(cls_id))
            xyxy = res.boxes.xyxy[i].cpu().numpy()

            if res.masks is not None and i < len(res.masks.data):
                mh, mw = res.orig_shape
                m = res.masks.data[i].cpu().numpy()
                if m.shape != (mh, mw):
                    m = cv2.resize(m, (mw, mh))
                mask = m > 0.5
            else:
                mask = np.zeros(res.orig_shape, dtype=bool)
                x1, y1, x2, y2 = map(int, xyxy)
                mask[y1:y2, x1:x2] = True

            vol, mode = volume_cm3_from_mask_depth(mask, depth, fx_px, stereo_ok=stereo_ok)
            if mode == "bbox_fallback" or vol <= 0:
                vol = volume_cm3_bbox_fallback(xyxy, fx_px, depth)
                mode = "bbox_fallback"
                used_fallback = True

            nut = nutrition_for_detection(cname, vol, mode)
            items.append(
                FoodItemResult(
                    class_name=cname,
                    confidence=conf,
                    volume_cm3=nut["volume_cm3"],
                    mass_g=nut["mass_g"],
                    kcal=nut["kcal"],
                    cook=nut["cook"],
                    volume_mode=nut["volume_mode"],
                )
            )

    total_kcal = sum(it.kcal for it in items)
    lat.volume_nutrition_ms = (time.perf_counter() - t3) * 1000

    t4 = time.perf_counter()
    llm_text = ""
    if enable_llm:
        llm_text = narrate_meal([it.__dict__ for it in items], total_kcal)
    lat.llm_ms = (time.perf_counter() - t4) * 1000

    t5 = time.perf_counter()
    if speak and total_kcal > 0:
        from core.tts_speaker import speak_english

        speak_english(
            f"The estimated calories of this food are {int(round(total_kcal))} kilocalories."
        )
    lat.tts_ms = (time.perf_counter() - t5) * 1000

    status = _status_html(stereo_q, fx_px, w, len(items), used_fallback)
    return AnalysisResult(
        left_vis=left_vis,
        right_vis=right,
        depth_vis=depth_vis,
        items=items,
        total_kcal=total_kcal,
        latency=lat,
        llm_text=llm_text,
        latency_md=lat.to_markdown(),
        power_md=power_report_markdown(lat),
        hologram_html=_hologram_html(items, total_kcal, llm_text),
        status_html=status,
    )
