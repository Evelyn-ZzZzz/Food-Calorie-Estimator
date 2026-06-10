"""
Simulated AI Glasses — Gradio UI (stereo depth + YOLOv8-seg + kcal + DeepSeek + TTS).

Run:
  cd "blend ED/AI glass"
  python app.py
  → open http://127.0.0.1:7860
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import (
    DEFAULT_BASELINE_MM,
    DEFAULT_FX_PX,
    DEMO_SAMPLES,
    MIDDLEBURY_BASELINE_MM,
    MIDDLEBURY_FX_PX,
    NUTRITION5K_TEST,
    STEREO_SAMPLES,
)
from core.llm_narrator import _load_env_file, llm_status_markdown
from core.pipeline import run_analysis
from core.tts_speaker import speak_english
from core.volume_nutrition import suggest_fx_px

_load_env_file()

HOLO_CSS = """
<style>
/* Simulated glasses frame around stereo camera inputs */
.glasses-frame-row {
  justify-content: center !important;
  align-items: flex-end !important;
  gap: 0 !important;
  padding: 28px 20px 20px !important;
  margin: 8px 0 16px !important;
  background: linear-gradient(180deg, #3d3d3d 0%, #1e1e1e 55%, #2a2a2a 100%) !important;
  border: 5px solid #5a5a5a !important;
  border-radius: 90px 90px 36px 36px !important;
  box-shadow: 0 12px 40px rgba(0,0,0,0.45), inset 0 2px 0 rgba(255,255,255,0.08) !important;
  position: relative !important;
}
.glasses-frame-row::after {
  content: "AI GLASSES — STEREO VIEW";
  position: absolute;
  top: 6px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  letter-spacing: 0.2em;
  color: #00e5ff;
  font-weight: 700;
}
.glasses-bridge {
  flex: 0 0 36px !important;
  min-width: 36px !important;
  align-self: center !important;
  margin-bottom: 40px !important;
  height: 12px !important;
  background: #444 !important;
  border-radius: 4px !important;
  box-shadow: inset 0 1px 2px #000 !important;
}
.glasses-lens-wrap {
  flex: 1 1 45% !important;
  max-width: 48% !important;
  padding: 8px !important;
  background: radial-gradient(ellipse at center, #0a1520 0%, #000 70%) !important;
  border: 4px solid #00bcd4 !important;
  border-radius: 50% !important;
  box-shadow: 0 0 24px rgba(0,188,212,0.35), inset 0 0 30px rgba(0,100,120,0.25) !important;
  overflow: hidden !important;
}
.glasses-lens-wrap .wrap, .glasses-lens-wrap > div {
  border-radius: 50% !important;
  overflow: hidden !important;
}
.holo-panel {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: linear-gradient(135deg, rgba(0,20,40,0.92), rgba(0,60,80,0.85));
  border: 1px solid rgba(0,255,200,0.45);
  border-radius: 16px;
  padding: 20px 24px;
  color: #ffffff;
}
.holo-header { font-size: 1.35rem; color: #00ffd0; margin-bottom: 12px; }
.holo-total { font-size: 1.5rem; color: #fff; margin-bottom: 16px; }
.holo-total b { color: #ffeb3b; font-size: 1.8rem; }
.holo-card {
  border-left: 3px solid #00e5ff;
  padding: 10px 14px;
  margin: 10px 0;
  background: rgba(255,255,255,0.06);
  border-radius: 8px;
}
.holo-title { font-size: 1.15rem; font-weight: 600; color: #ffffff; text-transform: capitalize; }
.holo-row { margin: 6px 0; font-size: 1rem; display: flex; justify-content: space-between; }
.holo-label { color: #ffffff !important; font-weight: 600; }
.holo-value { color: #b2ff59; font-weight: 600; }
.holo-sub { font-size: 0.85rem; color: #b0bec5; margin-top: 6px; }
.holo-llm { margin-top: 18px; padding-top: 14px; border-top: 1px dashed rgba(0,255,200,0.35); color: #e3f2fd; }
.param-current {
  font-size: 1.15rem;
  font-weight: 700;
  color: #00838f;
  margin: 0 0 6px 0;
  padding: 8px 12px;
  background: #e0f7fa;
  border-radius: 8px;
  border-left: 4px solid #00acc1;
}
.formula-panel {
  font-size: 0.95rem;
  line-height: 1.65;
  padding: 16px 20px;
  margin: 12px 0 16px;
  background: #f5f9fc;
  border: 1px solid #b2ebf2;
  border-radius: 12px;
}
.formula-panel h3 { color: #006064; margin: 0 0 10px 0; font-size: 1.05rem; }
.formula-panel code { background: #e0f2f1; padding: 2px 6px; border-radius: 4px; }
.formula-panel .eq { font-family: Consolas, monospace; font-size: 1.05rem; color: #004d40; margin: 8px 0; }
</style>
"""


FORMULA_MD = """
<div class="formula-panel">
<h3>Key formulas (for presentation)</h3>

<p><b>1. Stereo depth</b> — convert disparity to distance (same as <code>stereo_depth.py</code>):</p>
<div class="eq">Z = (f<sub>x</sub> · B) / d</div>
<ul>
<li><b>Z</b> — depth to the scene point (meters)</li>
<li><b>f<sub>x</sub></b> — focal length in pixels (horizontal)</li>
<li><b>B</b> — stereo baseline (meters); UI shows mm → divide by 1000</li>
<li><b>d</b> — disparity from left/right match (pixels)</li>
</ul>

<p><b>2. Estimate f<sub>x</sub> from phone EXIF</b> (35 mm full-frame equivalent, main camera ≈ 24 mm):</p>
<div class="eq">f<sub>x</sub> ≈ (f<sub>eq</sub> / 36) × W</div>
<ul>
<li><b>f<sub>eq</sub></b> — 35 mm equivalent focal length (e.g. 24 mm for iPhone 1×)</li>
<li><b>W</b> — image width in pixels (portrait: the smaller side, e.g. 4278)</li>
<li>Example: 24 mm, W = 4278 → f<sub>x</sub> ≈ <b>2852 px</b></li>
</ul>

<p><b>3. Pixel → real size</b> (pinhole model, used for volume):</p>
<div class="eq">real_size ≈ pixel_extent × Z / f<sub>x</sub></div>
<div class="eq">pixel footprint area ≈ (Z / f<sub>x</sub>)²</div>
</div>
"""


def _rgb_from_bgr(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _clamp_b(v) -> float:
    if v is None:
        return float(DEFAULT_BASELINE_MM)
    return float(max(10.0, min(500.0, float(v))))


def _clamp_fx(v) -> float:
    if v is None:
        return float(DEFAULT_FX_PX)
    return float(max(50.0, min(10000.0, float(v))))


def load_bundled_pair(stem: str, baseline_mm: float = 60.0):
    """Load samples/demo/{stem}_left.jpg and {stem}_right.jpg (iPhone stereo)."""
    left_p = DEMO_SAMPLES / f"{stem}_left.jpg"
    right_p = DEMO_SAMPLES / f"{stem}_right.jpg"
    L = cv2.imread(str(left_p))
    R = cv2.imread(str(right_p))
    if L is None or R is None:
        return None, None, DEFAULT_FX_PX, baseline_mm
    w = L.shape[1]
    return _rgb_from_bgr(L), _rgb_from_bgr(R), float(suggest_fx_px(w)), baseline_mm


def load_demo_pair(shift_px: int = 12):
    """Nutrition5k test image + synthetic right shift, or bundled single-image shift."""
    imgs = []
    if NUTRITION5K_TEST.is_dir():
        imgs = list(NUTRITION5K_TEST.glob("*.jpg")) + list(NUTRITION5K_TEST.glob("*.png"))
    if not imgs:
        imgs = list(DEMO_SAMPLES.glob("*_left.jpg"))
    if not imgs:
        z = np.zeros((480, 640, 3), dtype=np.uint8)
        return z, z, DEFAULT_FX_PX
    path = random.choice(imgs)
    if path.name.endswith("_left.jpg"):
        stem = path.stem.replace("_left", "")
        L, R, fx, _ = load_bundled_pair(stem)
        if L is not None:
            return L, R, fx
    bgr = cv2.imread(str(path))
    if bgr is None:
        z = np.zeros((480, 640, 3), dtype=np.uint8)
        return z, z, DEFAULT_FX_PX
    h, w = bgr.shape[:2]
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    right = cv2.warpAffine(bgr, M, (w, h))
    fx_hint = float(suggest_fx_px(w))
    return _rgb_from_bgr(bgr), _rgb_from_bgr(right), fx_hint


def load_middlebury_pair():
    """Load first im0/im1 under stereo_samples/middlebury if user downloaded them."""
    base = STEREO_SAMPLES / "middlebury"
    for scene in sorted(base.iterdir()) if base.is_dir() else []:
        left_p = scene / "im0.png"
        right_p = scene / "im1.png"
        if left_p.is_file() and right_p.is_file():
            L = cv2.imread(str(left_p))
            R = cv2.imread(str(right_p))
            if L is not None and R is not None:
                return (
                    _rgb_from_bgr(L),
                    _rgb_from_bgr(R),
                    MIDDLEBURY_FX_PX,
                    MIDDLEBURY_BASELINE_MM,
                )
    return None, None, MIDDLEBURY_FX_PX, MIDDLEBURY_BASELINE_MM


def auto_fx_from_left(left_rgb):
    if left_rgb is None:
        return DEFAULT_FX_PX
    w = left_rgb.shape[1]
    return float(suggest_fx_px(w))


def analyze(
    left_rgb,
    right_rgb,
    baseline_mm,
    fx_px,
    enable_llm,
    enable_tts,
):
    empty_holo = HOLO_CSS + '<div class="holo-panel">Upload left & right images first.</div>'
    if left_rgb is None or right_rgb is None:
        return (
            None,
            None,
            None,
            empty_holo,
            "",
            "",
            "",
            "",
            HOLO_CSS + "<div class='status-panel'>Missing images.</div>",
        )
    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
    out = run_analysis(
        left_bgr,
        right_bgr,
        baseline_mm=_clamp_b(baseline_mm),
        fx_px=_clamp_fx(fx_px),
        enable_llm=enable_llm,
        speak=enable_tts,
    )
    speech = ""
    if enable_tts and out.total_kcal > 0:
        speech = (
            f"The estimated calories of this food are {int(round(out.total_kcal))} kilocalories."
        )
    return (
        _rgb_from_bgr(out.left_vis),
        _rgb_from_bgr(out.depth_vis),
        _rgb_from_bgr(out.right_vis),
        HOLO_CSS + out.hologram_html,
        out.latency_md,
        out.power_md,
        out.llm_text,
        speech,
        HOLO_CSS + out.status_html,
    )


def build_ui():
    with gr.Blocks(title="AI Glasses — Nutrition Prototype") as demo:
        gr.Markdown(
            """
            # Simulated AI Glasses · Nutrition Analysis
            **Left / right cameras** (fixed baseline **B**) → stereo depth → **YOLOv8-seg** → volume / mass / kcal → AR panel + optional voice.
            """
        )
        llm_status = gr.Markdown(llm_status_markdown())
        gr.Markdown(FORMULA_MD)
        with gr.Row(elem_classes=["glasses-frame-row"]):
            with gr.Column(elem_classes=["glasses-lens-wrap"]):
                left_in = gr.Image(label="Left eye", type="numpy", height=320)
            with gr.Column(elem_classes=["glasses-bridge"], scale=0, min_width=40):
                gr.HTML("<!-- bridge -->", show_label=False, container=False)
            with gr.Column(elem_classes=["glasses-lens-wrap"]):
                right_in = gr.Image(label="Right eye", type="numpy", height=320)
        with gr.Row():
            btn_chips = gr.Button("Load chips demo (bundled)", variant="secondary")
            btn_bread = gr.Button("Load bread demo (bundled)", variant="secondary")
            btn_demo = gr.Button("Load random demo", variant="secondary")
            btn_mb = gr.Button("Load Middlebury (optional)", variant="secondary")
            btn_go = gr.Button("Run analysis", variant="primary", scale=2)
        with gr.Row():
            with gr.Column():
                b_current = gr.Markdown(
                    f'<div class="param-current">Current baseline B: <b>{DEFAULT_BASELINE_MM:.1f}</b> mm</div>'
                )
                baseline_mm = gr.Number(
                    value=DEFAULT_BASELINE_MM,
                    label="Baseline B (mm) — type value, press Enter",
                    precision=1,
                )
                baseline_slider = gr.Slider(
                    minimum=40,
                    maximum=150,
                    value=DEFAULT_BASELINE_MM,
                    step=0.1,
                    label="B slider (drag)",
                    interactive=True,
                )
            with gr.Column():
                fx_current = gr.Markdown(
                    f'<div class="param-current">Current focal fx: <b>{DEFAULT_FX_PX:.0f}</b> px</div>'
                )
                fx_px = gr.Number(
                    value=DEFAULT_FX_PX,
                    label="Focal length fx (px) — type value, press Enter",
                    precision=0,
                )
                fx_slider = gr.Slider(
                    minimum=300,
                    maximum=4000,
                    value=DEFAULT_FX_PX,
                    step=1,
                    label="fx slider (drag)",
                    interactive=True,
                )

        def _fmt_b(v):
            return f'<div class="param-current">Current baseline B: <b>{float(v):.1f}</b> mm</div>'

        def _fmt_fx(v):
            return f'<div class="param-current">Current focal fx: <b>{float(v):.0f}</b> px</div>'

        def _b_from_slider(v):
            v = _clamp_b(v)
            return v, _fmt_b(v)

        def _fx_from_slider(v):
            v = _clamp_fx(v)
            return v, _fmt_fx(v)

        def _b_from_number(v):
            v = _clamp_b(v)
            return _fmt_b(v)

        def _fx_from_number(v):
            v = _clamp_fx(v)
            return _fmt_fx(v)

        baseline_slider.release(_b_from_slider, baseline_slider, [baseline_mm, b_current])
        fx_slider.release(_fx_from_slider, fx_slider, [fx_px, fx_current])
        baseline_mm.submit(_b_from_number, baseline_mm, b_current)
        fx_px.submit(_fx_from_number, fx_px, fx_current)
        with gr.Row():
            btn_auto_fx = gr.Button("Auto fx from left image (24mm equiv.)")
            chk_llm = gr.Checkbox(value=True, label="Enable DeepSeek (reads .env automatically)")
            chk_tts = gr.Checkbox(
                value=False,
                label="Voice (laptop speaker / Realtek as Windows default)",
            )
            btn_test_voice = gr.Button("Test speaker", variant="secondary")
        status_out = gr.HTML(label="Diagnostics")
        with gr.Row():
            left_out = gr.Image(label="Detection (left)")
            depth_out = gr.Image(label="Depth map")
            right_out = gr.Image(label="Right view")
        holo = gr.HTML(label="AR nutrition overlay")
        with gr.Row():
            lat_md = gr.Markdown(label="Latency")
            pwr_md = gr.Markdown(label="Power")
        llm_out = gr.Textbox(label="DeepSeek reply (meal + calories)", lines=4)
        tts_out = gr.Textbox(label="TTS script", lines=1)

        def on_demo():
            L, R, fx = load_demo_pair()
            fx = _clamp_fx(fx)
            return (
                L,
                R,
                fx,
                fx,
                _fmt_fx(fx),
            )

        def _apply_pair(L, R, fx, b):
            fx, b = _clamp_fx(fx), _clamp_b(b)
            return L, R, fx, b, fx, b, _fmt_fx(fx), _fmt_b(b)

        btn_chips.click(
            lambda: _apply_pair(*load_bundled_pair("chips", 60.0)),
            outputs=[
                left_in, right_in, fx_px, baseline_mm, fx_slider, baseline_slider,
                fx_current, b_current,
            ],
        )
        btn_bread.click(
            lambda: _apply_pair(*load_bundled_pair("bread", 60.0)),
            outputs=[
                left_in, right_in, fx_px, baseline_mm, fx_slider, baseline_slider,
                fx_current, b_current,
            ],
        )

        btn_demo.click(
            on_demo,
            outputs=[left_in, right_in, fx_px, fx_slider, fx_current],
        )

        def on_middlebury():
            L, R, fx, b = load_middlebury_pair()
            fx, b = _clamp_fx(fx), _clamp_b(b)
            if L is None:
                return (
                    None,
                    None,
                    fx,
                    b,
                    fx,
                    b,
                    _fmt_fx(fx),
                    _fmt_b(b),
                )
            return L, R, fx, b, fx, b, _fmt_fx(fx), _fmt_b(b)

        btn_mb.click(
            on_middlebury,
            outputs=[
                left_in,
                right_in,
                fx_px,
                baseline_mm,
                fx_slider,
                baseline_slider,
                fx_current,
                b_current,
            ],
        )

        def on_auto_fx(left_rgb):
            fx = _clamp_fx(auto_fx_from_left(left_rgb))
            return fx, fx, _fmt_fx(fx)

        btn_auto_fx.click(on_auto_fx, inputs=[left_in], outputs=[fx_px, fx_slider, fx_current])

        btn_go.click(
            analyze,
            inputs=[left_in, right_in, baseline_mm, fx_px, chk_llm, chk_tts],
            outputs=[
                left_out,
                depth_out,
                right_out,
                holo,
                lat_md,
                pwr_md,
                llm_out,
                tts_out,
                status_out,
            ],
            queue=True,
            concurrency_limit=1,
        )

        def test_voice():
            return speak_english(
                "The estimated calories of this food are five hundred twenty kilocalories."
            )

        btn_test_voice.click(test_voice, outputs=[tts_out], queue=False)

    return demo


if __name__ == "__main__":
    app = build_ui()
    print("=" * 60)
    print("AI Glass — UI: http://127.0.0.1:7860")
    print("Project folder:", ROOT)
    print("TTS: set Realtek (R) Audio as Windows default output device.")
    print("=" * 60)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
        css=HOLO_CSS,
    )
