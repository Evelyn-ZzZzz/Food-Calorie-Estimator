# AI Glass — Simulated Smart Glasses · Food Recognition & Calorie Estimation

Stereo cameras → depth → **YOLOv8-seg** (113 food classes) → volume / mass / kcal → Gradio web UI, with optional **DeepSeek** narration and **TTS** voice output.

---

## Quick start (for teammates)

### 1. Environment

- Windows 10/11, Python 3.10+ (Anaconda recommended)
- **NVIDIA GPU** speeds up inference; CPU-only also works (slower)

```bash
cd "C:\Users\Administrator\Desktop\blend ED\AI glass"
pip install -r requirements.txt
```

### 2. Launch the web UI

**Option A — double-click**

```
run.bat
```

**Option B — command line**

```bash
python app.py
```

### 3. Open in browser

**http://127.0.0.1:7860**

(Local machine only. If the page fails after a network change, run `run.bat` again.)

### 4. Recommended first demo

1. Click **Load chips demo (bundled)** or **Load bread demo (bundled)** (stereo pairs included)
2. Set **B ≈ 60 mm** (phone shifted ~6 cm between shots) and **fx ≈ 2852 px** (iPhone 1× portrait; or use **Auto fx**)
3. Check **Enable DeepSeek** (requires `.env`; see below)
4. Click **Run analysis**
5. Review: detection view, depth map, AR nutrition panel, latency/power table, DeepSeek reply

---

## Project layout

```
AI glass/
├── README.md              ← This guide
├── requirements.txt       ← Python dependencies
├── run.bat                ← One-click UI launcher (Windows)
├── app.py                 ← Main entry: Gradio web UI
├── .env.example           ← DeepSeek API template (copy to .env)
│
├── models/
│   └── best.pt            ← Required: pretrained YOLOv8-seg weights (~7 MB)
│
├── core/                  ← Analysis pipeline (used by UI; rarely edited)
│   ├── config.py          ← Paths and default B, fx
│   ├── pipeline.py        ← Full flow: depth + YOLO + volume + LLM + timing
│   ├── stereo_depth.py    ← SGBM stereo, Z = (fx·B)/d
│   ├── yolo_detect.py     ← YOLOv8-seg inference
│   ├── volume_nutrition.py← Mask/depth → volume → mass → kcal
│   ├── nutrition_tables.py← Food density & kcal per 100 g lookup
│   ├── llm_narrator.py    ← DeepSeek / Ollama / local English template
│   ├── tts_speaker.py     ← Windows speech (pyttsx3)
│   └── metrics_report.py  ← Per-stage latency & estimated power
│
└── samples/
    ├── demo/              ← Bundled stereo demos (chips, bread left/right)
    └── stereo/            ← Notes for optional Middlebury calibration data
```

---

## UI features ↔ code

| UI feature | Code |
|------------|------|
| Left/right upload & demo load | `app.py` |
| Depth & fx formulas (presentation) | `app.py` (`FORMULA_MD`) |
| Baseline **B**, focal **fx** | `app.py` + `core/config.py` |
| Depth map | `core/stereo_depth.py` |
| Food detection boxes | `core/yolo_detect.py` + `models/best.pt` |
| Volume / calories | `core/volume_nutrition.py` + `nutrition_tables.py` |
| AR nutrition overlay | `core/pipeline.py` |
| DeepSeek English narration | `core/llm_narrator.py` + `.env` |
| TTS voice | `core/tts_speaker.py` (Windows default playback device) |
| Latency / power table | `core/metrics_report.py` |

---

## DeepSeek setup (optional)

1. Copy `.env.example` to `.env`
2. Add your API key (never commit `.env`):

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
DISABLE_OLLAMA=1
```

Without `.env`, the UI still runs; the LLM area uses a short **English local template**.

---

## Key parameters

| Parameter | Meaning | Typical value |
|-----------|---------|----------------|
| **B** | Distance between left/right cameras (mm) | iPhone shift along a line ≈ **60** mm |
| **fx** | Horizontal focal length (pixels) | iPhone 1× portrait: `fx ≈ (24/36) × image width`, e.g. W=4278 → **2852** |

Formulas are shown at the top of the UI. When typing **fx**, press **Enter** after the full number (avoids partial-input errors).

---

## Using your own stereo photos

1. Upload **left** and **right** images to Left eye / Right eye  
2. Set correct **B** and **fx** (or **Auto fx from left image**)  
3. Click **Run analysis**

The model was trained on **Nutrition5k** (113 classes, lab-style plate views). Casual phone photos may be misclassified (e.g. chips as bread) due to domain shift—not necessarily broken weights.

---

## FAQ

| Issue | Fix |
|-------|-----|
| Browser cannot open port 7860 | Close old `python app.py` processes; run `run.bat` again |
| Missing `best.pt` | Ensure `models/best.pt` was copied with the folder |
| Calories unrealistically high | Check **B** and **fx**; poor stereo triggers bbox fallback |
| No DeepSeek reply | Check `.env` and network; or uncheck Enable DeepSeek |
| No TTS audio | Windows Settings → Sound → set default output device |

---

## Sharing with your group

Zip the entire **`AI glass`** folder. Minimum contents:

- `app.py`, `core/`, `models/best.pt`, `samples/demo/`, `requirements.txt`, `README.md`, `run.bat`

Do **not** include `.env` (contains secrets). Each teammate adds their own DeepSeek key if needed.

---

## References

- Dataset: IS2AI Annotated Nutrition5k (113 classes)  
- Model: YOLOv8n-seg, weights in `models/best.pt`  
- UI URL: **http://127.0.0.1:7860**
