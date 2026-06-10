"""Paths and defaults — all relative to project root (AI glass/)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Trained YOLOv8-seg weights (113 food classes, Nutrition5k)
WEIGHTS = ROOT / "models" / "best.pt"

# Bundled stereo demo images (chips, bread, etc.)
DEMO_SAMPLES = ROOT / "samples" / "demo"
# Optional: full Nutrition5k test set for "Load demo" (set env or edit path)
NUTRITION5K_TEST = Path(os.environ.get("NUTRITION5K_TEST", r"D:\Nutrition5k\test\images"))
STEREO_SAMPLES = ROOT / "samples" / "stereo"

DEFAULT_BASELINE_MM = 65.0
DEFAULT_FX_PX = 2852.0  # iPhone 1× portrait ~4278 px wide, 24 mm equiv.

IMGSZ = 640
CONF_THRESH = 0.12

MIDDLEBURY_BASELINE_MM = 111.53
MIDDLEBURY_FX_PX = 1758.23
