"""YOLOv8-seg 推理封装。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.config import CONF_THRESH, IMGSZ, WEIGHTS

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {WEIGHTS}\n"
                "Ensure models/best.pt is included when sharing this folder."
            )
        _model = YOLO(str(WEIGHTS))
    return _model


def draw_detections_outline(bgr: np.ndarray, result, show_mask_fill: bool = False) -> np.ndarray:
    out = bgr.copy()
    if result.boxes is None:
        return out
    names = result.names
    for box, conf, cls in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    ):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 180)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{names.get(int(cls), cls)} {float(conf):.2f}"
        cv2.putText(out, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    if show_mask_fill and result.masks is not None:
        for mask_tensor in result.masks.data:
            m = mask_tensor.cpu().numpy()
            if m.shape[:2] != out.shape[:2]:
                m = cv2.resize(m, (out.shape[1], out.shape[0]))
            overlay = out.copy()
            overlay[m > 0.5] = (overlay[m > 0.5] * 0.5 + np.array([0, 200, 255]) * 0.5).astype(np.uint8)
            out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)
    return out


def predict_left(left_bgr: np.ndarray):
    device = 0 if torch.cuda.is_available() else "cpu"
    h, w = left_bgr.shape[:2]
    imgsz = min(1280, max(IMGSZ, max(h, w)))
    return get_model().predict(
        source=left_bgr,
        imgsz=imgsz,
        conf=CONF_THRESH,
        verbose=False,
        device=device,
    )[0]
