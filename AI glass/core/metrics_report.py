"""Latency breakdown and rough power estimate for course demo."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyMs:
    capture_ms: float = 0.0
    stereo_depth_ms: float = 0.0
    yolo_ms: float = 0.0
    volume_nutrition_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    ui_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.capture_ms
            + self.stereo_depth_ms
            + self.yolo_ms
            + self.volume_nutrition_ms
            + self.llm_ms
            + self.tts_ms
            + self.ui_ms
        )

    def to_markdown(self) -> str:
        rows = [
            ("Capture / sync", self.capture_ms),
            ("Stereo depth (SGBM)", self.stereo_depth_ms),
            ("YOLOv8-seg inference", self.yolo_ms),
            ("Volume / mass / kcal", self.volume_nutrition_ms),
            ("LLM narration (optional)", self.llm_ms),
            ("TTS speech", self.tts_ms),
            ("UI render", self.ui_ms),
            ("**Total**", self.total_ms),
        ]
        lines = ["| Stage | Time (ms) |", "|-------|-----------|"]
        for name, ms in rows:
            lines.append(f"| {name} | {ms:.1f} |")
        return "\n".join(lines)


POWER_W = {
    "stereo_cameras": 2.0,
    "gpu_cpu_compute": 8.0,
    "wireless": 0.5,
    "speaker": 1.0,
}


def power_report_markdown(latency: LatencyMs) -> str:
    t_s = max(latency.total_ms / 1000.0, 0.001)
    lines = [
        "### Power estimate (one analysis cycle)",
        "",
        "| Module | Assumed power (W) | Energy (J) |",
        "|--------|-------------------|------------|",
    ]
    total_j = 0.0
    for name, w in POWER_W.items():
        e = w * t_s
        total_j += e
        lines.append(f"| {name} | {w} | {e:.3f} |")
    lines.append(f"| **Total** | — | **{total_j:.3f}** |")
    lines.append("")
    lines.append(
        f"*Assumes modules active for ~{t_s*1000:.0f} ms; real glasses can duty-cycle to save power.*"
    )
    return "\n".join(lines)
