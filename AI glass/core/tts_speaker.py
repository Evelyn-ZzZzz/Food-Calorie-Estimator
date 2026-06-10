"""Speak through the Windows default playback device (set Realtek as default in Sound settings)."""
from __future__ import annotations

import sys


def speak_english(text: str, rate: int = 165, volume: float = 1.0) -> str:
    """
    Uses pyttsx3 (Windows SAPI5) -> system default audio output.
    To use Realtek: Windows Settings > Sound > Output > choose Realtek (R) Audio.
    """
    text = (text or "").strip()
    if not text:
        return "Nothing to speak."
    try:
        import pyttsx3
    except ImportError as e:
        return f"TTS failed: install pyttsx3 ({e})"

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        # Prefer an English voice if available
        voices = engine.getProperty("voices")
        for v in voices:
            name = (getattr(v, "name", "") or "").lower()
            vid = (getattr(v, "id", "") or "").lower()
            if "english" in name or "en-us" in name or "zira" in name or "david" in vid:
                engine.setProperty("voice", v.id)
                break
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        dev_hint = "default Windows playback device"
        if sys.platform == "win32":
            dev_hint = "default playback (set Realtek (R) Audio as default in Windows Sound)"
        return f"Spoken on {dev_hint}."
    except Exception as e:
        return f"TTS error: {e}"
