"""
Meal narration via DeepSeek (OpenAI-compatible) / Ollama / template.
Loads secrets from project root .env (gitignored).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_ENV_LOADED = False


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    _ENV_LOADED = True


def _build_prompt(items: list[dict], total_kcal: float) -> tuple[str, str]:
    parts = []
    for it in items[:8]:
        parts.append(
            f"- {it.get('class_name', 'unknown')}: "
            f"confidence {it.get('confidence', 0):.0%}, "
            f"estimated volume ~{it.get('volume_cm3', 0):.0f} cm³, "
            f"mass ~{it.get('mass_g', 0):.0f} g, "
            f"~{it.get('kcal', 0):.0f} kcal"
        )
    summary = "\n".join(parts) if parts else "No food items detected."

    prompt = (
        "The user captured a meal with a stereo AI-glasses prototype. "
        "Our pipeline ran object detection (YOLOv8-seg), stereo depth, volume integration, "
        "and calorie estimation from a food density table.\n\n"
        "Write a natural **English** voice-over script (as if spoken through smart glasses), "
        "3 to 5 sentences, friendly and informative. Include:\n"
        "1) What foods were recognized and a brief comment on each major item.\n"
        "2) The estimated total calories and whether that seems low, moderate, or high for one meal.\n"
        "3) One or two practical dietary suggestions (balance, portion, veggies, etc.).\n"
        "4) A short caveat that estimates depend on camera distance, stereo calibration, and assumptions.\n\n"
        "Do NOT use bullet lists in the final answer — write flowing prose only.\n\n"
        f"Detection results:\n{summary}\n\n"
        f"Estimated total calories: ~{total_kcal:.0f} kcal"
    )
    return summary, prompt


def _template(summary: str, total_kcal: float) -> str:
    if "No food" in summary:
        return (
            "I could not identify any food in this capture. "
            "Try a clearer top-down view with even lighting, and make sure both left and right "
            "stereo images are uploaded. Our model recognizes 113 food categories from the Nutrition5k dataset."
        )
    level = "moderate"
    if total_kcal > 1200:
        level = "quite high"
    elif total_kcal < 400:
        level = "relatively light"
    return (
        f"Based on our vision pipeline, I detected several items on your plate. "
        f"The estimated total energy is about {total_kcal:.0f} kilocalories, which is {level} for a single meal. "
        "Consider adding fresh vegetables or lean protein if you want a more balanced plate. "
        "Please note these numbers depend on stereo depth accuracy and portion assumptions. "
        "(Local template — connect DeepSeek in .env for a richer narration.)"
    )


def _try_deepseek(prompt: str) -> str | None:
    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
        model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
        client = OpenAI(api_key=api_key, base_url=base)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an articulate English-speaking nutrition assistant for smart glasses. "
                        "Always respond in English only. Use clear, conversational prose (3–5 sentences). "
                        "Be helpful, slightly warm, and precise about uncertainty in automated estimates."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=380,
            temperature=0.75,
        )
        text = resp.choices[0].message.content.strip()
        return text or None
    except Exception:
        return None


def _try_ollama(prompt: str) -> str | None:
    _load_env_file()
    if os.environ.get("DISABLE_OLLAMA", "").strip().lower() in ("1", "true", "yes"):
        return None
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message") or {}).get("content", "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def llm_status_markdown() -> str:
    _load_env_file()
    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    if key and "deepseek" in base.lower():
        masked = key[:7] + "..." + key[-4:] if len(key) > 12 else "(set)"
        return (
            f"**LLM:** DeepSeek · model `{model}` · key {masked} — "
            "outputs **English** narration (3–5 sentences) in **DeepSeek reply** after Run analysis."
        )
    if key:
        return f"**LLM:** API `{base or 'OpenAI-compatible'}` · model `{model}` (key set)."
    return "**LLM:** no API key in `.env` — English template only."


def narrate_meal(items: list[dict], total_kcal: float) -> str:
    _load_env_file()
    summary, prompt = _build_prompt(items, total_kcal)

    text = _try_deepseek(prompt)
    if text:
        return text

    text = _try_ollama(prompt)
    if text:
        return text

    return _template(summary, total_kcal)
