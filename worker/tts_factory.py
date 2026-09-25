"""阶段 23: TTS 工厂 — 按 TTS_PROVIDER 选实现 (cartesia | minimax | moss)。

设计目标：
- 单一入口 build_tts(provider, ...) -> (tts_instance, label)
- 各 provider 互不耦合，加新 provider 只改这里
- 失败时降级到 moss（CPU 兜底），保证不断流
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# 兼容单元测试：factory 被直接 import 时也读到 .env.local
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_name in (".env.local", ".env"):
    env_file = PROJECT_ROOT / env_name
    if env_file.exists():
        load_dotenv(env_file)
        break

from worker.moss_tts import MossHttpTTS


def _build_voxcpm() -> Tuple[object, str]:
    """VoxCPM2 声音克隆 TTS（RunPod 云 GPU + SSH 隧道）。"""
    from worker.voxcpm_tts import VoxCPMHttpTTS

    url = os.getenv("VOXCPM_URL", "http://localhost:8000").strip()
    voice = os.getenv("VOXCPM_VOICE", "fengge").strip()
    style = os.getenv("VOXCPM_STYLE", "").strip()
    sample_rate = int(os.getenv("VOXCPM_SAMPLE_RATE", "24000").strip())

    tts = VoxCPMHttpTTS(url=url, voice=voice, style=style, sample_rate=sample_rate)
    label = f"voxcpm:{url}/{voice}"
    return tts, label


def _build_cartesia() -> Tuple[object, str]:
    """Cartesia sonic-3 工厂。"""
    from livekit.plugins.cartesia import TTS as CartesiaTTS

    api_key = os.getenv("CARTESIA_API_KEY", "").strip()
    voice_id = os.getenv("CARTESIA_VOICE_ID", "").strip()
    model = os.getenv("CARTESIA_MODEL", "sonic-3").strip()
    language = os.getenv("CARTESIA_LANGUAGE", "zh").strip()
    _speed_raw = os.getenv("CARTESIA_SPEED", "").strip()
    speed: float | None = float(_speed_raw) if _speed_raw else None

    if not api_key:
        raise RuntimeError("CARTESIA_API_KEY not set in .env.local")
    if not voice_id:
        raise RuntimeError("CARTESIA_VOICE_ID not set in .env.local")

    tts = CartesiaTTS(
        api_key=api_key,
        model=model,
        voice=voice_id,
        language=language,
        sample_rate=24000,
        word_timestamps=False,  # 中文用 sonic 模型时不支持 word_timestamps，关闭避免 warning
        speed=speed,
    )
    label = f"cartesia:{model}/{voice_id[:8]}/{language}"
    return tts, label


def _build_minimax() -> Tuple[object, str]:
    """MiniMax speech-02 工厂。"""
    from worker.minimax_tts_plugin import MinimaxTTS  # 阶段 24

    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    voice_id = os.getenv("MINIMAX_VOICE_ID", "").strip()
    model = os.getenv("MINIMAX_MODEL", "speech-02-turbo").strip()
    sample_rate = int(os.getenv("MINIMAX_SAMPLE_RATE", "24000").strip())
    language_boost = os.getenv("MINIMAX_LANGUAGE_BOOST", "Chinese").strip()
    speed = float(os.getenv("MINIMAX_SPEED", "1.0").strip())
    vol = float(os.getenv("MINIMAX_VOL", "1.0").strip())
    pitch = int(os.getenv("MINIMAX_PITCH", "0").strip())

    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY not set in .env.local")
    if not voice_id:
        raise RuntimeError("MINIMAX_VOICE_ID not set in .env.local")

    tts = MinimaxTTS(
        api_key=api_key,
        voice_id=voice_id,
        model=model,
        sample_rate=sample_rate,
        language_boost=language_boost,
        speed=speed,
        vol=vol,
        pitch=pitch,
    )
    label = f"minimax:{model}/{voice_id[:12]}/{language_boost}/{sample_rate}Hz/speed={speed}/pitch={pitch}"
    return tts, label


def _build_moss(moss_url: str, moss_voice: str) -> Tuple[object, str]:
    """MOSS 本地 CPU TTS（兜底方案）。"""
    tts = MossHttpTTS(url=moss_url, voice=moss_voice)
    return tts, f"moss:{moss_voice}"


def build_tts(provider: str, moss_url: str, moss_voice: str) -> Tuple[object, str]:
    """按 provider 选 TTS；provider 失败自动降级到 moss。

    Returns:
        (tts_instance, label_for_log)
    """
    provider = (provider or "cartesia").strip().lower()
    started = time.time()

    # 1. 优先按 provider 选
    if provider == "voxcpm":
        try:
            tts, label = _build_voxcpm()
            print(f"[tts_factory] loaded {label} in {(time.time()-started)*1000:.0f}ms", flush=True)
            return tts, label
        except Exception as exc:
            print(f"[tts_factory] voxcpm init failed: {exc!r} - falling back to moss", flush=True)

    elif provider == "cartesia":
        try:
            tts, label = _build_cartesia()
            print(f"[tts_factory] loaded {label} in {(time.time()-started)*1000:.0f}ms", flush=True)
            return tts, label
        except Exception as exc:
            print(f"[tts_factory] cartesia init failed: {exc!r} - falling back to moss", flush=True)

    if provider == "minimax":
        try:
            tts, label = _build_minimax()
            print(f"[tts_factory] loaded {label} in {(time.time()-started)*1000:.0f}ms", flush=True)
            return tts, label
        except Exception as exc:
            print(f"[tts_factory] minimax init failed: {exc!r} - falling back to moss", flush=True)

    elif provider == "moss":
        tts, label = _build_moss(moss_url, moss_voice)
        print(f"[tts_factory] loaded {label} in {(time.time()-started)*1000:.0f}ms", flush=True)
        return tts, label

    else:
        print(f"[tts_factory] unknown provider={provider!r} - falling back to moss", flush=True)

    # 2. 兜底 MOSS
    tts, label = _build_moss(moss_url, moss_voice)
    print(f"[tts_factory] fallback to {label}", flush=True)
    return tts, label
