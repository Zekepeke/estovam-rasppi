"""
config.py — Single source of truth for all settings and environment variables.

Every tunable constant lives here. No module should hard-code values or call
os.getenv() directly — import from this file instead.
"""

import os
import logging
from dotenv import load_dotenv

# Load .env.local first, fall back to .env
load_dotenv(".env.local")

# =============================================================================
# API KEYS
# =============================================================================
# https://aistudio.google.com/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# https://elevenlabs.io/app/settings/api-keys
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# =============================================================================
# AUDIO
# =============================================================================
SAMPLE_RATE: int = 16_000  # Hz — standard for voice processing
CHANNELS: int = 1  # Mono
DTYPE: str = "int16"  # 16-bit PCM

# =============================================================================
# WAKE WORD  (openwakeword — fully local, no API key)
# =============================================================================
# Pre-trained model names: "hey_jarvis", "hey_mycroft", "alexa", "hey_rhasspy"
# Full list: https://github.com/dscripka/openwakeword#pre-trained-models
WAKE_WORD_MODEL: str = "hey_jarvis"
WAKE_WORD_THRESHOLD: float = 0.4  # 0–1; raise to reduce false positives
WAKE_WORD_DISPLAY: str = "Hey Jarvis"  # Human-readable label for logs/UI

# openwakeword requires exactly 80 ms chunks at 16 kHz
WAKE_WORD_CHUNK_SIZE: int = 1_280  # samples (16_000 Hz × 0.08 s)

# =============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# =============================================================================
SILENCE_THRESHOLD: int = 500  # RMS amplitude — tune for your environment
SILENCE_DURATION: float = 1.5  # Seconds of silence before stopping
MAX_RECORDING_DURATION: float = 30.0  # Hard ceiling (safety guard)
MIN_RECORDING_DURATION: float = 0.5  # Don't check silence until after this
RECORDING_CHUNK_MS: int = 100  # Audio chunk size in milliseconds

# =============================================================================
# RUBIK Pi 3  (AI coprocessor — Qualcomm QCS6490, Adreno 643 GPU)
# =============================================================================
RUBIKPI_HOST: str = "http://100.116.151.71"
RUBIKPI_LLM_PORT: int = 8080  # llama-server OpenAI-compat  (POST /v1/chat/completions)

# =============================================================================
# SPEECH-TO-TEXT  (faster-whisper running locally on Pi 5)
# =============================================================================
# Options: "tiny.en" (fastest) → "base.en" → "small.en" → "medium.en" (slowest)
WHISPER_MODEL: str = "base.en"
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"  # int8 quantisation for ARM Cortex-A76

# =============================================================================
# LLM  (Dolphin on RUBIK Pi for conversation, Gemini for vision only)
# =============================================================================
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.7
GEMINI_MAX_TOKENS: int = 1_024

# Dolphin3.0-Llama3.2-3B via llama-server (n_ctx 8192)
DOLPHIN_URL: str = f"{RUBIKPI_HOST}:{RUBIKPI_LLM_PORT}/v1/chat/completions"
DOLPHIN_TEMPERATURE: float = 0.7
DOLPHIN_MAX_TOKENS: int = 512
DOLPHIN_HISTORY_TURNS: int = 10  # user+assistant pairs kept; context is only 8k
DOLPHIN_TIMEOUT: float = 60.0  # seconds to wait for connect / first token

# The camera only fires when the transcript contains one of these (lowercase
# substring match). Those turns go to Gemini; everything else goes to Dolphin.
VISION_TRIGGER_PHRASES: tuple[str, ...] = (
    "what do you see",
    "what can you see",
    "what are you seeing",
    "look at",
    "take a look",
    "have a look",
    "can you see",
    "do you see",
    "what is this",
    "what's this",
    "what am i holding",
    "describe what",
    "in front of you",
    "how do i look",
    "what am i wearing",
    "read this",
    "use your camera",
)

LLM_SYSTEM_PROMPT: str = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a helpful voice assistant. Keep replies short and conversational.",
)

# =============================================================================
# TEXT-TO-SPEECH  (ElevenLabs)
# =============================================================================
# Find voice IDs: https://api.elevenlabs.io/v1/voices
# Free plans can only use premade voices via the API (library voices → 402).
ELEVENLABS_VOICE_ID: str = "EXAVITQu4vr4xnSDxMaL"  # "Sarah" (premade)
ELEVENLABS_MODEL_ID: str = "eleven_turbo_v2_5"  # Lowest-latency model
ELEVENLABS_OUTPUT_FORMAT: str = "pcm_16000"  # Matches SAMPLE_RATE

# =============================================================================
# CAMERA  (Picamera2 / Arducam IMX708)
# =============================================================================
# Set CAMERA_ENABLED=False to run in audio-only mode without code changes.
CAMERA_ENABLED: bool = True

# Half-sensor binned mode: full colour, fast readout, excellent for vision AI.
CAMERA_CAPTURE_WIDTH: int = 2_028
CAMERA_CAPTURE_HEIGHT: int = 1_520
CAMERA_JPEG_QUALITY: int = 85  # 60–95; higher = larger payload for Gemini

# Seconds to wait after camera.start() for AE/AWB/AF to converge.
# Can be reduced to 1.0 s on well-lit scenes.
CAMERA_WARMUP_SECONDS: float = 2.0

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"
