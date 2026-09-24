#!/usr/bin/env python3
"""
main.py — Multimodal Voice Assistant for Raspberry Pi 5

Entry point and top-level orchestrator. This file wires together the
hardware drivers (audio, camera) and AI services (STT, LLM, TTS) into
a single, readable interaction loop.

Run with:
    source .venv/bin/activate
    python main.py

Press Ctrl+C to exit cleanly.
"""

import logging
import signal
import sys
import threading
import time
from typing import Optional

import config
from hardware.audio  import WakeWordDetector, AudioRecorder, list_audio_devices
from hardware.camera import CameraCapture
from services.stt    import SpeechToText
from services.llm    import LLMRouter
from services.tts    import TextToSpeech


# =============================================================================
# LOGGING SETUP
# =============================================================================

def _configure_logging() -> None:
    """Configure the root logger — call once before anything else."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


log = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_config() -> None:
    """
    Check that required API keys are present before attempting any I/O.
    Exits with a clear error message if keys are missing.
    """
    missing: list[str] = []

    if not config.GEMINI_API_KEY or "YOUR_" in config.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY  →  https://aistudio.google.com/apikey")

    if not config.ELEVENLABS_API_KEY or "YOUR_" in config.ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY  →  https://elevenlabs.io/app/settings/api-keys")

    if missing:
        log.error("Missing API keys in .env.local:")
        for m in missing:
            log.error("  %s", m)
        sys.exit(1)


# =============================================================================
# VOICE ASSISTANT ORCHESTRATOR
# =============================================================================

class VoiceAssistant:
    """
    Owns all hardware and service instances and drives the main interaction
    loop.

    Interaction flow per wake word event
    ──────────────────────────────────────
    1. Wake word detected by WakeWordDetector
    2. AudioRecorder records until VAD silence
    3. SpeechToText transcribes the recorded audio
    4. If the transcript asks about the scene (config.VISION_TRIGGER_PHRASES),
       CameraCapture grabs a frame and LLMRouter sends it to Gemini;
       otherwise the turn goes to Dolphin on the RUBIK Pi, text-only
    5. TextToSpeech (ElevenLabs) plays the response sentence-by-sentence
    """

    def __init__(self) -> None:
        self.wake_detector = WakeWordDetector()
        self.recorder      = AudioRecorder()
        self.stt           = SpeechToText()
        self.llm           = LLMRouter()
        self.tts           = TextToSpeech()
        self.camera        = CameraCapture() if config.CAMERA_ENABLED else None

        # Set by SIGTERM / SIGINT handlers to break the main loop cleanly
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Initialize all components sequentially.

        Camera is initialized last and its failure is non-fatal — the
        assistant falls back to audio-only mode so a missing or broken
        camera never prevents the assistant from starting.

        Returns:
            True if all mandatory components initialized successfully.
        """
        log.info("=" * 62)
        log.info("  INITIALIZING MULTIMODAL VOICE ASSISTANT")
        log.info("=" * 62)

        list_audio_devices()

        # Mandatory components — any failure is fatal
        for component, name in [
            (self.wake_detector, "WakeWordDetector"),
            (self.stt,           "SpeechToText"),
            (self.llm,           "LLMRouter"),
            (self.tts,           "TextToSpeech"),
        ]:
            if not component.initialize():
                log.error("Failed to initialise %s — cannot continue", name)
                return False

        # Optional camera — failure degrades to audio-only
        if self.camera is not None:
            if not self.camera.initialize():
                log.warning(
                    "Camera unavailable — continuing in audio-only mode"
                )
                self.camera = None

        mode = "VISION + AUDIO" if self.camera else "AUDIO-ONLY"
        log.info("=" * 62)
        log.info("  ALL COMPONENTS READY  [%s]", mode)
        log.info("=" * 62)
        return True

    def cleanup(self) -> None:
        """
        Release all hardware resources.

        Called on clean shutdown (Ctrl+C / SIGTERM) and ensures:
        - sounddevice streams are closed (prevents ALSA lock on ReSpeaker)
        - Picamera2 is stopped (prevents libcamera lock on camera device)
        """
        log.info("Releasing hardware resources ...")
        self.wake_detector.cleanup()
        if self.camera is not None:
            self.camera.cleanup()
        log.info("Shutdown complete. Goodbye.")

    # ------------------------------------------------------------------
    # Vision routing
    # ------------------------------------------------------------------

    @staticmethod
    def _wants_vision(text: str) -> bool:
        """True if the transcript asks about what the camera can see."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in config.VISION_TRIGGER_PHRASES)

    def _capture_if_needed(self, text: str) -> Optional[bytes]:
        """
        Grab a frame only for vision questions.

        The camera runs continuously (warm, continuous AF), so capturing
        after transcription costs ~0.1 s — far cheaper than sending a JPEG
        with every turn. Returning None routes the turn to Dolphin.
        """
        if not self._wants_vision(text):
            log.info("No vision request — Dolphin, text-only")
            return None
        if self.camera is None:
            log.warning("Vision requested but camera unavailable — Dolphin, text-only")
            return None

        image_bytes = self.camera.capture_to_memory()
        if image_bytes:
            log.info("Vision request — Gemini with image (%.0f KB)", len(image_bytes) / 1_024)
        return image_bytes

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main interaction loop.

        Blocks until self._shutdown is set (Ctrl+C or SIGTERM).
        """
        log.info("Voice Assistant is ready!")
        log.info("  Say '%s' to activate ...", config.WAKE_WORD_DISPLAY)
        log.info("  Press Ctrl+C to exit")

        try:
            while not self._shutdown.is_set():

                # ── 1. Wake word ──────────────────────────────────────────
                detected = self.wake_detector.listen_for_wake_word()
                if not detected:
                    log.warning("Wake word detection failed — retrying in 1 s")
                    time.sleep(1)
                    continue

                # ── 2. Record ─────────────────────────────────────────────
                log.info("Recording ...")
                audio = self.recorder.record()

                if audio is None or len(audio) < config.SAMPLE_RATE * 0.5:
                    log.warning("Recording too short or failed — ready again")
                    continue

                # ── 3. Transcribe ─────────────────────────────────────────
                text = self.stt.transcribe(audio)
                if not text.strip():
                    log.warning("No speech detected")
                    self.tts.speak(
                        "I didn't catch that, Sir. Please try again."
                    )
                    continue

                image_bytes = self._capture_if_needed(text)

                # ── 4 + 5. LLM (Dolphin or Gemini) → streaming TTS ────────
                # llm.stream_response() is a generator. tts.speak_streaming()
                # consumes it sentence-by-sentence, playing audio as each
                # sentence arrives — so the first audio plays before the LLM
                # has finished generating the full response.
                response_gen = self.llm.stream_response(
                    prompt=text,
                    image_bytes=image_bytes,
                )
                self.tts.speak_streaming(response_gen)

                log.info(
                    "─" * 40 + "  Say '%s' again ...", config.WAKE_WORD_DISPLAY
                )

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")

        finally:
            self.cleanup()

    def request_shutdown(self) -> None:
        """Signal the main loop to exit cleanly (used by signal handlers)."""
        self._shutdown.set()


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

def _install_signal_handlers(assistant: VoiceAssistant) -> None:
    """
    Install SIGINT / SIGTERM handlers so Ctrl+C and systemd stop both
    trigger a graceful shutdown instead of abruptly killing the process.

    A abrupt kill can leave the ReSpeaker ALSA device or the Picamera2
    libcamera pipeline in a locked state, requiring a reboot to recover.
    """
    def _handler(sig: int, _frame) -> None:
        log.info("Signal %d received — requesting shutdown ...", sig)
        assistant.request_shutdown()

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    _configure_logging()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║        RASPBERRY PI 5 — MULTIMODAL VOICE ASSISTANT               ║
║                                                                  ║
║  Wake Word  →  Audio  →  Whisper STT (Pi 5)                      ║
║             →  Dolphin (RUBIK) / Gemini Vision  →  ElevenLabs TTS ║
║                                                                  ║
║  Hardware:  ReSpeaker v3.0  |  Arducam IMX708 (Camera Module 3)  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    _validate_config()

    assistant = VoiceAssistant()
    _install_signal_handlers(assistant)

    if not assistant.initialize():
        log.error("Initialisation failed — exiting")
        sys.exit(1)

    assistant.run()


if __name__ == "__main__":
    main()
