"""
services/stt.py — Speech-to-text via faster-whisper running locally on Pi 5.

Runs the Whisper model in-process — no network hop, saves ~50 ms per turn
compared to the previous remote HTTP approach.
"""

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

import config

log = logging.getLogger(__name__)


class SpeechToText:
    """
    Transcribes audio in-process using faster-whisper on the Pi 5.

    initialize() loads the model and warms up the int8 kernels so the first
    real transcription does not stall. transcribe() is otherwise stateless.
    """

    def __init__(self, model_name: str = config.WHISPER_MODEL) -> None:
        self.model_name = model_name
        self._model: WhisperModel | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Load the Whisper model and run a silent warmup pass.

        The warmup compiles the int8 ONNX/CTranslate2 kernels so the first
        real transcription call does not incur a 3-second compilation hitch.

        Returns:
            True on success, False on failure.
        """
        log.info(
            "Loading Whisper model '%s'  device=%s  compute_type=%s ...",
            self.model_name,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        try:
            self._model = WhisperModel(
                self.model_name,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )

            # Warmup — forces kernel compilation; result is discarded.
            silence = np.zeros(16_000, dtype=np.float32)
            list(self._model.transcribe(silence, language="en")[0])

            log.info("Whisper '%s' ready (int8 kernels compiled)", self.model_name)
            return True

        except Exception:
            log.exception("Failed to load Whisper model")
            return False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = config.SAMPLE_RATE,
    ) -> str:
        """
        Transcribe an int16 audio array locally via faster-whisper.

        Args:
            audio:       1-D or 2-D int16 NumPy array from AudioRecorder.
            sample_rate: Sample rate of the audio (default 16 000 Hz).

        Returns:
            Transcribed text, or an empty string on failure.
        """
        if self._model is None:
            log.error("SpeechToText not initialised")
            return ""

        log.info("Transcribing locally (faster-whisper) ...")
        t0 = time.monotonic()

        try:
            # faster-whisper expects float32 normalised to [-1, 1]
            audio_f32 = audio.flatten().astype(np.float32) / 32768.0

            segments, _info = self._model.transcribe(audio_f32, language="en")
            transcript = " ".join(seg.text for seg in segments).strip()

            log.info(
                "Transcription done  (%.2f s)  text=%r",
                time.monotonic() - t0,
                transcript,
            )
            return transcript

        except Exception:
            log.exception("Transcription error")
            return ""
