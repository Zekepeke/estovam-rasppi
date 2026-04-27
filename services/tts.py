"""
services/tts.py — Streaming text-to-speech via Orpheus-3B on RUBIK Pi 3.

Pipeline
────────
Gemini streams tokens  →  sentence buffer  →  Orpheus HTTP (RUBIK Pi 3 :8081)
                                                      ↓
                                           WAV 24 kHz  →  sounddevice

The sentence buffer lets audio start playing as soon as the FIRST complete
sentence arrives — the user hears the answer before Gemini has finished
generating the rest of the response.

Worker-thread architecture
──────────────────────────
A background TTS worker thread dequeues complete sentences and serialises
Orpheus requests + sounddevice playback. The main thread keeps feeding
sentences without ever blocking on audio I/O.
"""

import io
import logging
import queue
import re
import threading
import wave
from typing import Generator

import numpy as np
import requests
import sounddevice as sd

import config

log = logging.getLogger(__name__)

_TTS_URL = f"{config.RUBIKPI_HOST}:{config.RUBIKPI_TTS_PORT}/synthesize"

# Regex that splits on whitespace after a sentence-ending punctuation mark.
# "Hello world. How are you?" → ["Hello world.", "How are you?"]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete sentences from the accumulation buffer.

    Returns:
        (complete_sentences, remainder)  — remainder has no trailing sentence end.
    """
    parts = _SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    return parts[:-1], parts[-1]


def _synthesize_and_play(sentence: str) -> None:
    """Call Orpheus, decode the WAV response, and play it synchronously."""
    log.info("TTS → Orpheus: %r", sentence[:80])
    try:
        resp = requests.post(
            _TTS_URL,
            json={"text": sentence, "speaker": config.TTS_SPEAKER},
            timeout=config.TTS_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        sd.play(pcm, samplerate=config.TTS_SAMPLE_RATE, blocking=True)

    except requests.exceptions.ConnectionError:
        log.warning("TTS server unreachable at %s — skipping sentence", _TTS_URL)
    except requests.exceptions.Timeout:
        log.warning(
            "TTS request timed out after %.0f s — skipping sentence",
            config.TTS_REQUEST_TIMEOUT,
        )
    except Exception:
        log.exception("TTS error on sentence %r", sentence[:80])


class TextToSpeech:
    """
    Orpheus streaming TTS with sentence-level buffering.

    Keep one instance alive for the program lifetime.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Confirm the Orpheus client is configured and ready.

        The Orpheus server is NOT pre-warmed here — the first real TTS call
        may take 60+ seconds; that is expected and logged.

        Returns:
            True always (no local state to initialise).
        """
        log.info(
            "Orpheus TTS client ready  url=%s  speaker=%s",
            _TTS_URL,
            config.TTS_SPEAKER,
        )
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Non-streaming TTS — useful for short status phrases like
        "I didn't catch that" where latency is not critical.
        """
        _synthesize_and_play(text)

    def speak_streaming(self, text_generator: Generator[str, None, None]) -> None:
        """
        Consume a streaming text generator and play audio as sentences complete.

        Thread layout
        ─────────────
        Caller thread  — iterates text_generator, accumulates sentences,
                         enqueues them into sentence_queue.
        Worker thread  — dequeues sentences, calls Orpheus, plays audio.

        The worker runs slightly behind the caller, which means:
          • Sentence 1 audio starts playing while Gemini generates sentence 2.
          • No audio gap between sentences (worker queues them up).
          • The caller never blocks on audio I/O.
        """
        log.info("Speaking (streaming) ...")

        sentence_queue:     queue.Queue[str | None] = queue.Queue()
        streaming_complete: threading.Event         = threading.Event()

        def _worker() -> None:
            """Background thread: sentence → Orpheus → speaker."""
            while True:
                try:
                    sentence = sentence_queue.get(timeout=0.1)
                except queue.Empty:
                    if streaming_complete.is_set() and sentence_queue.empty():
                        break
                    continue

                if sentence is None:
                    break

                try:
                    _synthesize_and_play(sentence)
                finally:
                    sentence_queue.task_done()

        worker = threading.Thread(target=_worker, daemon=True, name="tts-worker")
        worker.start()

        buffer = ""
        try:
            for token in text_generator:
                buffer += token
                sentences, buffer = _split_sentences(buffer)
                for sentence in sentences:
                    if sentence.strip():
                        sentence_queue.put(sentence.strip())

            # Flush any remaining text that did not end with punctuation
            if buffer.strip():
                sentence_queue.put(buffer.strip())

        except Exception:
            log.exception("Error consuming text generator")

        finally:
            streaming_complete.set()
            # Allow up to TTS_REQUEST_TIMEOUT + 30 s for the last sentence to finish
            worker.join(timeout=config.TTS_REQUEST_TIMEOUT + 30)
            if worker.is_alive():
                log.warning("TTS worker did not finish within timeout")
