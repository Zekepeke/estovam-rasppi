"""
services/llm.py — Streaming LLM backends.

  DolphinLLM  — local Dolphin on the RUBIK Pi (llama-server), normal chat
  GeminiLLM   — Google Gemini, used only for vision turns
  LLMRouter   — picks one per turn based on whether an image is attached

Multimodal request structure
─────────────────────────────
When image_bytes is provided, the current turn's Content is built as:

    Content(role="user", parts=[
        Part(inline_data=Blob(mime_type="image/jpeg", data=<bytes>)),
        Part(text=<transcript>),
    ])

The image comes FIRST — Gemini attends to leading parts more reliably.

History management
──────────────────
Conversation history is maintained across turns as text-only Content
objects so the model retains context. Images are intentionally NOT stored
in history: re-sending large JPEGs on every turn would make requests
expensive and slow. The model already has the description from its
previous answer if the user asks a follow-up about what it saw.
"""

import json
import logging
from typing import Generator, Optional

import config

log = logging.getLogger(__name__)


class GeminiLLM:
    """
    Streaming Google Gemini client with optional vision input.

    Keep one instance alive for the program lifetime — the client
    connection and conversation history are maintained across turns.
    """

    def __init__(
        self,
        api_key: str = config.GEMINI_API_KEY,
        model:   str = config.GEMINI_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model   = model
        self._client = None  # genai.Client — created in initialize()

        # Text-only turn history: list of {"role": str, "parts": [...]}
        self._history: list[dict] = []
        self.last_response: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Create the google-genai client.

        Returns:
            True on success, False on import or auth error.
        """
        try:
            from google import genai
        except ImportError:
            log.error(
                "google-genai not installed. Run: pip install google-genai"
            )
            return False

        try:
            self._client = genai.Client(api_key=self.api_key)
            log.info("Gemini client ready  model=%s", self.model)
            return True

        except Exception:
            log.exception("Failed to initialise Gemini client")
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def stream_response(
        self,
        prompt:      str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response from Gemini, with an optional camera image.

        Args:
            prompt:      Transcribed user speech.
            image_bytes: In-memory JPEG bytes (from CameraCapture).
                         Pass None for a text-only request.

        Yields:
            Text tokens as they arrive from the API.
        """
        if self._client is None:
            log.error("GeminiLLM not initialised — call initialize() first")
            return

        # Late import to avoid module-level dependency issues
        from google.genai import types as genai_types

        has_image = bool(image_bytes)
        log.info(
            "Sending request to Gemini  [%s]",
            "vision + text" if has_image else "text-only",
        )

        try:
            # --- Build current-turn Content --------------------------------
            # Image first (if present), then the text question.
            parts: list = []

            if has_image:
                parts.append(
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            mime_type="image/jpeg",
                            data=image_bytes,
                        )
                    )
                )

            parts.append(genai_types.Part(text=prompt))

            current_content = genai_types.Content(role="user", parts=parts)

            # --- Assemble full contents list --------------------------------
            # Prepend text-only history so Gemini has conversational context.
            contents = list(self._history) + [current_content]

            # --- Stream from Gemini -----------------------------------------
            response = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=config.LLM_SYSTEM_PROMPT,
                    temperature=config.GEMINI_TEMPERATURE,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                ),
            )

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # --- Update history (text-only) --------------------------------
            # Store the user's text (not the image) so history stays compact.
            self.record_turn(prompt, full_response)
            self.last_response = full_response

            log.info("Gemini response: %r", full_response)

        except Exception:
            log.exception("Gemini streaming error")
            yield "I'm sorry, I encountered an error processing your request."

    def record_turn(self, prompt: str, response: str) -> None:
        """Append a text-only user/model exchange to the history."""
        self._history.append({"role": "user",  "parts": [{"text": prompt}]})
        self._history.append({"role": "model", "parts": [{"text": response}]})

    def clear_history(self) -> None:
        """Reset the multi-turn conversation context."""
        self._history.clear()
        log.info("Conversation history cleared")


class DolphinLLM:
    """
    Streaming client for Dolphin running under llama-server on the RUBIK Pi.

    Same interface as GeminiLLM. llama-server speaks the OpenAI chat
    completions protocol, so streaming is plain SSE ("data: {...}" lines)
    and needs nothing beyond requests. Text-only: image_bytes is ignored.
    """

    def __init__(self, url: str = config.DOLPHIN_URL) -> None:
        self.url = url
        self._session = None  # requests.Session — created in initialize()

        # OpenAI-format turn history: list of {"role": str, "content": str}
        self._history: list[dict] = []
        self.last_response: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Create the HTTP session and check that llama-server is reachable.

        An unreachable server is logged but not fatal: the RUBIK Pi may
        come up after the assistant, and each request retries anyway.

        Returns:
            True always, unless requests is missing.
        """
        try:
            import requests
        except ImportError:
            log.error("requests not installed. Run: pip install requests")
            return False

        self._session = requests.Session()
        health_url = self.url.split("/v1/")[0] + "/health"
        try:
            resp = self._session.get(health_url, timeout=5)
            resp.raise_for_status()
            log.info("Dolphin client ready  url=%s", self.url)
        except Exception as exc:
            log.warning("Dolphin server not reachable yet at %s (%s)", health_url, exc)
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def stream_response(
        self,
        prompt:      str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response from Dolphin.

        Args:
            prompt:      Transcribed user speech.
            image_bytes: Ignored — Dolphin is text-only.

        Yields:
            Text tokens as they arrive from llama-server.
        """
        if self._session is None:
            log.error("DolphinLLM not initialised — call initialize() first")
            return

        if image_bytes:
            log.warning("DolphinLLM is text-only — ignoring image")

        log.info("Sending request to Dolphin  [text-only]")
        self.last_response = None

        messages = (
            [{"role": "system", "content": config.LLM_SYSTEM_PROMPT}]
            + self._history
            + [{"role": "user", "content": prompt}]
        )

        try:
            with self._session.post(
                self.url,
                json={
                    "messages":    messages,
                    "stream":      True,
                    "temperature": config.DOLPHIN_TEMPERATURE,
                    "max_tokens":  config.DOLPHIN_MAX_TOKENS,
                },
                stream=True,
                timeout=config.DOLPHIN_TIMEOUT,
            ) as resp:
                resp.raise_for_status()

                full_response = ""
                # Split raw bytes, then decode: decode_unicode=True uses
                # str.splitlines(), which breaks SSE lines on U+2028 etc.
                for raw in resp.iter_lines():
                    line = raw.decode("utf-8")
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or []
                    token = choices[0].get("delta", {}).get("content") if choices else None
                    if token:
                        full_response += token
                        yield token

            self.record_turn(prompt, full_response)
            self.last_response = full_response
            log.info("Dolphin response: %r", full_response)

        except Exception:
            log.exception("Dolphin streaming error")
            yield "I'm sorry, I encountered an error processing your request."

    def record_turn(self, prompt: str, response: str) -> None:
        """Append a user/assistant exchange, trimming to DOLPHIN_HISTORY_TURNS."""
        self._history.append({"role": "user",      "content": prompt})
        self._history.append({"role": "assistant", "content": response})
        del self._history[:-2 * config.DOLPHIN_HISTORY_TURNS]

    def clear_history(self) -> None:
        """Reset the multi-turn conversation context."""
        self._history.clear()
        log.info("Conversation history cleared")


class LLMRouter:
    """
    Routes each turn to Gemini (when an image is attached) or Dolphin
    (everything else), with the same interface as the individual backends.

    After a turn completes, the exchange is copied into the other
    backend's history so follow-ups keep their context: if Gemini
    described the room, Dolphin can still answer "what colour was it?".
    """

    def __init__(self) -> None:
        self.gemini  = GeminiLLM()
        self.dolphin = DolphinLLM()

    def initialize(self) -> bool:
        return self.dolphin.initialize() and self.gemini.initialize()

    def stream_response(
        self,
        prompt:      str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        backend, other = (
            (self.gemini, self.dolphin) if image_bytes else (self.dolphin, self.gemini)
        )
        backend.last_response = None
        yield from backend.stream_response(prompt, image_bytes)
        if backend.last_response is not None:
            other.record_turn(prompt, backend.last_response)

    def clear_history(self) -> None:
        self.gemini.clear_history()
        self.dolphin.clear_history()
