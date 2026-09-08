# Desk Robot — agent conventions

## Architecture rules (locked)
- Pi 5 is the orchestrator. It speaks protocols to coprocessors
  (HTTP to RUBIK, serial to the motion board) — it never becomes one.
- Four seams only: AIEngine, CameraInterface, AudioSink, MotionInterface.
  New hardware gets a new implementation of an existing seam, not a new seam.
- Motion is event-driven. main.py emits state on EventBus; nothing
  downstream is ever imported by the core loop directly.
- No hardcoded hosts, ports, or model names outside config.py.

## Current topology (decided 2026-09-08 — update this block, not just config.py, when it changes)
- RUBIK Pi 3 is reached via its Tailscale address, set once in config.py.
  Never hardcode a LAN IP in README, .env.example, or code.
- STT runs locally on the Pi 5 (faster-whisper, in-process). RUBIK is
  reserved for the LLM.
- TTS is ElevenLabs. Not revisited until after the event bus ships.
- Vision: turns with an image go to Gemini multimodal (AIEngine's second
  tier); RUBIK's llama-server stays text-only and does not take images.

## Deferred until Tommy's motion platform lands
- SerialMotion, real pose ranges, baud/axis conventions.
- Emotion-tag parsing — there is no motion consumer for it yet.
Do not scaffold these speculatively; MockMotion is the seam contract.

## Running it
- `python main.py` is the only entry point. Files at repo root
  (parrot.py, test_wakeword.py) are throwaway hardware probes, not
  part of the app — do not import from them.
