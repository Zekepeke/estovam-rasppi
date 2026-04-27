# Multimodal Voice Assistant — Raspberry Pi 5 + RUBIK Pi 3

A low-latency voice + vision assistant pipeline running on Raspberry Pi 5 with a RUBIK Pi 3 (Qualcomm QCS6490, Adreno 643 GPU) as an AI coprocessor.  
Speaks, listens, and **sees** — STT runs locally on Pi 5 via faster-whisper, LLM via Gemini, TTS via Orpheus-3B on the RUBIK Pi 3, vision via a 12 MP autofocus camera.

---

## Architecture

### Hardware

| Component | Model | Interface |
|---|---|---|
| Single-board computer | Raspberry Pi 5 (4 GB / 8 GB) | — |
| AI Coprocessor | RUBIK Pi 3 (Qualcomm QCS6490, Adreno 643 GPU) | Ethernet/Wi-Fi |
| Microphone array | ReSpeaker Mic Array v3.0 | USB |
| Speaker | Amazon Basics USB-powered speaker | 3.5 mm (ReSpeaker jack) |
| Camera | Arducam IMX708 (Camera Module 3) — 12 MP, Autofocus | 15–22 pin FFC MIPI CSI-2 |

> The ReSpeaker performs on-device Acoustic Echo Cancellation (AEC), so the
> microphone does not pick up the assistant's own TTS output.

### Software Stack

| Layer | Library / Service | Notes |
|---|---|---|
| Wake word | `openwakeword` | Fully local on Pi 5 — no API key, no cloud |
| Audio I/O | `sounddevice` | Direct ALSA access via libportaudio |
| Camera | `picamera2` | Official libcamera Python wrapper for Pi 5 |
| Speech-to-text | `faster-whisper` on Pi 5 | Local in-process — int8 Whisper base.en on ARM Cortex-A76 |
| LLM | Google Gemini (`gemini-2.5-flash`) | Cloud — handles both vision and text queries |
| Text-to-speech | Orpheus-3B on RUBIK Pi 3 (port 8081) | SNAC codec, Adreno 643 GPU; 24 kHz mono WAV |

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. PASSIVE LOOP  (Raspberry Pi 5)                          │
│     openwakeword feeds 80 ms audio chunks until             │
│     "Hey Jarvis" is detected (score ≥ threshold)            │
└────────────────────┬────────────────────────────────────────┘
                     │ wake word fired
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  2. PARALLEL CAPTURE  (Raspberry Pi 5)                       │
│     Two threads, started simultaneously                      │
│                                                              │
│   Thread A — AudioRecorder                                   │
│     sounddevice InputStream → VAD silence detection →        │
│     int16 NumPy array (user's speech)                        │
│                                                              │
│   Thread B — CameraCapture                                   │
│     Picamera2 preview (already warm, continuous AF) →        │
│     capture_file() → in-memory JPEG bytes                    │
│                                                              │
│   Both threads join(); audio is the long pole.               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. TRANSCRIBE  (Raspberry Pi 5 — local)                    │
│     faster-whisper base.en int8 runs in-process             │
│     → transcript string (no network call)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. LLM REQUEST  (Google Gemini — cloud)                    │
│     Pi 5 → Gemini 2.5 Flash (vision + text)                 │
│       messages: [system, history…, image, user transcript]  │
│     → streaming token generator (SSE)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. STREAMING TTS  (RUBIK Pi 3 — Orpheus-3B)                │
│     Tokens → sentence buffer →                              │
│     POST http://192.168.4.20:8081/synthesize →              │
│     WAV 24 kHz → sounddevice playback                       │
│     (first audio plays before LLM finishes generating)      │
└─────────────────────────────────────────────────────────────┘
```

#### Why latency stays low

- **Camera always warm.** Picamera2 is initialized once at startup. With
  `AfMode.Continuous`, the lens tracks the scene at all times. When the wake
  word fires, `capture_file()` grabs the current frame in ~0.1 s — no cold
  libcamera init, no autofocus wait.
- **Parallel capture.** The microphone starts recording at exactly the same
  moment the camera grabs its frame. Neither blocks the other.
- **No disk I/O.** The captured image is held in a `BytesIO` buffer and sent
  directly to Gemini as inline base64 — zero filesystem overhead.
- **Local STT.** faster-whisper runs in-process on the Pi 5, eliminating the
  HTTP round-trip to the RUBIK Pi 3 and saving ~50 ms per turn.
- **Streaming LLM + TTS.** Orpheus begins synthesising the first sentence
  while Gemini is still generating the rest.

---

## Known Performance

| Stage | Approximate latency |
|---|---|
| Wake word → audio capture | Instant (openwakeword is always listening) |
| STT — Whisper base.en int8 on Pi 5 | ~1–2 s for a typical utterance |
| LLM time-to-first-token (Gemini 2.5 Flash) | ~0.5–1 s |
| LLM generation | ~100 tokens/s (cloud) |
| TTS time-to-first-audio (Orpheus-3B on Adreno 643) | **~30–60 s for first sentence** ← current bottleneck |
| TTS subsequent sentences | Overlap with LLM generation |

> **TTS latency note:** Orpheus-3B generates tokens at approximately 4 t/s on the
> Adreno 643 GPU. A short sentence (~20 tokens) takes ~5 s; a longer one can
> take 30–60 s. This is the primary known limitation of the current pipeline.
> Reducing sentence length via the system prompt helps.

---

## Setup

### 1. System dependencies

Run these on the Pi **before** creating the Python environment.

```bash
# Update package lists
sudo apt update

# libcamera runtime + tools (required by picamera2)
sudo apt install -y libcamera-apps libcamera-dev

# Picamera2 Python library and its system-level dependencies
sudo apt install -y python3-picamera2

# PortAudio (required by sounddevice)
sudo apt install -y libportaudio2 portaudio19-dev

# Optional: verify the camera is detected by libcamera
libcamera-hello --list-cameras
```

> **Virtual environment note:** `python3-picamera2` installs into the system
> Python. To use it inside a venv, either:
>
> - Create the venv with `--system-site-packages`:
>   ```bash
>   python3 -m venv --system-site-packages .venv
>   ```
> - Or install the PyPI package instead (no system access needed):
>   ```bash
>   pip install picamera2
>   ```
>   The PyPI package pulls in `libcamera` bindings automatically on Pi OS Bookworm.

### 2. Clone and enter the repo

```bash
git clone <repository-url>
cd respeaker-demo
```

### 3. Create the Python virtual environment

```bash
# Use --system-site-packages if you installed picamera2 via apt above
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install \
  sounddevice \
  numpy \
  openwakeword \
  faster-whisper \
  requests \
  python-dotenv \
  picamera2        # skip if installed via apt + system-site-packages
```

Or use a `requirements.txt` if present:

```bash
pip install -r requirements.txt
```

### 5. Configure API keys

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```dotenv
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

- Gemini key: <https://aistudio.google.com/apikey>

### 6. (Optional) Verify camera

```bash
# Quick preview test — should show a live feed for 5 seconds
libcamera-hello -t 5000

# Or from Python
python3 -c "from picamera2 import Picamera2; c = Picamera2(); print(c.camera_properties)"
```

### 7. Run

```bash
source .venv/bin/activate
python main.py
```

Say **"Hey Jarvis"** to activate.

---

## RUBIK Pi 3 Setup

The RUBIK Pi 3 (Qualcomm QCS6490, Adreno 643 GPU) acts as the AI coprocessor,
running the LLM and TTS inference servers. The Pi 5 reaches it at `192.168.4.20`.  
To change this, update `RUBIKPI_HOST` in [config.py](config.py).

### LLM server — llama-server (Dolphin 3.0 Llama 3.2 3B Q4_K_M)

Runs on port **8080** with an OpenAI-compatible API.

```bash
# On the RUBIK Pi 3 — llama.cpp must be compiled with QNN/Adreno GPU support
cd ~/dev/llm/llama.cpp
./build/bin/llama-server \
  -m ~/dev/llm/models/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --chat-template chatml
```

**Endpoint summary:**

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/v1/chat/completions` | OpenAI Chat Completions JSON (`"stream": true`) | SSE token stream |
| GET  | `/health` | — | `{"status": "ok"}` |

> **Note:** Dolphin is set up and running but not yet used directly by the Pi 5
> orchestrator — the current pipeline sends all queries to Gemini (for vision
> support). See [Architecture Roadmap](#architecture-roadmap) for next steps.

### TTS server — Orpheus-3B (port 8081)

Runs on port **8081**. This server is being built separately. It accepts a
synthesis request and returns 24 kHz mono PCM WAV bytes.

**Expected endpoint:**

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/synthesize` | `{"text": "...", "speaker": "tara"}` | WAV bytes (`audio/wav`) |

**Speaker options:** `tara`, `leah`, `jess`, `mia`, `zoe`, `leo`, `dan`, `zac`

The server uses Orpheus-3B with SNAC codec decoding on the Adreno 643 GPU
(~4 tokens/second). The Pi 5 client is already wired up; TTS calls will fail
with a connection-refused warning (not a crash) until the server is running.

---

## Configuration

All tunable constants live in [config.py](config.py).

| Constant | Default | Description |
|---|---|---|
| `RUBIKPI_HOST` | `"http://192.168.4.20"` | RUBIK Pi 3 IP |
| `RUBIKPI_LLM_PORT` | `8080` | llama-server port (placeholder for future local LLM routing) |
| `RUBIKPI_TTS_PORT` | `8081` | Orpheus TTS server port |
| `WAKE_WORD_MODEL` | `"hey_jarvis"` | openwakeword model name |
| `WAKE_WORD_THRESHOLD` | `0.6` | Detection sensitivity (0–1) |
| `SILENCE_THRESHOLD` | `500` | RMS amplitude below which audio is silence |
| `SILENCE_DURATION` | `1.5` | Seconds of silence before recording stops |
| `WHISPER_MODEL` | `"base.en"` | `tiny.en` / `base.en` / `small.en` |
| `WHISPER_DEVICE` | `"cpu"` | Inference device for faster-whisper |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | Quantisation for ARM Cortex-A76 |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Gemini model ID |
| `TTS_SPEAKER` | `"tara"` | Orpheus voice |
| `TTS_SAMPLE_RATE` | `24000` | Orpheus output sample rate (Hz) |
| `TTS_REQUEST_TIMEOUT` | `180.0` | Per-sentence Orpheus timeout (seconds) |
| `CAMERA_ENABLED` | `True` | Set `False` to run audio-only |
| `CAMERA_CAPTURE_WIDTH/HEIGHT` | `2028 × 1520` | Half-sensor binned mode (fast, high quality) |
| `CAMERA_JPEG_QUALITY` | `85` | JPEG compression (60–95) |
| `CAMERA_WARMUP_SECONDS` | `2.0` | Seconds to wait for AE/AWB/AF to converge |

---

## Architecture Roadmap

The pipeline currently uses **Gemini for all LLM queries** because vision
support (Picamera2 → Gemini multimodal) is the primary feature. Future work:

1. **Local LLM routing** — add a routing layer that sends text-only queries to
   Dolphin 3.0 on the RUBIK Pi 3 (port 8080) and vision queries to Gemini.
   This reduces cloud dependency and latency for text-only turns.
2. **TTS latency** — profile Orpheus on Adreno 643; consider batching or
   quantising further to push below 10 s for a typical sentence.
3. **Conversation memory** — optional ChromaDB-backed user profile / summariser.
4. **Face recognition** — person-aware greetings using the camera.

---

## Troubleshooting

**Camera not detected**
```bash
libcamera-hello --list-cameras
# Should show: "Available cameras: 1"
# If empty, check the FFC ribbon cable is seated correctly (Pi 5 uses a
# 15-pin connector; the Arducam adapter bridges to 22-pin).
```

**`ImportError: No module named 'picamera2'` inside venv**  
Re-create the venv with `--system-site-packages`, or run `pip install picamera2`.

**Audio device index wrong**  
The script prints all available devices on startup. Set `sd.default.device` at
the top of `voice_assistant.py` to pin a specific input/output index.

**Wake word fires too easily / too rarely**  
Adjust `WAKE_WORD_THRESHOLD` (raise to reduce false positives, lower to
increase sensitivity).

**Whisper transcription is slow**  
Switch to `WHISPER_MODEL = "tiny.en"` for the fastest option.

**TTS returns connection refused**  
The Orpheus server on the RUBIK Pi 3 is not yet running. The assistant will
log a warning and skip playback for that sentence — it will not crash.
Start the Orpheus server on port 8081 to restore TTS.
