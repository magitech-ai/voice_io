# `voice_io`

An offline, CPU-only bidirectional **Speech-to-Text (`SpeechToText`)** and **Text-to-Speech (`TextToSpeech`)** SDK and CLI designed to wire directly into AI agents. Runs fast on standard x86_64/ARM CPUs with **no GPU and no PyTorch required**, managed with Astral `uv`.

## Features

- **Unified AI Agent Session (`VoiceIO`)**:
  - Hands-free turn capture via energy-based **VAD endpointing** (`listen` / `alisten`) with pre-roll buffering so word onsets are never clipped.
  - **Incremental LLM token streaming** (`speak_stream` / `aspeak_stream`) accepting sync or async token iterators, flushing sentence-by-sentence directly to speakers.
  - **Barge-in playback cancellation** (`interrupt()` / `cancel_event`) and automatic half-duplex mic gating to prevent speaker-to-mic self-echo.
  - **Async-first support** (`alisten`, `atranscribe`, `aspeak`, `aspeak_stream`) backed by `asyncio.to_thread` and thread-safe engine locks.
- **LLM Markdown Sanitizer (`SpeechTextSanitizer` & `StreamingSentenceBuffer`)**:
  - Strips headings, bold/italic markers, bullet prefixes, Markdown link URLs, inline backticks, and fenced code blocks before synthesis while preserving spoken prose, numbers, `snake_case` names, and bare URLs.
- **Pluggable Backend Architecture (`TtsBackend`, `SttBackend`)**:
  - Agent code depends only on generic `VoiceIO`, `TextToSpeech`, and `SpeechToText` interfaces—swap TTS or STT model backends in one constructor argument without changing application code.
  - Default TTS backend: 24 kHz pure-NumPy int8 flow synthesis (`juno` and `mira` voices) with spoken-form normalization, phonetic respelling, and clause-level streaming; requires separately obtained model weights.
  - Default STT backend: 16 kHz offline ONNX encoder-decoder recognition (`tiny` and `base` models) with KV-cached decoding, FFT resampling, WAV/FFmpeg byte decoding, and quiet-frame chunking.

## Installation (`uv`)

```bash
uv sync
```

## TTS Model Weights

The source repository does not include `woof_int8.npz`. The default local TTS
backend requires this model archive. Obtain it separately through an authorized
distribution and point `VOICE_IO_WEIGHTS` to its path, or pass the path using
the SDK's `weights` argument. Without the archive, use a custom
`TtsBackend` implementation instead. The model is licensed for non-commercial
use only; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

System audio utilities used for live microphone/speaker I/O on Linux (optional when passing audio bytes in memory):
- Capture: `parec` (PipeWire/PulseAudio) or `arecord` (ALSA)
- Playback: `paplay` (PipeWire/PulseAudio) or `aplay` (ALSA)
- Non-WAV containers (WebM, Ogg, MP3): `ffmpeg`

## Python SDK Quickstart

### 1. Synchronous Agent Voice Loop

```python
from voice_io import VoiceIO

voice = VoiceIO(stt_model="base", voice="juno")

# Record one spoken turn (stops automatically after 0.9s trailing silence)
user_turn = voice.listen()
print("User:", user_turn.text)

# Speak an LLM Markdown response back cleanly
agent_reply = "### Done\nUpdated `config.yaml`. Run **pytest** to verify."
voice.speak(agent_reply)
```

### 2. Async Agent + Streaming LLM Tokens

```python
import asyncio
from collections.abc import AsyncIterator
from voice_io import VoiceIO

async def stream_llm_reply(prompt: str) -> AsyncIterator[str]:
  # Replace with your OpenAI / Anthropic / Gemini / local LLM async stream
  for token in ["Sure! ", "Checking ", "`status.py` ", "now."]:
    await asyncio.sleep(0.02)
    yield token

async def main() -> None:
  voice = VoiceIO()
  heard = await voice.alisten()
  if heard.text:
    await voice.aspeak_stream(stream_llm_reply(heard.text))

asyncio.run(main())
```

### 3. In-Memory Bytes (WebSockets / Server Agents)

```python
from voice_io import SpeechToText, TextToSpeech

stt = SpeechToText(model_name="base")
tts = TextToSpeech(voice="juno")

# Decode arbitrary WAV/WebM/Ogg/MP3 bytes from a client socket
result = stt.transcribe_bytes(incoming_audio_bytes)

# Synthesize directly to 16-bit PCM WAV bytes for streaming back
reply_wav = tts.synthesize_wav_bytes("Here is what I found.")
```

### 4. Swapping TTS or STT Backends

```python
from voice_io import SpeechToText, TextToSpeech, VoiceIO

# Inject any object implementing TtsBackend or SttBackend
tts = TextToSpeech(backend=my_custom_tts_backend)
stt = SpeechToText(backend=my_custom_stt_backend)
voice = VoiceIO(stt=stt, tts=tts)
```

## CLI Usage (`voice-io`)

Designed to compose cleanly in shell pipelines (clean outputs on `stdout`, timing/status on `stderr`):

```bash
# Speak directly to speakers
uv run voice-io "Agent connected and ready." --play

# Pipe streamed lines into TTS
printf "First sentence.\nSecond sentence.\n" | uv run voice-io --stdin

# Save speech to WAV
uv run voice-io "Hello world." -o hello.wav

# Listen with VAD auto-stop and emit structured JSON
uv run voice-io --listen --json

# End-to-end mic -> STT -> TTS hardware check
uv run voice-io --listen --echo

# Transcribe an audio file
uv run voice-io --transcribe samples/mira_1.wav --stt-model base --json
```

## Development & Verification

```bash
# Source-only smoke tests, also used in CI (no TTS model weights required)
uv run --locked python -m pytest -q tests/test_source_only.py

# Full test suite; model-backed TTS tests require the separately obtained
# weights described above
uv run pytest -v

# Check Google Python Style Guide compliance
uvx ruff check --select E,F,W,I,N,D,UP --ignore E501,D105,D107 --config 'lint.pydocstyle.convention = "google"' voice_io/ tests/
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map, backend protocols, and VAD design.

## Licenses & Model Attributions

- **SDK source**: Licensed under the MIT License in [LICENSE-MIT](LICENSE-MIT), subject to the third-party attributions in [NOTICE](NOTICE).
- **Default TTS weights (`woof_int8.npz`, `"Voice by Underdog Woof"`, `underdog.ai`)**: Not included in this repository. When obtained separately, they are licensed under the **Underdog Non-Commercial License 1.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)). Commercial use requires a separate license from Conway (`license@conway.tech`). When using these weights in a non-commercial app that plays outputs, display `"Voice by Underdog Woof"` with a link to `underdog.ai` per Section 4 of [LICENSE](LICENSE).
- **Default STT ONNX weights (`UsefulSensors/moonshine` English `tiny`/`base`)**: Released by Useful Sensors under the permissive **MIT License**.
- The SDK architecture (`TtsBackend` / `SttBackend`) is backend-agnostic so commercial builds can inject Apache 2.0 / MIT TTS weights via `TextToSpeech(backend=...)`.
