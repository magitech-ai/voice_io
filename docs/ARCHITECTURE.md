# `voice_io` Architecture

`voice_io` provides a modular, offline CPU speech I/O layer tailored for conversational AI agents. Application code interacts strictly with generic capability interfaces (`VoiceIO`, `TextToSpeech`, `SpeechToText`, `AudioRecorder`, `AudioPlayer`), keeping underlying model weights and runtimes isolated behind narrow protocols.

## Data Flow

```text
Microphone (parec / arecord) OR Audio Bytes / File (WAV / WebM / MP3)
  │
  ▼  AudioRecorder.listen_utterance() + VadSegmenter (pre-roll + trailing silence stop)
AudioProcessor (16 kHz mono float32, FFT resample, 30s quiet-cut chunking)
  │
  ▼  SttBackend (default: OnnxSttBackend, KV-cached ONNX encoder-decoder)
TranscriptionResult (text, duration_s, inference_s, rtf, num_tokens, peaks)
  │
  ▼  AI Agent / LLM Reasoning Loop (sync or async token stream)
  │
  ├── StreamingSentenceBuffer + SpeechTextSanitizer (Markdown & code-block cleanup)
  ├── Normalizer + Respeller (numbers, currency, dates, URLs, respell.tsv)
  ├── TextSplitter (bounded sentence/clause chunks for low TTFA)
  │
  ▼  TtsBackend (default: LocalInt8FlowBackend, 24 kHz pure-NumPy int8 flow)
AudioPlayer (interruptible paplay / aplay PCM stream + barge-in cancel_event)
```

## Module Map

| Module | Primary Symbols | Responsibility |
|---|---|---|
| `voice_io/__init__.py` | `VoiceIO`, `TextToSpeech`, `SpeechToText` | Public SDK surface with lazy `__getattr__` imports for fast TTS-only startup. |
| `voice_io/backends.py` | `TtsBackend`, `SttBackend` | `@runtime_checkable` protocols decoupling agent code from concrete models. |
| `voice_io/session.py` | `VoiceIO` | Bidirectional agent session coordinator (`listen`, `transcribe`, `speak`, `speak_stream`, `interrupt`, and async equivalents). |
| `voice_io/audio.py` | `AudioProcessor`, `VadSegmenter`, `AudioRecorder`, `AudioPlayer` | Resampling, WAV/FFmpeg decoding, PCM serialization, VAD endpointing, mic capture, and interruptible speaker playback. |
| `voice_io/tts.py` | `TextToSpeech` | High-level synthesis facade, Markdown cleanup, normalization orchestration, clause streaming, and stall retries. |
| `voice_io/stt.py` | `SpeechToText`, `OnnxSttBackend`, `TranscriptionResult` | Offline CPU ONNX speech recognition, peak extraction, and async wrappers. |
| `voice_io/runtime.py` | `LocalInt8FlowBackend`, `SpeechModel`, `AudioDecoder`, `WeightLoader` | Pure-NumPy int8 flow transformer and Mimi-adapted 24 kHz audio decoder. |
| `voice_io/textsplit.py` | `SpeechTextSanitizer`, `StreamingSentenceBuffer`, `TextSplitter` | Markdown-to-speech cleaning, incremental LLM delta buffering, and clause chunking. |
| `voice_io/normalize.py` | `Normalizer`, `NumberVerbalizer` | Spoken-form text normalization for numbers, dates, times, currency, units, and URLs. |
| `voice_io/respell.py` | `Respeller` | Word-boundary and accent-folding phonetic lexicon backed by `respell.tsv`. |
| `voice_io/__main__.py` | `VoiceIoCli`, `main` | Scriptable `voice-io` CLI entrypoint (`--listen`, `--transcribe`, `--stdin`, `--json`, `--echo`). |

## Plugging In Custom Model Backends

### Custom TTS Backend

Implement `TtsBackend` (`sample_rate`, `voices`, `max_frames`, `seed()`, `synthesize_raw()`) and pass it to `TextToSpeech`:

```python
tts = TextToSpeech(backend=my_tts_backend)
voice = VoiceIO(tts=tts)
```

`TextToSpeech` automatically handles Markdown sanitization, spoken-form normalization, sentence splitting, token-stream buffering, and in-memory WAV serialization on top of your backend. Setting `max_frames = 0` disables latent-cap retries when using non-frame-capped models.

### Custom STT Backend

Implement `SttBackend` (`sample_rate`, `model_name`, `transcribe_pcm(audio) -> tuple[str, int]`) and pass it to `SpeechToText`:

```python
stt = SpeechToText(backend=my_stt_backend)
voice = VoiceIO(stt=stt)
```

`SpeechToText` handles container decoding, spectral resampling to `backend.sample_rate`, silence skipping, quiet-cut chunking for long audio, and async worker execution automatically.
