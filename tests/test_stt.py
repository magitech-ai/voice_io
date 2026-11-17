"""Tests for SpeechToText SDK, custom STT backends, and AudioProcessor."""

import asyncio
import os

import numpy as np

from voice_io import (
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    AudioProcessor,
    SpeechToText,
    SttBackend,
    TextToSpeech,
    to_wav_bytes,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _MockSttBackend:
    """Deterministic mock STT backend verifying SttBackend protocol support."""

    @property
    def sample_rate(self) -> int:
        return STT_SAMPLE_RATE

    @property
    def model_name(self) -> str:
        return "mock-stt"

    def transcribe_pcm(self, audio: np.ndarray) -> tuple[str, int]:
        return f"frames:{len(audio)}", 3


def test_audio_processor_resample_wav_bytes_and_peaks() -> None:
    """Verifies 24 kHz -> 16 kHz resampling, WAV bytes round-trip, and peaks."""
    t = np.linspace(0, 1.0, TTS_SAMPLE_RATE, endpoint=False, dtype=np.float32)
    sine_24k = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    wav_bytes = to_wav_bytes(sine_24k, sample_rate=TTS_SAMPLE_RATE)

    resampled, sr = AudioProcessor.load_wav_bytes(wav_bytes, STT_SAMPLE_RATE)
    assert sr == STT_SAMPLE_RATE
    assert abs(len(resampled) - STT_SAMPLE_RATE) <= 2

    peaks = AudioProcessor.compute_peaks(resampled, num_bins=64)
    assert len(peaks) == 64
    assert max(peaks) == 1.0
    assert min(peaks) >= 0.0


def test_audio_processor_chunking() -> None:
    """Verifies long audio splitting stays bounded without dropping samples."""
    long_audio = (
        np.random.default_rng(0)
        .normal(0, 0.05, size=STT_SAMPLE_RATE * 35)
        .astype(np.float32)
    )
    chunks = AudioProcessor.split_chunks(
        long_audio, sample_rate=STT_SAMPLE_RATE, max_chunk_s=20.0
    )
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) == len(long_audio)


def test_custom_stt_backend_and_silence_gate() -> None:
    """Verifies custom SttBackend delegation and zero-audio short-circuit."""
    mock = _MockSttBackend()
    assert isinstance(mock, SttBackend)
    stt = SpeechToText(backend=mock)

    quiet = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
    quiet_res = stt.transcribe_array(quiet, sample_rate=STT_SAMPLE_RATE)
    assert quiet_res.text == ""
    assert quiet_res.num_tokens == 0

    t = np.linspace(0, 0.5, int(0.5 * TTS_SAMPLE_RATE), dtype=np.float32)
    tone = 0.3 * np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
    res = stt.transcribe_bytes(to_wav_bytes(tone, sample_rate=TTS_SAMPLE_RATE))
    assert res.model_name == "mock-stt"
    assert res.text.startswith("frames:")
    assert res.num_tokens == 3


def test_onnx_stt_transcribes_bundled_samples_and_async() -> None:
    """Verifies offline ONNX tiny/base STT models on bundled WAV samples."""
    sample_juno = os.path.join(REPO_ROOT, "samples", "juno_0.wav")
    sample_mira = os.path.join(REPO_ROOT, "samples", "mira_1.wav")

    tiny = SpeechToText(model_name="tiny")
    res_tiny = tiny.transcribe_file(sample_juno)
    assert "march" in res_tiny.text.lower()
    assert "tracking" in res_tiny.text.lower()
    assert res_tiny.rtf > 1.0

    base = SpeechToText(model_name="base")
    res_base = base.transcribe_file(sample_mira)
    lower = res_base.text.lower()
    assert "call me back" in lower
    assert "friday" in lower
    assert res_base.rtf > 1.0

    with open(sample_mira, "rb") as handle:
        mira_bytes = handle.read()
    async_res = asyncio.run(base.atranscribe_bytes(mira_bytes))
    assert async_res.text == res_base.text


def test_tts_to_stt_round_trip() -> None:
    """Synthesizes speech with TextToSpeech and recovers words via SpeechToText."""
    tts = TextToSpeech(voice="juno", retries=0)
    wav_24k = tts("Please open the studio door before noon.", seed=0)

    stt = SpeechToText(model_name="base")
    result = stt.transcribe_array(wav_24k, sample_rate=TTS_SAMPLE_RATE)
    lower = result.text.lower()
    assert "studio" in lower
    assert "door" in lower
    assert "noon" in lower
