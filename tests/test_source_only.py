"""Hermetic smoke tests that do not require the separately licensed TTS weights."""

import numpy as np

from voice_io import SpeechToText, TextToSpeech


class _TtsBackend:
    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def voices(self) -> tuple[str, ...]:
        return ("test",)

    @property
    def max_frames(self) -> int:
        return 100

    def seed(self, seed_value: int) -> None:
        del seed_value

    def synthesize_raw(
        self,
        text: str,
        *,
        temp: float = 0.5,
        eos: float = 0.9,
    ) -> tuple[np.ndarray, int]:
        del text, temp, eos
        return np.full(4000, 0.1, dtype=np.float32), 4


class _SttBackend:
    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def model_name(self) -> str:
        return "test"

    def transcribe_pcm(self, audio: np.ndarray) -> tuple[str, int]:
        return f"samples={len(audio)}", 1


def test_custom_speech_backends_work_without_default_model_weights() -> None:
    """Exercises SDK synthesis, WAV serialization, and transcription via plugins."""
    tts = TextToSpeech(backend=_TtsBackend(), normalize=False, retries=0)
    audio = tts("Hello.")

    assert audio.dtype == np.float32
    assert audio.size == 4000
    assert tts.synthesize_wav_bytes("Hello.")[:4] == b"RIFF"

    stt = SpeechToText(backend=_SttBackend())
    result = stt.transcribe_array(audio, sample_rate=16000)

    assert result.text == "samples=4000"
    assert result.num_tokens == 1
