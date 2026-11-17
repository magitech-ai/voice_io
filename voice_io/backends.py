"""Backend protocol interfaces for pluggable TTS and STT engines."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TtsBackend(Protocol):
    """Protocol implemented by text-to-speech synthesis backends."""

    @property
    def sample_rate(self) -> int:
        """Native output waveform sample rate in Hz."""

    @property
    def voices(self) -> tuple[str, ...]:
        """Supported speaker or voice identifiers."""

    @property
    def max_frames(self) -> int:
        """Maximum latent or generation frame budget per utterance."""

    def seed(self, seed_value: int) -> None:
        """Reseeds deterministic synthesis sampling, if supported."""

    def synthesize_raw(
        self,
        text: str,
        *,
        temp: float = 0.5,
        eos: float = 0.9,
    ) -> tuple[np.ndarray, int]:
        """Synthesizes normalized text into ``(waveform_1d_float32, frame_count)``."""


@runtime_checkable
class SttBackend(Protocol):
    """Protocol implemented by speech-to-text recognition backends."""

    @property
    def sample_rate(self) -> int:
        """Expected mono input waveform sample rate in Hz."""

    @property
    def model_name(self) -> str:
        """Identifier of the active recognition model."""

    def transcribe_pcm(self, audio: np.ndarray) -> tuple[str, int]:
        """Transcribes mono 1D float32 audio and returns ``(text, num_tokens)``."""
