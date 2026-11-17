"""voice_io: Offline CPU bidirectional TTS and STT SDK for AI agents.

Example usage:
  from voice_io import VoiceIO

  voice = VoiceIO()
  user = voice.listen()
  voice.speak("Got it, checking your repository now.")
"""

from typing import Any

from .audio import (
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    AudioPlayer,
    AudioProcessor,
    AudioRecorder,
    VadSegmenter,
)
from .backends import SttBackend, TtsBackend
from .textsplit import SpeechTextSanitizer, StreamingSentenceBuffer
from .tts import DEFAULT_WEIGHTS_PATH, VOICES, TextToSpeech

to_pcm16 = AudioProcessor.to_pcm16
to_wav_bytes = AudioProcessor.to_wav_bytes

__all__ = [
    "VoiceIO",
    "TextToSpeech",
    "SpeechToText",
    "TranscriptionResult",
    "TtsBackend",
    "SttBackend",
    "AudioProcessor",
    "AudioRecorder",
    "AudioPlayer",
    "VadSegmenter",
    "SpeechTextSanitizer",
    "StreamingSentenceBuffer",
    "TTS_SAMPLE_RATE",
    "STT_SAMPLE_RATE",
    "VOICES",
    "DEFAULT_WEIGHTS_PATH",
    "to_pcm16",
    "to_wav_bytes",
]


def __getattr__(name: str) -> Any:
    """Lazily imports ONNX STT and VoiceIO facade modules on first access."""
    if name in ("SpeechToText", "TranscriptionResult"):
        from . import stt

        return getattr(stt, name)
    if name == "VoiceIO":
        from . import session

        return session.VoiceIO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
