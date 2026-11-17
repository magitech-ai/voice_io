"""Tests for VAD endpointing, AudioPlayer cancellation, VoiceIO session, and CLI."""

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator

import numpy as np
import pytest

from voice_io import (
    STT_SAMPLE_RATE,
    AudioPlayer,
    AudioRecorder,
    SpeechToText,
    TextToSpeech,
    VadSegmenter,
    VoiceIO,
    to_pcm16,
)
from voice_io.__main__ import VoiceIoCli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _EchoTtsBackend:
    """Lightweight TTS backend producing deterministic tones per piece."""

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def voices(self) -> tuple[str, ...]:
        return ("default",)

    @property
    def max_frames(self) -> int:
        return 100

    def seed(self, seed_value: int) -> None:
        del seed_value

    def synthesize_raw(
        self, text: str, *, temp: float = 0.5, eos: float = 0.9
    ) -> tuple[np.ndarray, int]:
        del text, temp, eos
        wav = np.full(1600, 0.2, dtype=np.float32)
        return wav, 10


class _FixedSttBackend:
    """Lightweight STT backend returning frame count."""

    @property
    def sample_rate(self) -> int:
        return STT_SAMPLE_RATE

    @property
    def model_name(self) -> str:
        return "fixed"

    def transcribe_pcm(self, audio: np.ndarray) -> tuple[str, int]:
        return f"captured {len(audio)} samples", 4


class _StubRecorder(AudioRecorder):
    """Stub recorder returning synthetic speech without opening mic hardware."""

    def __init__(self, samples: np.ndarray) -> None:
        super().__init__()
        self._samples = samples

    def listen_utterance(self, **kwargs: object) -> np.ndarray:
        del kwargs
        return self._samples

    def record_seconds(self, duration_s: float) -> np.ndarray:
        del duration_s
        return self._samples


def _pcm_frame(amp: float, frame_ms: int = 40) -> bytes:
    n = int(STT_SAMPLE_RATE * frame_ms / 1000)
    t = np.linspace(0, frame_ms / 1000.0, n, endpoint=False, dtype=np.float32)
    tone = (amp * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    return to_pcm16(tone)


def test_vad_segmenter_preroll_click_rejection_and_silence_stop() -> None:
    """Verifies VAD ignores single clicks, keeps pre-roll, and ends on silence."""
    vad = VadSegmenter(
        sample_rate=STT_SAMPLE_RATE,
        frame_ms=40,
        energy_threshold=0.05,
        speech_trigger_frames=2,
        silence_duration_s=0.16,
        pre_roll_s=0.12,
        min_speech_s=0.12,
        wait_timeout_s=1.0,
    )
    quiet = _pcm_frame(0.001)
    speech = _pcm_frame(0.3)

    # Single transient click surrounded by silence must not trigger speech.
    assert vad.process_pcm16(quiet) == "waiting"
    assert vad.process_pcm16(speech) == "waiting"
    assert vad.process_pcm16(quiet) == "waiting"
    assert not vad.triggered

    # Pre-trigger quiet frame followed by consecutive voiced frames triggers.
    assert vad.process_pcm16(quiet) == "waiting"
    assert vad.process_pcm16(speech) == "waiting"
    assert vad.process_pcm16(speech) == "speaking"
    assert vad.triggered

    # Trailing silence ends the utterance automatically.
    states = [vad.process_pcm16(quiet) for _ in range(5)]
    assert "done" in states
    audio = vad.get_audio()
    # At least pre-roll (3 frames) + speech + silence tail are preserved.
    assert len(audio) >= int(STT_SAMPLE_RATE * 0.32)


def test_vad_segmenter_timeout_returns_empty() -> None:
    """Verifies VAD times out cleanly when user never speaks."""
    vad = VadSegmenter(
        sample_rate=STT_SAMPLE_RATE,
        frame_ms=40,
        energy_threshold=0.05,
        wait_timeout_s=0.20,
    )
    quiet = _pcm_frame(0.001)
    states = [vad.process_pcm16(quiet) for _ in range(6)]
    assert "timeout" in states
    assert vad.get_audio().size == 0


def test_audio_player_respects_pre_set_cancel_event() -> None:
    """Verifies cancelled AudioPlayer exits without spawning playback."""
    cancel = threading.Event()
    cancel.set()
    player = AudioPlayer(sample_rate=16000)
    samples, ttfa, completed = player.play_stream(
        [np.zeros(400, dtype=np.float32)], cancel_event=cancel
    )
    assert samples == 0
    assert ttfa is None
    assert completed is False


def test_voice_io_session_listen_speak_stream_and_interrupt() -> None:
    """Verifies unified VoiceIO sync, async token stream, and barge-in."""
    stt = SpeechToText(backend=_FixedSttBackend())
    tts = TextToSpeech(backend=_EchoTtsBackend(), retries=0, normalize=False)
    fake_speech = np.full(int(0.5 * STT_SAMPLE_RATE), 0.2, dtype=np.float32)
    voice = VoiceIO(stt=stt, tts=tts, recorder=_StubRecorder(fake_speech))

    heard = voice.listen()
    assert heard.text == f"captured {len(fake_speech)} samples"

    # Array/bytes dispatch on transcribe
    direct = voice.transcribe(fake_speech, sample_rate=STT_SAMPLE_RATE)
    assert direct.text == heard.text

    wav = voice.speak("First sentence. Second sentence.", play=False)
    assert len(wav) == 3200

    cancel = threading.Event()
    cancel.set()
    cancelled_wav = voice.speak("Will not synthesize.", play=False, cancel_event=cancel)
    assert cancelled_wav.size == 0

    async def _async_tokens() -> AsyncIterator[str]:
        for tok in ("Hello ", "user. ", "How ", "are you?"):
            yield tok

    async def _run_async() -> tuple[str, np.ndarray]:
        res = await voice.alisten()
        stream_wav = await voice.aspeak_stream(_async_tokens(), play=False)
        return res.text, stream_wav

    text, stream_wav = asyncio.run(_run_async())
    assert text == heard.text
    assert len(stream_wav) == 3200


def test_cli_help_and_transcribe_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies voice-io CLI parser and structured --json transcription."""
    with pytest.raises(SystemExit) as exc_info:
        VoiceIoCli().run(["--help"])
    assert exc_info.value.code == 0
    capsys.readouterr()

    sample_mira = os.path.join(REPO_ROOT, "samples", "mira_1.wav")
    code = VoiceIoCli().run(
        ["--transcribe", sample_mira, "--stt-model", "base", "--json"]
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert "call me back" in payload["text"].lower()
    assert payload["model_name"] == "base"
    assert payload["rtf"] > 1.0
