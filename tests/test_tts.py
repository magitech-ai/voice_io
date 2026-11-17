"""Tests for TextToSpeech SDK, normalization, Markdown cleanup, and streaming."""

import asyncio
import os
import tempfile
from collections.abc import Iterator

import numpy as np

from voice_io import (
    TTS_SAMPLE_RATE,
    SpeechTextSanitizer,
    StreamingSentenceBuffer,
    TextToSpeech,
    TtsBackend,
    to_pcm16,
)
from voice_io.normalize import Normalizer, NumberVerbalizer
from voice_io.respell import Respeller
from voice_io.textsplit import TextSplitter


class _MockTtsBackend:
    """Minimal synthetic backend verifying the TtsBackend protocol contract."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sr = sample_rate
        self.calls: list[str] = []
        self.last_seed: int | None = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def voices(self) -> tuple[str, ...]:
        return ("default",)

    @property
    def max_frames(self) -> int:
        return 100

    def seed(self, seed_value: int) -> None:
        self.last_seed = seed_value

    def synthesize_raw(
        self,
        text: str,
        *,
        temp: float = 0.5,
        eos: float = 0.9,
    ) -> tuple[np.ndarray, int]:
        del temp, eos
        self.calls.append(text)
        t = np.linspace(0, 0.25, int(self._sr * 0.25), endpoint=False, dtype=np.float32)
        wav = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        return wav, 12


def test_normalizer_spoken_forms() -> None:
    """Verifies numbers, currency, dates, and code identifiers normalize cleanly."""
    norm = Normalizer(use_nemo=False)
    assert "four hundred" in norm("400") or "four hundred" in NumberVerbalizer.cardinal(
        400
    )
    out = norm("Your order ships on March 3rd for $42.50. Call 415-555-0199.")
    assert "forty two dollars and fifty cents" in out
    assert "four one five, five five five, zero one nine nine" in out


def test_respeller_respects_word_boundaries_and_possessives() -> None:
    """Verifies lexicon respelling matches whole words and preserves possessives."""
    resp = Respeller()
    assert resp("Siobhan called") == "Shivawn called"
    assert resp("Eoin's book") == "Owen's book"


def test_text_splitter_splits_sentences_and_long_clauses() -> None:
    """Verifies sentence splitting and bounded clause splitting."""
    splitter = TextSplitter(clause_max=60)
    pieces = splitter.split(
        "First short sentence. Second sentence is much longer, with multiple "
        "clauses, so it should break cleanly at punctuation."
    )
    assert len(pieces) >= 3
    assert pieces[0] == "First short sentence."


def test_ensure_terminal_punct_edge_cases() -> None:
    """Verifies auto-punctuation appends periods only when appropriate."""
    cases = [
        ("hello", "hello."),
        ("hello  ", "hello."),
        ("hello.", "hello."),
        ("hello!", "hello!"),
        ("hello?", "hello?"),
        ("hello…", "hello…"),
        ('He said "hello."', 'He said "hello."'),
        ('He said "hello"', 'He said "hello".'),
        ("(hello)", "(hello)."),
        ("(hello.)", "(hello.)"),
        ("Wait,", "Wait,"),
        ("", ""),
        ("   ", "   "),
    ]
    for raw, expected in cases:
        assert TextToSpeech._ensure_terminal_punct(raw) == expected


def test_markdown_sanitizer_cleans_agent_responses() -> None:
    """Verifies Markdown cleanup keeps snake_case intact and drops code fences."""
    md = (
        "### Build Status\n"
        "- Updated `voice_io_config` in [the guide](https://example.com/docs).\n"
        "- Run **pytest** with *verbose* output and _no_ skips.\n\n"
        "```python\nprint('should be omitted')\n```\n"
        "Visit https://example.com next."
    )
    cleaned = SpeechTextSanitizer().clean(md)
    assert "Build Status." in cleaned
    assert "voice_io_config" in cleaned
    assert "the guide." in cleaned
    assert "https://example.com/docs" not in cleaned
    assert "should be omitted" not in cleaned
    assert "Run pytest with verbose output and no skips." in cleaned
    assert "Visit https://example.com next." in cleaned


def test_streaming_sentence_buffer_handles_tokens_and_code_blocks() -> None:
    """Verifies token-by-token buffering emits safe sentence boundaries."""
    buf = StreamingSentenceBuffer(sanitizer=SpeechTextSanitizer(), min_chars=10)
    deltas = [
        "Checking ",
        "`main.py` at ",
        "3.14 p.m. ",
        "today. Here ",
        "is code:\n```python\n",
        "x = 1.0\n```\n",
        "All checks passed",
    ]
    emitted: list[str] = []
    for delta in deltas:
        emitted.extend(buf.push(delta))
    emitted.extend(buf.flush())

    assert emitted[0] == "Checking main.py at 3.14 p.m. today."
    assert all("x = 1.0" not in chunk for chunk in emitted)
    assert "All checks passed" in emitted[-1]


def test_custom_tts_backend_contract_and_token_stream() -> None:
    """Verifies TextToSpeech delegates cleanly to any TtsBackend implementation."""
    mock = _MockTtsBackend(sample_rate=16000)
    assert isinstance(mock, TtsBackend)
    tts = TextToSpeech(backend=mock, retries=0)

    def _tokens() -> Iterator[str]:
        yield "Hello agent. "
        yield "Second sentence!"

    chunks = list(tts.stream_tokens(_tokens(), seed=42))
    assert mock.last_seed == 42
    assert len(chunks) == 2
    assert mock.calls == ["Hello agent.", "Second sentence!"]
    wav_bytes = tts.synthesize_wav_bytes("Ready.")
    assert wav_bytes[:4] == b"RIFF"


def test_default_tts_determinism_wav_and_async() -> None:
    """Verifies deterministic synthesis, streaming, WAV serialization, and async."""
    tts = TextToSpeech(voice="juno", retries=0)
    chunks = list(tts.stream("Hello there. Testing stream.", seed=0))
    assert len(chunks) == 2
    wav1 = tts("Hello.", seed=0)
    wav2 = tts("Hello.", seed=0)
    assert wav1.dtype == np.float32
    assert wav1.ndim == 1
    assert len(wav1) > TTS_SAMPLE_RATE * 0.4
    assert np.array_equal(wav1, wav2)
    assert len(to_pcm16(wav1)) == len(wav1) * 2

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        tts.save(path, wav1)
        assert os.path.getsize(path) > 44
    finally:
        os.remove(path)

    async def _run_async() -> np.ndarray:
        async_wav = await tts.asynthesize("Hello.", seed=0)
        streamed = [c async for c in tts.astream("Hello.", seed=0)]
        assert len(streamed) == 1
        return async_wav

    assert np.array_equal(asyncio.run(_run_async()), wav1)
