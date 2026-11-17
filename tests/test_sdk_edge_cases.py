"""Tests for text splitting, TTS/STT SDK edge cases, ONNX mocks, and VoiceIO async."""

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voice_io import (
    STT_SAMPLE_RATE,
    AudioRecorder,
    SpeechTextSanitizer,
    SpeechToText,
    StreamingSentenceBuffer,
    TextToSpeech,
    VoiceIO,
    to_pcm16,
    to_wav_bytes,
)
from voice_io.stt import OnnxSttBackend
from voice_io.textsplit import TextSplitter, split_text
from voice_io.tts import _resolve_weights_path


class _TinyTtsBackend:
    """Fast deterministic backend for TTS and VoiceIO edge-case tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def voices(self) -> tuple[str, ...]:
        return ("default", "alt")

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
        del temp, eos
        self.calls.append(text)
        wav = np.full(800, 0.2, dtype=np.float32)
        return wav, 4


class _TinySttBackend:
    """Fast deterministic backend recording received PCM arrays."""

    def __init__(self) -> None:
        self.seen_lengths: list[int] = []

    @property
    def sample_rate(self) -> int:
        return STT_SAMPLE_RATE

    @property
    def model_name(self) -> str:
        return "tiny-mock"

    def transcribe_pcm(self, audio: np.ndarray) -> tuple[str, int]:
        self.seen_lengths.append(len(audio))
        return f"samples={len(audio)}", 2


class _RecordingSpy(AudioRecorder):
    """Subclassed recorder tracking active state and stop calls without mic I/O."""

    def __init__(self, *, active: bool = False) -> None:
        super().__init__()
        self._active = active
        self.stopped = False

    @property
    def is_recording(self) -> bool:
        return self._active

    def stop(self) -> np.ndarray:
        self._active = False
        self.stopped = True
        return np.zeros(0, dtype=np.float32)

    def record_seconds(self, duration_s: float) -> np.ndarray:
        del duration_s
        return np.full(STT_SAMPLE_RATE // 4, 0.2, dtype=np.float32)

    def listen_utterance(self, **_kwargs: Any) -> np.ndarray:
        return np.full(STT_SAMPLE_RATE // 4, 0.2, dtype=np.float32)


def test_textsplit_sanitizer_abbreviations_oversized_and_streaming_buffer() -> None:
    """Covers Markdown cleanup, abbreviations, word-wrap chunks, and stream buffer."""
    sanitizer = SpeechTextSanitizer()
    md = (
        "# Heading\n"
        "> Quoted text\n"
        "- Bullet `code_here` with [a link](https://example.com)\n"
        "1. First numbered item\n"
        "```python\nprint('ignore code block')\n```\n"
        "| col1 | col2 |\n"
        "| ---- | ---- |\n"
        "| val1 | val2 |\n"
    )
    cleaned = sanitizer.clean(md)
    assert "ignore code block" not in cleaned
    assert "a link" in cleaned
    assert sanitizer.clean("   \n\t  ") == ""

    assert split_text("   ") == []
    splitter = TextSplitter(clause_max=24)
    chunks = splitter.split(
        'Dr. Smith met Prof. Lee in the U.S. at 3.14 p.m. "Wait... really?!"'
    )
    assert len(chunks) >= 2

    long_no_punct = "one two three, four five six, seven eight nine ten eleven"
    wrapped = TextSplitter(clause_max=20, first_max=20).split(long_no_punct)
    assert len(wrapped) >= 3
    assert all(len(c) <= 60 for c in wrapped)

    buf = StreamingSentenceBuffer(min_chars=8)
    emitted: list[str] = []
    for tok in ["Hello ", "world. ", "Second ", "part!"]:
        emitted.extend(buf.push(tok))
    emitted.extend(buf.flush())
    assert any("Hello world" in s for s in emitted)
    assert buf.flush() == []

    cue_san = SpeechTextSanitizer(code_block_cue="Code omitted.")
    assert "Code omitted." in cue_san.clean("Look:\n```python\nx=1\n```\nDone.")
    assert "unclosed" in cue_san.clean("```python\nunclosed")
    assert cue_san.clean("---\n   * bullet\n__bold__ ~~strike~~")
    assert cue_san.clean("Paragraph 1\n\nParagraph 2") == "Paragraph 1 Paragraph 2"
    cue_buf = StreamingSentenceBuffer(sanitizer=cue_san, min_chars=8)
    cue_emitted = cue_buf.push("Look:\n```python\nx=1\n```\nEnd")
    cue_emitted.extend(cue_buf.flush())
    assert cue_emitted

    semi_chunks = TextSplitter(clause_max=20, first_max=20).split(
        "Alpha beta gamma; delta epsilon zeta: eta theta. Hi. Ok."
    )
    assert len(semi_chunks) >= 2
    paren_chunks = TextSplitter(clause_max=30, first_max=20).split(
        "One (a. b. c.) two! Three? Four... Five: six, seven. " + "word " * 40 + "end."
    )
    assert len(paren_chunks) >= 3


def test_tts_weights_empty_inputs_tokens_cancel_save_and_async(
    tmp_path: Path,
) -> None:
    """Covers weights search, empty/whitespace TTS, stream_tokens, cancel, and async."""
    assert Path(_resolve_weights_path()).exists()

    backend = _TinyTtsBackend()
    tts = TextToSpeech(backend=backend, retries=0)

    empty_wav = tts("   ")
    assert len(empty_wav) == 0
    assert list(tts.stream("  \n ")) == []

    async def _run_async() -> None:
        a_wav = await tts.asynthesize("")
        assert len(a_wav) == 0

        streamed_empty = [c async for c in tts.astream("   ")]
        assert streamed_empty == []

        non_empty = [
            c async for c in tts.astream("First sentence. Second sentence.", seed=1)
        ]
        assert len(non_empty) == 2

    asyncio.run(_run_async())

    cancel = threading.Event()
    pieces: list[np.ndarray] = []
    for chunk in tts.stream(
        "One sentence. Two sentence. Three sentence.",
        cancel_event=cancel,
    ):
        pieces.append(chunk)
        cancel.set()
    assert len(pieces) == 1

    tok_cancel = threading.Event()
    tok_pieces: list[np.ndarray] = []
    for chunk in tts.stream_tokens(
        ["Hello ", "agent. ", "Second ", "sentence!"],
        cancel_event=tok_cancel,
    ):
        tok_pieces.append(chunk)
        tok_cancel.set()
    assert len(tok_pieces) == 1

    out_path = tmp_path / "saved.wav"
    audio = tts("Hello there.", seed=11)
    tts.save(str(out_path), audio)
    assert out_path.exists()
    assert len(tts.synthesize_wav_bytes("Ready.")) > 44

    class _RunawayThenOkBackend(_TinyTtsBackend):
        def __init__(self) -> None:
            super().__init__()
            self._attempt = 0

        def synthesize_raw(
            self,
            text: str,
            *,
            temp: float = 0.5,
            eos: float = 0.9,
        ) -> tuple[np.ndarray, int]:
            del temp, eos
            self._attempt += 1
            self.calls.append(text)
            if self._attempt == 1:
                return np.full(4800, 0.2, dtype=np.float32), self.max_frames
            return np.full(800, 0.2, dtype=np.float32), 4

    retry_backend = _RunawayThenOkBackend()
    retry_tts = TextToSpeech(backend=retry_backend, retries=1, sanitize_markdown=False)
    recovered = retry_tts("Raw *input*.", sanitize=False)
    assert len(recovered) == 800
    assert len(retry_backend.calls) == 2
    assert list(retry_tts.stream("One.", sanitize=False))


def test_stt_backend_validation_models_dir_masks_padding_and_async(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Covers OnnxSttBackend validation, models_dir, attention masks, and padding."""
    with pytest.raises(ValueError, match="Unsupported"):
        OnnxSttBackend(model_name="invalid-model")

    with pytest.raises(ValueError, match="precision"):
        OnnxSttBackend(model_name="tiny", precision="bf16")

    models_dir = tmp_path / "moonshine-tiny"
    models_dir.mkdir(parents=True)
    for fname in ("encoder_model.onnx", "decoder_model_merged.onnx"):
        (models_dir / fname).write_bytes(b"ONNX")
    (models_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hi": 3},
                    "merges": [],
                },
            }
        ),
        encoding="utf-8",
    )

    class _FakeSession:
        def __init__(self, path: str, **_kwargs: Any) -> None:
            self._is_encoder = "encoder" in Path(path).name
            self.seen_keys: list[set[str]] = []

        def get_inputs(self) -> list[Any]:
            if self._is_encoder:
                return [
                    SimpleNamespace(name="input_values"),
                    SimpleNamespace(name="attention_mask"),
                ]
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="encoder_hidden_states"),
                SimpleNamespace(name="encoder_attention_mask"),
                SimpleNamespace(name="use_cache_branch"),
                SimpleNamespace(name="past_key_values.0.decoder.key"),
                SimpleNamespace(name="past_key_values.0.decoder.value"),
                SimpleNamespace(name="past_key_values.0.encoder.key"),
                SimpleNamespace(name="past_key_values.0.encoder.value"),
            ]

        def get_outputs(self) -> list[Any]:
            if self._is_encoder:
                return [SimpleNamespace(name="last_hidden_state")]
            return [
                SimpleNamespace(name="logits"),
                SimpleNamespace(name="present.0.decoder.key"),
                SimpleNamespace(name="present.0.decoder.value"),
                SimpleNamespace(name="present.0.encoder.key"),
                SimpleNamespace(name="present.0.encoder.value"),
            ]

        def run(self, _out: Any, feed: dict[str, Any]) -> list[np.ndarray]:
            self.seen_keys.append(set(feed.keys()))
            if self._is_encoder:
                return [np.zeros((1, 4, 8), dtype=np.float32)]
            step = len(self.seen_keys)
            logits = np.full((1, 1, 4), -10.0, dtype=np.float32)
            logits[0, 0, 3 if step == 1 else 2] = 10.0
            kv = np.zeros((1, 1, 1, 2), dtype=np.float32)
            return [logits, kv, kv, kv, kv]

    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    onnx_backend = OnnxSttBackend(model_name="tiny", models_dir=str(models_dir))
    assert onnx_backend.sample_rate == STT_SAMPLE_RATE
    assert onnx_backend.model_name == "tiny"
    text, num_tokens = onnx_backend.transcribe_pcm(
        np.full(STT_SAMPLE_RATE // 2, 0.25, dtype=np.float32)
    )
    assert isinstance(text, str)
    assert num_tokens >= 1

    batched_onnx = np.full((2, 1600), 0.2, dtype=np.float32)
    text_2d, _ = onnx_backend.transcribe_pcm(batched_onnx)
    assert isinstance(text_2d, str)

    tiny_stt_backend = _TinySttBackend()
    stt = SpeechToText(backend=tiny_stt_backend)

    batched = np.full((2, 4800), 0.2, dtype=np.float32)
    res_2d = stt.transcribe_array(batched, sample_rate=STT_SAMPLE_RATE)
    assert res_2d.text.startswith("samples=")

    t_short = np.linspace(0.0, 0.2, 3200, endpoint=False, dtype=np.float32)
    short_pcm = (0.25 * np.sin(2.0 * np.pi * 220.0 * t_short)).astype(np.float32)
    res_short = stt.transcribe_array(short_pcm, sample_rate=STT_SAMPLE_RATE)
    assert tiny_stt_backend.seen_lengths[-1] == int(0.25 * STT_SAMPLE_RATE)
    assert res_short.num_tokens == 2

    wav_file = tmp_path / "short.wav"
    wav_file.write_bytes(to_wav_bytes(batched[0], sample_rate=STT_SAMPLE_RATE))

    async def _run_async_stt() -> None:
        a1 = await stt.atranscribe_array(batched[0], sample_rate=STT_SAMPLE_RATE)
        a2 = await stt.atranscribe_file(str(wav_file))
        assert a1.num_tokens == 2
        assert a2.num_tokens == 2

    asyncio.run(_run_async_stt())


def test_voice_io_lazy_props_interrupt_sync_async_streams_and_player_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Covers VoiceIO lazy factories, active recorder stop, and async playback streams."""
    import voice_io.session as session_mod

    tts_obj = TextToSpeech(backend=_TinyTtsBackend(), retries=0)
    stt_obj = SpeechToText(backend=_TinySttBackend())
    spy_recorder = _RecordingSpy(active=True)

    monkeypatch.setattr(session_mod, "TextToSpeech", lambda **_kw: tts_obj)
    monkeypatch.setattr(session_mod, "SpeechToText", lambda **_kw: stt_obj)
    monkeypatch.setattr(session_mod, "AudioRecorder", lambda: spy_recorder)

    lazy_io = VoiceIO()
    assert lazy_io.tts is tts_obj
    assert lazy_io.stt is stt_obj
    assert lazy_io.recorder is spy_recorder

    played: list[int] = []
    write_returns = [True, False]

    class _PlayerStub:
        def __init__(self, sample_rate: int = 16000) -> None:
            del sample_rate

        def __enter__(self) -> "_PlayerStub":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def write(self, chunk: bytes) -> bool:
            played.append(len(chunk))
            if write_returns:
                return write_returns.pop(0)
            return False

        def stop(self) -> None:
            pass

        def play_stream(
            self,
            chunks: Iterator[np.ndarray],
            *,
            cancel_event: threading.Event | None = None,
        ) -> tuple[int, float | None, bool]:
            total = 0
            for chunk in chunks:
                if cancel_event is not None and cancel_event.is_set():
                    return total, 0.01, False
                total += len(chunk)
                if not self.write(to_pcm16(chunk)):
                    return total, 0.01, False
            return total, 0.01, True

    monkeypatch.setattr(session_mod, "AudioPlayer", _PlayerStub)

    wav_file = tmp_path / "voice.wav"
    wav_file.write_bytes(
        to_wav_bytes(
            np.full(STT_SAMPLE_RATE // 4, 0.2, dtype=np.float32),
            sample_rate=STT_SAMPLE_RATE,
        )
    )
    assert lazy_io.transcribe(str(wav_file)).num_tokens == 2
    assert lazy_io.listen(duration_s=0.1).num_tokens == 2
    assert lazy_io.listen().num_tokens == 2

    audio_out = lazy_io.speak("Hello there.", play=True)
    assert len(audio_out) > 0
    assert spy_recorder.stopped is True

    lazy_io.interrupt()

    def _toks() -> Iterator[str]:
        yield "Hello "
        yield "world. "
        yield "Again."

    streamed = lazy_io.speak_stream(_toks(), play=False)
    assert len(streamed) >= 1600

    write_returns.clear()
    write_returns.extend([True, False, True])

    io2 = VoiceIO(tts=tts_obj, stt=stt_obj, recorder=_RecordingSpy())
    cancel_ev = threading.Event()

    def _cancel_mid_stream() -> Iterator[str]:
        yield "One sentence. "
        cancel_ev.set()
        yield "Two sentence."

    partial = io2.speak_stream(_cancel_mid_stream(), play=False, cancel_event=cancel_ev)
    assert len(partial) == 800

    async def _run_async_voice_io() -> None:
        io3 = VoiceIO(tts=tts_obj, stt=stt_obj, recorder=_RecordingSpy())
        spoken = await io3.aspeak("Async line.", play=True)
        assert len(spoken) > 0

        async def _async_toks() -> AsyncIterator[str]:
            yield "Alpha "
            yield "beta. "
            yield "Gamma "
            yield "delta."

        write_returns.clear()
        write_returns.extend([True, False, True])
        async_audio = await io3.aspeak_stream(_async_toks(), play=True)
        assert len(async_audio) >= 800

        write_returns.clear()
        write_returns.extend([True, True, True])
        sync_in_async = await io3.aspeak_stream(["One.", " Two."], play=True)
        assert len(sync_in_async) >= 800

        io3.interrupt()
        async_cancel_ev = threading.Event()
        async_cancel_ev.set()
        cancelled_async = await io3.aspeak_stream(
            _async_toks(), play=False, cancel_event=async_cancel_ev
        )
        assert len(cancelled_async) == 0

        silent_async = await io3.aspeak("Quiet async.", play=False)
        assert len(silent_async) > 0
        assert (await io3.alisten(duration_s=0.1)).num_tokens == 2
        assert (await io3.atranscribe(str(wav_file))).num_tokens == 2

        backend_io = VoiceIO(
            stt_backend=_TinySttBackend(),
            tts_backend=_TinyTtsBackend(),
            recorder=_RecordingSpy(),
        )
        assert backend_io.stt is not None
        assert backend_io.tts is not None

        assert len(io3.speak("", play=True)) == 0
        assert len(await io3.aspeak("", play=False)) == 0
        wav_bytes = to_wav_bytes(
            np.full(STT_SAMPLE_RATE // 4, 0.2, dtype=np.float32),
            sample_rate=STT_SAMPLE_RATE,
        )
        assert io3.transcribe(wav_bytes).num_tokens == 2
        assert (
            await io3.atranscribe(
                np.full(STT_SAMPLE_RATE // 4, 0.2, dtype=np.float32),
                sample_rate=STT_SAMPLE_RATE,
            )
        ).num_tokens == 2
        assert (await io3.alisten()).num_tokens == 2

        chunk_cancel = threading.Event()

        class _TwoChunkCancelTts(TextToSpeech):
            def stream_tokens(
                self,
                tokens: Iterable[str],
                *,
                seed: int | None = None,
                cancel_event: threading.Event | None = None,
            ) -> Iterator[np.ndarray]:
                del tokens, seed, cancel_event
                yield np.full(800, 0.2, dtype=np.float32)
                chunk_cancel.set()
                yield np.full(800, 0.2, dtype=np.float32)

        fake_tts = _TwoChunkCancelTts(backend=_TinyTtsBackend(), retries=0)
        io_chunk_cancel = VoiceIO(tts=fake_tts, stt=stt_obj, recorder=_RecordingSpy())
        half = io_chunk_cancel.speak_stream(
            ["a", "b"], play=False, cancel_event=chunk_cancel
        )
        assert len(half) == 800

    asyncio.run(_run_async_voice_io())


def test_tts_textsplit_and_voice_io_additional_edge_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers remaining TTS, textsplit, and async VoiceIO interrupt/cancel branches."""
    # _resolve_weights_path env override and package-local weights path
    monkeypatch.setenv("VOICE_IO_WEIGHTS", "/tmp/custom-weights.npz")
    assert _resolve_weights_path() == "/tmp/custom-weights.npz"
    monkeypatch.delenv("VOICE_IO_WEIGHTS", raising=False)
    monkeypatch.delenv("WOOF_WEIGHTS", raising=False)
    orig_exists = os.path.exists
    with monkeypatch.context() as mp:
        mp.setattr(
            os.path,
            "exists",
            lambda path: (
                True
                if str(path).endswith("voice_io/woof_int8.npz")
                else orig_exists(path)
            ),
        )
        assert _resolve_weights_path().endswith("voice_io/woof_int8.npz")

    # TextToSpeech._stall_duration with fewer than two voiced frames
    assert TextToSpeech._stall_duration(np.zeros(4000, dtype=np.float32)) == 0.0

    # TextToSpeech.stream_tokens pre-cancelled and cancelled right before flush
    tts = TextToSpeech(backend=_TinyTtsBackend(), sanitize_markdown=False)
    pre_cancel = threading.Event()
    pre_cancel.set()
    assert list(tts.stream_tokens(["Hello."], cancel_event=pre_cancel)) == []

    flush_cancel = threading.Event()

    def _tok_cancel_before_flush() -> Iterator[str]:
        yield "Unpunctuated remainder"
        flush_cancel.set()

    assert (
        list(tts.stream_tokens(_tok_cancel_before_flush(), cancel_event=flush_cancel))
        == []
    )

    # TextToSpeech.astream pre-cancelled and mid-stream cancelled
    class _MidCancelTtsBackend(_TinyTtsBackend):
        def __init__(self, event: threading.Event) -> None:
            super().__init__()
            self._event = event

        def synthesize_raw(
            self,
            text: str,
            *,
            temp: float = 0.5,
            eos: float = 0.9,
        ) -> tuple[np.ndarray, int]:
            self._event.set()
            return super().synthesize_raw(text, temp=temp, eos=eos)

    mid_cancel = threading.Event()
    tts_mid = TextToSpeech(
        backend=_MidCancelTtsBackend(mid_cancel),
        sanitize_markdown=False,
    )

    async def _run_astream_edges() -> None:
        assert [
            x async for x in tts_mid.astream("Hello.", cancel_event=pre_cancel)
        ] == []
        chunks = [
            x
            async for x in tts_mid.astream(
                "First sentence. Second sentence.",
                cancel_event=mid_cancel,
            )
        ]
        assert len(chunks) == 1

    asyncio.run(_run_astream_edges())

    # TextSplitter conjunction splitting, first_max cuts, and em-dash merging
    conj_pieces = TextSplitter(clause_max=70, first_max=0).split(
        "This first clause talks for thirty five chars "
        "and continues for thirty five more chars."
    )
    assert len(conj_pieces) == 2

    assert (
        len(
            TextSplitter(clause_max=200, first_max=40).split(
                'Short "quote," tail words stay together for much longer now.'
            )
        )
        == 1
    )
    first_inside = TextSplitter(clause_max=200, first_max=45).split(
        "Opening clause fits cleanly, followed by twenty more trailing words."
    )
    assert len(first_inside) == 2
    first_past = TextSplitter(clause_max=200, first_max=22).split(
        "Opening clause exceeds twenty two chars, and continues after the comma."
    )
    assert len(first_past) == 2

    dash_merged = TextSplitter(clause_max=80, first_max=0).split(
        "Alpha clause--beta continues; " + " ".join(["more"] * 14) + "."
    )
    assert dash_merged[0].startswith("Alpha clause--beta continues;")

    # StreamingSentenceBuffer paragraph boundary, empty delta, and open-fence flush
    buf = StreamingSentenceBuffer(min_chars=8)
    assert buf.push("") == []
    assert buf.push("First paragraph\n\nSecond line") == ["First paragraph"]

    fence_buf = StreamingSentenceBuffer(min_chars=12)
    assert fence_buf.push("Lead sentence\n```python\nprint(1)") == []
    assert fence_buf.flush() == ["Lead sentence"]

    # VoiceIO.interrupt during active playback + aspeak_stream branches
    stt_obj = SpeechToText(backend=_TinySttBackend())

    class _FakePlayer:
        last_stopped = False

        def __init__(self, *, sample_rate: int = 16000) -> None:
            del sample_rate
            _FakePlayer.last_stopped = False

        def __enter__(self) -> "_FakePlayer":
            return self

        def __exit__(self, *args: Any) -> None:
            del args

        def stop(self) -> None:
            _FakePlayer.last_stopped = True

        def write(self, _pcm: bytes) -> bool:
            active_io.interrupt()
            return False

        def play_stream(
            self,
            chunks: Iterable[np.ndarray],
            *,
            cancel_event: threading.Event | None = None,
        ) -> bool:
            for chunk in chunks:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                if not self.write(to_pcm16(chunk)):
                    return False
            return True

    monkeypatch.setattr("voice_io.session.AudioPlayer", _FakePlayer)
    active_io = VoiceIO(
        tts=TextToSpeech(backend=_TinyTtsBackend(), sanitize_markdown=False),
        stt=stt_obj,
        recorder=_RecordingSpy(),
    )
    interrupted = active_io.speak("Cancel me cleanly.", play=True)
    assert len(interrupted) > 0
    assert _FakePlayer.last_stopped is True

    async def _run_aspeak_stream_branches() -> None:
        io_a = VoiceIO(
            tts=TextToSpeech(backend=_TinyTtsBackend(), sanitize_markdown=False),
            stt=stt_obj,
            recorder=_RecordingSpy(),
        )

        async def _two_sent_async() -> AsyncIterator[str]:
            yield "Alpha sentence here! Beta sentence here! "

        seeded = await io_a.aspeak_stream(_two_sent_async(), seed=7, play=False)
        assert len(seeded) >= 1600

        # Cancel during sentence 1 chunk emission with 2 chunks -> hits mid-chunk check
        ev_chunk = threading.Event()

        class _WriteCancelPlayer(_FakePlayer):
            def write(self, _pcm: bytes) -> bool:
                ev_chunk.set()
                return True

        monkeypatch.setattr("voice_io.session.AudioPlayer", _WriteCancelPlayer)
        monkeypatch.setattr(
            io_a.tts,
            "stream",
            lambda _s, **_kw: iter(
                [
                    np.full(400, 0.2, dtype=np.float32),
                    np.full(400, 0.2, dtype=np.float32),
                ]
            ),
        )
        chunk_res = await io_a.aspeak_stream(
            _two_sent_async(), play=True, cancel_event=ev_chunk
        )
        assert len(chunk_res) == 400

        # Cancel during empty-chunk sentence 1 -> hits entry turn_cancel check on sentence 2
        ev_entry = threading.Event()
        calls = 0

        def _empty_first_then_cancel(
            _sentence: str, **_kw: Any
        ) -> Iterator[np.ndarray]:
            nonlocal calls
            calls += 1
            if calls == 1:
                ev_entry.set()
                return iter(())
            return iter([np.full(400, 0.2, dtype=np.float32)])

        monkeypatch.setattr(io_a.tts, "stream", _empty_first_then_cancel)
        entry_res = await io_a.aspeak_stream(
            _two_sent_async(), play=False, cancel_event=ev_entry
        )
        assert len(entry_res) == 0

    asyncio.run(_run_aspeak_stream_branches())
