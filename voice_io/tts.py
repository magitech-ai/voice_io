"""High-level TextToSpeech SDK with pluggable backend and LLM streaming."""

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Iterable, Iterator

import numpy as np

from .audio import TTS_SAMPLE_RATE, AudioProcessor
from .backends import TtsBackend
from .normalize import Normalizer
from .runtime import LocalInt8FlowBackend
from .textsplit import (
    SpeechTextSanitizer,
    StreamingSentenceBuffer,
    TextSplitter,
)

VOICES = ("juno", "mira")


def _resolve_weights_path() -> str:
    """Resolves default int8 TTS weights from env, package dir, or repo root."""
    env_path = os.environ.get("VOICE_IO_WEIGHTS") or os.environ.get("WOOF_WEIGHTS")
    if env_path:
        return env_path
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_weights = os.path.join(pkg_dir, "woof_int8.npz")
    if os.path.exists(pkg_weights):
        return pkg_weights
    return os.path.join(os.path.dirname(pkg_dir), "woof_int8.npz")


DEFAULT_WEIGHTS_PATH = _resolve_weights_path()


class TextToSpeech:
    """English text-to-speech synthesizer with Markdown cleanup and streaming.

    Supports swapping synthesis engines cleanly through the :class:`TtsBackend`
    protocol while keeping normalization, punctuation completion, clause
    splitting, stall retries, and LLM token-stream buffering consistent.
    """

    def __init__(
        self,
        *,
        backend: TtsBackend | None = None,
        weights: str = DEFAULT_WEIGHTS_PATH,
        voice: str = "juno",
        temp: float = 0.5,
        eos: float = 0.9,
        max_spc: float = 0.095,
        stall_s: float = 1.0,
        retries: int = 3,
        normalize: bool = True,
        sanitize_markdown: bool = True,
        code_block_cue: str = "",
    ) -> None:
        """Initializes the TextToSpeech synthesizer.

        Args:
          backend: Optional custom :class:`TtsBackend` instance. Defaults to
            :class:`LocalInt8FlowBackend`.
          weights: Weights archive path used when creating the default backend.
          voice: Speaker voice identifier when creating the default backend.
          temp: Sampling temperature passed to the backend.
          eos: End-of-speech probability threshold passed to the backend.
          max_spc: Maximum allowed seconds per character before retrying.
          stall_s: Maximum allowed internal silence (seconds) before retrying.
          retries: Maximum number of redraws for a stalled/runaway piece.
          normalize: Whether to run spoken-form normalization on text.
          sanitize_markdown: Whether to strip Markdown syntax before normalization.
          code_block_cue: Optional spoken phrase replacing fenced code blocks.
        """
        self.backend: TtsBackend = (
            backend
            if backend is not None
            else LocalInt8FlowBackend(weights, voice=voice)
        )
        self.norm = Normalizer(use_nemo=False) if normalize else None
        self._sanitizer = (
            SpeechTextSanitizer(code_block_cue=code_block_cue)
            if sanitize_markdown
            else None
        )
        self._splitter = TextSplitter()
        self.temp = temp
        self.eos = eos
        self.max_spc = max_spc
        self.stall_s = stall_s
        self.retries = retries
        self.sample_rate = self.backend.sample_rate
        self.sr = self.sample_rate

    @staticmethod
    def _ensure_terminal_punct(text: str) -> str:
        """Appends a period when non-empty text lacks terminal punctuation."""
        stripped = text.rstrip()
        if not stripped:
            return text
        core = stripped.rstrip("\"'”’)]")
        if not core or core.endswith((".", "!", "?", "…", ",", ";", ":", "-", "—")):
            return text
        return stripped + "."

    @staticmethod
    def _stall_duration(
        wav: np.ndarray,
        *,
        sample_rate: int = TTS_SAMPLE_RATE,
        hop: int = 960,
        db: float = -50.0,
    ) -> float:
        """Returns longest internal silence run (seconds) between voiced frames."""
        num_frames = len(wav) // hop
        if num_frames < 3:
            return 0.0
        rms = np.sqrt((wav[: num_frames * hop].reshape(num_frames, hop) ** 2).mean(1))
        quiet = 20 * np.log10(rms + 1e-9) < db
        voiced = np.nonzero(~quiet)[0]
        if len(voiced) < 2:
            return 0.0
        best = 0
        current = 0
        for is_quiet in quiet[voiced[0] : voiced[-1] + 1]:
            current = current + 1 if is_quiet else 0
            best = max(best, current)
        return best * hop / sample_rate

    def _prepare_text(self, text: str, *, sanitize: bool | None) -> str:
        """Applies optional Markdown cleanup, spoken normalization, and punctuation."""
        do_sanitize = self._sanitizer is not None if sanitize is None else sanitize
        prepared = (
            self._sanitizer.clean(text)
            if do_sanitize and self._sanitizer is not None
            else text
        )
        if self.norm is not None:
            prepared = self.norm(prepared)
        return self._ensure_terminal_punct(prepared)

    def _synthesize_piece(self, text: str) -> np.ndarray:
        """Synthesizes a single utterance piece, retrying on stalls or runaways."""
        cap = int(self.backend.max_frames)
        best: tuple[tuple[int, float, float], np.ndarray] | None = None
        for _ in range(1 + self.retries):
            wav, frame_count = self.backend.synthesize_raw(
                text, temp=self.temp, eos=self.eos
            )
            run = self._stall_duration(wav, sample_rate=self.sample_rate)
            dur = len(wav) / self.sample_rate
            spc = dur / max(1, len(text))
            too_long = (spc > self.max_spc and dur > 2.5) or (
                cap > 0 and frame_count >= cap
            )
            bad = int(run >= self.stall_s) + int(too_long)
            score = (bad, run, spc)
            if best is None or score < best[0]:
                best = (score, wav)
            if not bad:
                break
        assert best is not None
        return best[1]

    def stream(
        self,
        text: str,
        *,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
        sanitize: bool | None = None,
    ) -> Iterator[np.ndarray]:
        """Yields mono ``float32`` audio chunks one sentence or clause at a time."""
        if cancel_event is not None and cancel_event.is_set():
            return
        if seed is not None:
            self.backend.seed(seed)
        prepared = self._prepare_text(text, sanitize=sanitize)
        if not prepared.strip():
            return
        pieces = self._splitter.split(prepared) or [prepared]
        for piece in pieces:
            if cancel_event is not None and cancel_event.is_set():
                break
            yield self._synthesize_piece(piece).astype(np.float32, copy=False)

    def stream_tokens(
        self,
        tokens: Iterable[str],
        *,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[np.ndarray]:
        """Synthesizes incremental audio directly from streamed LLM text deltas."""
        if cancel_event is not None and cancel_event.is_set():
            return
        if seed is not None:
            self.backend.seed(seed)
        buf = StreamingSentenceBuffer(sanitizer=self._sanitizer)
        for delta in tokens:
            if cancel_event is not None and cancel_event.is_set():
                return
            for sentence in buf.push(delta):
                yield from self.stream(
                    sentence,
                    seed=None,
                    cancel_event=cancel_event,
                    sanitize=False,
                )
                if cancel_event is not None and cancel_event.is_set():
                    return
        for remainder in buf.flush():
            if cancel_event is not None and cancel_event.is_set():
                return
            yield from self.stream(
                remainder,
                seed=None,
                cancel_event=cancel_event,
                sanitize=False,
            )

    def __call__(
        self,
        text: str,
        *,
        seed: int | None = None,
        sanitize: bool | None = None,
    ) -> np.ndarray:
        """Synthesizes ``text`` into a single 1D mono ``float32`` waveform array."""
        chunks = list(self.stream(text, seed=seed, sanitize=sanitize))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def synthesize_wav_bytes(
        self,
        text: str,
        *,
        seed: int | None = None,
        sanitize: bool | None = None,
    ) -> bytes:
        """Synthesizes ``text`` directly into in-memory 16-bit PCM WAV bytes."""
        wav = self(text, seed=seed, sanitize=sanitize)
        return AudioProcessor.to_wav_bytes(wav, sample_rate=self.sample_rate)

    def save(self, path: str, wav: np.ndarray) -> None:
        """Saves a ``float32`` waveform array to a 16-bit PCM WAV file."""
        AudioProcessor.write_wav(path, wav, sample_rate=self.sample_rate)

    async def asynthesize(
        self,
        text: str,
        *,
        seed: int | None = None,
        sanitize: bool | None = None,
    ) -> np.ndarray:
        """Asynchronously synthesizes ``text`` in a worker thread."""
        return await asyncio.to_thread(
            self.__call__, text, seed=seed, sanitize=sanitize
        )

    async def astream(
        self,
        text: str,
        *,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
        sanitize: bool | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """Asynchronously yields waveform chunks without blocking the event loop."""
        if cancel_event is not None and cancel_event.is_set():
            return
        if seed is not None:
            self.backend.seed(seed)
        prepared = await asyncio.to_thread(self._prepare_text, text, sanitize=sanitize)
        if not prepared.strip():
            return
        pieces = self._splitter.split(prepared) or [prepared]
        for piece in pieces:
            if cancel_event is not None and cancel_event.is_set():
                break
            wav = await asyncio.to_thread(self._synthesize_piece, piece)
            yield wav.astype(np.float32, copy=False)
