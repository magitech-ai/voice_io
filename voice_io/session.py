"""Unified bidirectional voice I/O session facade for AI agents."""

import asyncio
import os
import threading
from collections.abc import AsyncIterable, Callable, Iterable

import numpy as np

from .audio import AudioPlayer, AudioProcessor, AudioRecorder
from .backends import SttBackend, TtsBackend
from .stt import SpeechToText, TranscriptionResult
from .textsplit import SpeechTextSanitizer, StreamingSentenceBuffer
from .tts import TextToSpeech


class VoiceIO:
    """Bidirectional STT + TTS orchestrator for conversational AI agents.

    Coordinates microphone capture (VAD endpointing or fixed duration), offline
    speech recognition, Markdown-aware text-to-speech synthesis, speaker
    playback, barge-in cancellation, and half-duplex echo avoidance.
    """

    def __init__(
        self,
        *,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        stt_backend: SttBackend | None = None,
        tts_backend: TtsBackend | None = None,
        recorder: AudioRecorder | None = None,
        stt_model: str = "base",
        voice: str = "juno",
        sanitize_markdown: bool = True,
    ) -> None:
        """Initializes a VoiceIO session with lazy-loaded STT and TTS engines."""
        self._stt = stt
        self._tts = tts
        self._stt_backend = stt_backend
        self._tts_backend = tts_backend
        self._recorder = recorder or AudioRecorder()
        self._stt_model = stt_model
        self._voice = voice
        self._sanitize_markdown = sanitize_markdown

        self._init_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._active_player: AudioPlayer | None = None

    @property
    def stt(self) -> SpeechToText:
        """Returns the lazily constructed :class:`SpeechToText` recognizer."""
        with self._init_lock:
            if self._stt is None:
                self._stt = SpeechToText(
                    backend=self._stt_backend, model_name=self._stt_model
                )
            return self._stt

    @property
    def tts(self) -> TextToSpeech:
        """Returns the lazily constructed :class:`TextToSpeech` synthesizer."""
        with self._init_lock:
            if self._tts is None:
                self._tts = TextToSpeech(
                    backend=self._tts_backend,
                    voice=self._voice,
                    sanitize_markdown=self._sanitize_markdown,
                )
            return self._tts

    @property
    def recorder(self) -> AudioRecorder:
        """Returns the session's :class:`AudioRecorder`."""
        return self._recorder

    def interrupt(self) -> None:
        """Immediately cancels active TTS generation and speaker playback."""
        with self._playback_lock:
            cancel_event = self._active_cancel
            player = self._active_player
        if cancel_event is not None:
            cancel_event.set()
        if player is not None:
            player.stop()

    def transcribe(
        self,
        source: bytes | str | os.PathLike[str] | np.ndarray,
        *,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribes audio bytes, a file path, or a waveform array."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            return self.stt.transcribe_bytes(bytes(source))
        if isinstance(source, (str, os.PathLike)):
            return self.stt.transcribe_file(os.fspath(source))
        return self.stt.transcribe_array(source, sample_rate=sample_rate)

    def listen(
        self,
        *,
        duration_s: float | None = None,
        energy_threshold: float = 0.018,
        silence_duration_s: float = 0.9,
        wait_timeout_s: float = 15.0,
        max_duration_s: float = 30.0,
        cancel_event: threading.Event | None = None,
        on_level: Callable[[float, str], None] | None = None,
    ) -> TranscriptionResult:
        """Records a user turn from the microphone and transcribes it.

        When ``duration_s`` is omitted, uses VAD endpointing to stop automatically
        after trailing silence.
        """
        if duration_s is not None and duration_s > 0:
            audio = self._recorder.record_seconds(duration_s)
        else:
            audio = self._recorder.listen_utterance(
                energy_threshold=energy_threshold,
                silence_duration_s=silence_duration_s,
                wait_timeout_s=wait_timeout_s,
                max_duration_s=max_duration_s,
                cancel_event=cancel_event,
                on_level=on_level,
            )
        return self.stt.transcribe_array(audio, sample_rate=self.stt.sample_rate)

    def _prepare_turn_playback(
        self, cancel_event: threading.Event | None
    ) -> tuple[threading.Event, AudioPlayer]:
        """Halts any active capture/playback and creates a new playback session."""
        if self._recorder.is_recording:
            self._recorder.stop()
        turn_cancel = cancel_event or threading.Event()
        player = AudioPlayer(sample_rate=self.tts.sample_rate)
        with self._playback_lock:
            self._active_cancel = turn_cancel
            self._active_player = player
        return turn_cancel, player

    def _clear_turn_playback(self, player: AudioPlayer) -> None:
        """Clears active playback references after a turn ends."""
        with self._playback_lock:
            if self._active_player is player:
                self._active_player = None
                self._active_cancel = None

    def _run_chunk_stream(
        self,
        chunks: Iterable[np.ndarray],
        *,
        play: bool,
        cancel_event: threading.Event | None,
    ) -> np.ndarray:
        """Consumes synthesized audio chunks, optionally playing them live."""
        collected: list[np.ndarray] = []

        def _tee() -> Iterable[np.ndarray]:
            for chunk in chunks:
                collected.append(chunk)
                yield chunk

        if not play:
            for chunk in chunks:
                if cancel_event is not None and cancel_event.is_set():
                    break
                collected.append(chunk)
        else:
            turn_cancel, player = self._prepare_turn_playback(cancel_event)
            try:
                player.play_stream(_tee(), cancel_event=turn_cancel)
            finally:
                self._clear_turn_playback(player)

        if not collected:
            return np.zeros(0, dtype=np.float32)
        return collected[0] if len(collected) == 1 else np.concatenate(collected)

    def speak(
        self,
        text: str,
        *,
        play: bool = True,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        """Synthesizes ``text`` and optionally streams it to speakers."""
        chunks = self.tts.stream(text, seed=seed, cancel_event=cancel_event)
        return self._run_chunk_stream(chunks, play=play, cancel_event=cancel_event)

    def speak_stream(
        self,
        tokens: Iterable[str],
        *,
        play: bool = True,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        """Streams LLM text deltas directly into speech synthesis and playback."""
        chunks = self.tts.stream_tokens(tokens, seed=seed, cancel_event=cancel_event)
        return self._run_chunk_stream(chunks, play=play, cancel_event=cancel_event)

    async def atranscribe(
        self,
        source: bytes | str | os.PathLike[str] | np.ndarray,
        *,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Asynchronously transcribes audio bytes, a file path, or a waveform."""
        return await asyncio.to_thread(self.transcribe, source, sample_rate=sample_rate)

    async def alisten(
        self,
        *,
        duration_s: float | None = None,
        energy_threshold: float = 0.018,
        silence_duration_s: float = 0.9,
        wait_timeout_s: float = 15.0,
        max_duration_s: float = 30.0,
        cancel_event: threading.Event | None = None,
        on_level: Callable[[float, str], None] | None = None,
    ) -> TranscriptionResult:
        """Asynchronously records and transcribes one spoken turn."""
        return await asyncio.to_thread(
            self.listen,
            duration_s=duration_s,
            energy_threshold=energy_threshold,
            silence_duration_s=silence_duration_s,
            wait_timeout_s=wait_timeout_s,
            max_duration_s=max_duration_s,
            cancel_event=cancel_event,
            on_level=on_level,
        )

    async def aspeak(
        self,
        text: str,
        *,
        play: bool = True,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        """Asynchronously synthesizes ``text`` and optionally plays it."""
        return await asyncio.to_thread(
            self.speak,
            text,
            play=play,
            seed=seed,
            cancel_event=cancel_event,
        )

    async def aspeak_stream(
        self,
        tokens: Iterable[str] | AsyncIterable[str],
        *,
        play: bool = True,
        seed: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        """Consumes sync or async LLM token streams cleanly without blocking."""
        if not isinstance(tokens, AsyncIterable):
            return await asyncio.to_thread(
                self.speak_stream,
                tokens,
                play=play,
                seed=seed,
                cancel_event=cancel_event,
            )

        if seed is not None:
            self.tts.backend.seed(seed)

        sanitizer = SpeechTextSanitizer() if self._sanitize_markdown else None
        buf = StreamingSentenceBuffer(sanitizer=sanitizer)
        collected: list[np.ndarray] = []

        turn_cancel, player = (
            self._prepare_turn_playback(cancel_event)
            if play
            else (cancel_event or threading.Event(), None)
        )

        async def _speak_sentence(sentence: str) -> bool:
            if turn_cancel.is_set():
                return False
            chunks = await asyncio.to_thread(
                lambda: list(
                    self.tts.stream(
                        sentence,
                        seed=None,
                        cancel_event=turn_cancel,
                        sanitize=False,
                    )
                )
            )
            for wav in chunks:
                if turn_cancel.is_set():
                    return False
                collected.append(wav)
                if player is not None:
                    ok = await asyncio.to_thread(
                        player.write, AudioProcessor.to_pcm16(wav)
                    )
                    if not ok:
                        return False
            return True

        try:
            if player is not None:
                player.__enter__()
            async for delta in tokens:
                if turn_cancel.is_set():
                    break
                for sentence in buf.push(delta):
                    if not await _speak_sentence(sentence):
                        break
            if not turn_cancel.is_set():
                for remainder in buf.flush():
                    if not await _speak_sentence(remainder):
                        break
        finally:
            if player is not None:
                try:
                    player.__exit__(None, None, None)
                finally:
                    self._clear_turn_playback(player)

        if not collected:
            return np.zeros(0, dtype=np.float32)
        return collected[0] if len(collected) == 1 else np.concatenate(collected)
