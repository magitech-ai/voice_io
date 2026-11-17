"""Audio preprocessing, playback with barge-in, and VAD microphone capture."""

import collections.abc
import io
import queue
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from types import TracebackType

import numpy as np

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000


class AudioProcessor:
    """Audio decoding, resampling, serialization, chunking, and RMS metrics."""

    @staticmethod
    def to_pcm16(audio: np.ndarray) -> bytes:
        """Converts 1D float32 audio in ``[-1.0, 1.0]`` to little-endian PCM16."""
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        return (np.clip(arr, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()

    @classmethod
    def to_wav_bytes(
        cls, audio: np.ndarray, sample_rate: int = TTS_SAMPLE_RATE
    ) -> bytes:
        """Serializes mono float32 waveform samples into in-memory PCM16 WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(cls.to_pcm16(audio))
        return buf.getvalue()

    @classmethod
    def write_wav(
        cls, path: str, audio: np.ndarray, sample_rate: int = TTS_SAMPLE_RATE
    ) -> None:
        """Writes mono float32 waveform samples to a 16-bit PCM WAV file."""
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(cls.to_pcm16(audio))

    @staticmethod
    def resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int = STT_SAMPLE_RATE,
    ) -> np.ndarray:
        """Resamples 1D float32 audio from ``orig_sr`` to ``target_sr``.

        Uses FFT spectral resampling for brick-wall anti-aliasing without SciPy.

        Args:
          audio: 1D mono waveform array.
          orig_sr: Input sample rate in Hz.
          target_sr: Output sample rate in Hz.

        Returns:
          Resampled 1D ``np.float32`` array.
        """
        arr = np.asarray(audio, dtype=np.float32)
        if orig_sr <= 0 or target_sr <= 0:
            raise ValueError("Sample rates must be positive integers.")
        if orig_sr == target_sr or arr.size == 0:
            return arr
        target_len = max(1, int(round(arr.size * target_sr / orig_sr)))
        if arr.size < 4 or target_len < 4:
            x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
            return np.interp(x_new, x_old, arr).astype(np.float32)

        spectrum = np.fft.rfft(arr.astype(np.float64))
        new_bins = target_len // 2 + 1
        if new_bins <= spectrum.size:
            resized = spectrum[:new_bins].copy()
            if target_len % 2 == 0 and resized.size > 0:
                resized[-1] = resized[-1].real
        else:
            resized = np.zeros(new_bins, dtype=np.complex128)
            resized[: spectrum.size] = spectrum
        out = np.fft.irfft(resized, n=target_len) * (target_len / arr.size)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def decode_pcm_bytes(raw: bytes, sampwidth: int, channels: int) -> np.ndarray:
        """Decodes raw little-endian PCM frames into mono float32 samples."""
        if not raw:
            return np.zeros(0, dtype=np.float32)
        if sampwidth == 1:
            samples = (
                np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0
        elif sampwidth == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sampwidth == 3:
            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            padded = np.zeros((triplets.shape[0], 4), dtype=np.uint8)
            padded[:, :3] = triplets
            padded[:, 3] = np.where(triplets[:, 2] & 0x80, 0xFF, 0x00)
            int24 = padded.view("<i4").reshape(-1)
            samples = int24.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")
        if channels > 1:
            usable = (samples.size // channels) * channels
            samples = samples[:usable].reshape(-1, channels).mean(axis=1)
        return np.asarray(np.clip(samples, -1.0, 1.0), dtype=np.float32)

    @classmethod
    def _ffmpeg_decode(
        cls, data: bytes, target_sr: int = STT_SAMPLE_RATE
    ) -> tuple[np.ndarray, int]:
        """Decodes arbitrary audio containers (WebM, Ogg, MP3, etc.) via FFmpeg."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "Audio is not standard PCM WAV and ffmpeg is not installed."
            )
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "pipe:1",
            ],
            input=data,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"FFmpeg failed to decode audio: {stderr}")
        samples = cls.decode_pcm_bytes(proc.stdout, sampwidth=2, channels=1)
        return samples, target_sr

    @classmethod
    def load_wav_bytes(
        cls, data: bytes, target_sr: int = STT_SAMPLE_RATE
    ) -> tuple[np.ndarray, int]:
        """Loads audio bytes (WAV or FFmpeg-supported formats) as mono float32."""
        if not data:
            return np.zeros(0, dtype=np.float32), target_sr
        try:
            with wave.open(io.BytesIO(data), "rb") as wav_file:
                orig_sr = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
            mono = cls.decode_pcm_bytes(frames, sampwidth=sampwidth, channels=channels)
            return cls.resample(mono, orig_sr, target_sr), target_sr
        except (wave.Error, EOFError, ValueError):
            return cls._ffmpeg_decode(data, target_sr=target_sr)

    @classmethod
    def load_file(cls, path: str, target_sr: int = STT_SAMPLE_RATE) -> np.ndarray:
        """Loads an audio file from disk and resamples it to ``target_sr`` mono."""
        with open(path, "rb") as handle:
            audio, _ = cls.load_wav_bytes(handle.read(), target_sr=target_sr)
        return audio

    @staticmethod
    def compute_peaks(audio: np.ndarray, num_bins: int = 64) -> list[float]:
        """Computes ``num_bins`` normalized RMS envelope bars in ``[0.0, 1.0]``."""
        if num_bins <= 0:
            return []
        if audio.size == 0:
            return [0.0] * num_bins
        edges = np.linspace(0, audio.size, num_bins + 1, dtype=int)
        raw_bars: list[float] = []
        for start, end in zip(edges[:-1], edges[1:]):
            seg = audio[start:end]
            if seg.size == 0:
                raw_bars.append(0.0)
            else:
                rms = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2)))
                raw_bars.append(rms)
        peak = max(raw_bars)
        if peak < 1e-6:
            return [0.0] * num_bins
        return [round(min(1.0, val / peak), 4) for val in raw_bars]

    @staticmethod
    def split_chunks(
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
        max_chunk_s: float = 30.0,
    ) -> list[np.ndarray]:
        """Splits long audio near quiet frames to stay within model context."""
        max_samples = int(max_chunk_s * sample_rate)
        if audio.size <= max_samples:
            return [audio]
        chunks: list[np.ndarray] = []
        start = 0
        win = max(1, int(0.05 * sample_rate))
        search_pad = int(3.0 * sample_rate)
        while start < audio.size:
            if audio.size - start <= max_samples:
                chunks.append(audio[start:])
                break
            target = start + max_samples
            lo = max(start + sample_rate, target - search_pad)
            hi = min(audio.size - win, target)
            best_cut = target
            if hi > lo:
                segment = audio[lo:hi]
                usable = (segment.size // win) * win
                if usable >= win:
                    energies = np.mean(segment[:usable].reshape(-1, win) ** 2, axis=1)
                    best_idx = int(np.argmin(energies))
                    best_cut = lo + best_idx * win + win // 2
            chunks.append(audio[start:best_cut])
            start = best_cut
        return chunks


class VadSegmenter:
    """Energy-based voice activity endpoint detector for conversational turns."""

    def __init__(
        self,
        *,
        sample_rate: int = STT_SAMPLE_RATE,
        frame_ms: int = 40,
        energy_threshold: float = 0.018,
        speech_trigger_frames: int = 2,
        silence_duration_s: float = 0.9,
        pre_roll_s: float = 0.3,
        min_speech_s: float = 0.2,
        max_duration_s: float = 30.0,
        wait_timeout_s: float = 15.0,
    ) -> None:
        """Initializes the VAD state machine.

        Args:
          sample_rate: Stream sample rate in Hz.
          frame_ms: Frame window duration in milliseconds.
          energy_threshold: Normalized RMS amplitude above which a frame is voiced.
          speech_trigger_frames: Consecutive voiced frames required for onset.
          silence_duration_s: Trailing silence required to end an active utterance.
          pre_roll_s: Buffered audio preserved prior to speech onset.
          min_speech_s: Minimum captured duration after onset before stopping.
          max_duration_s: Hard maximum utterance length after onset.
          wait_timeout_s: Maximum wait time for speech onset (0 disables timeout).
        """
        frame_s = max(0.005, frame_ms / 1000.0)
        self._threshold = energy_threshold
        self._trigger_frames = max(1, speech_trigger_frames)
        self._silence_limit = max(1, int(round(silence_duration_s / frame_s)))
        self._min_active = max(1, int(round(min_speech_s / frame_s)))
        self._max_active = max(1, int(round(max_duration_s / frame_s)))
        self._wait_limit = (
            max(1, int(round(wait_timeout_s / frame_s))) if wait_timeout_s > 0 else 0
        )
        pre_roll_len = max(1, int(round(pre_roll_s / frame_s)))

        self._pre_roll: deque[bytes] = deque(maxlen=pre_roll_len)
        self._captured: list[bytes] = []
        self._triggered = False
        self._speech_run = 0
        self._silence_run = 0
        self._active_frames = 0
        self._waited_frames = 0
        self._sample_rate = sample_rate

    @property
    def triggered(self) -> bool:
        """Returns True once speech onset has been detected."""
        return self._triggered

    def process_pcm16(self, pcm_bytes: bytes) -> str:
        """Processes one PCM16 mono frame and returns the segmenter state.

        Returns:
          One of ``"waiting"``, ``"speaking"``, ``"done"``, or ``"timeout"``.
        """
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        voiced = rms >= self._threshold

        if not self._triggered:
            self._pre_roll.append(pcm_bytes)
            self._waited_frames += 1
            self._speech_run = self._speech_run + 1 if voiced else 0
            if self._speech_run >= self._trigger_frames:
                self._triggered = True
                self._captured.extend(self._pre_roll)
                self._active_frames = 1
                self._silence_run = 0
                return "speaking"
            if self._wait_limit and self._waited_frames >= self._wait_limit:
                return "timeout"
            return "waiting"

        self._captured.append(pcm_bytes)
        self._active_frames += 1
        self._silence_run = 0 if voiced else self._silence_run + 1
        if (
            self._active_frames >= self._min_active
            and self._silence_run >= self._silence_limit
        ) or self._active_frames >= self._max_active:
            return "done"
        return "speaking"

    def get_audio(self) -> np.ndarray:
        """Returns the captured utterance as 1D float32 audio."""
        if not self._triggered or not self._captured:
            return np.zeros(0, dtype=np.float32)
        return AudioProcessor.decode_pcm_bytes(
            b"".join(self._captured), sampwidth=2, channels=1
        )


class AudioPlayer:
    """Interruptible PCM speaker player supporting chunked TTS streaming."""

    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE) -> None:
        """Initializes the player for the target ``sample_rate`` in Hz."""
        self._sample_rate = sample_rate
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._stopped = False

    @classmethod
    def available_backend(cls) -> str | None:
        """Returns the first available system raw audio player, or None."""
        for cmd in ("paplay", "aplay"):
            if shutil.which(cmd):
                return cmd
        return None

    def _player_commands(self) -> tuple[tuple[str, list[str]], ...]:
        """Returns supported player command templates for this sample rate."""
        return (
            (
                "paplay",
                [
                    "--raw",
                    "--format=s16le",
                    f"--rate={self._sample_rate}",
                    "--channels=1",
                ],
            ),
            (
                "aplay",
                [
                    "-q",
                    "-t",
                    "raw",
                    "-f",
                    "S16_LE",
                    "-r",
                    str(self._sample_rate),
                    "-c",
                    "1",
                ],
            ),
        )

    def __enter__(self) -> "AudioPlayer":
        self._stopped = False
        for cmd, args in self._player_commands():
            if shutil.which(cmd):
                with self._lock:
                    self._proc = subprocess.Popen([cmd, *args], stdin=subprocess.PIPE)
                return self
        raise RuntimeError("No raw audio player found (tried paplay, aplay).")

    def write(self, pcm_bytes: bytes) -> bool:
        """Writes raw PCM16 bytes to the player's stdin. Returns False if stopped."""
        with self._lock:
            proc = self._proc
            stopped = self._stopped
        if stopped or proc is None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(pcm_bytes)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, OSError):
            return False

    def stop(self) -> None:
        """Immediately halts playback (safe to call from another thread)."""
        with self._lock:
            self._stopped = True
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except OSError:
                pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        returncode = proc.wait()
        if exc_type is None and returncode != 0:
            raise subprocess.CalledProcessError(returncode, proc.args)

    def play_stream(
        self,
        chunks: collections.abc.Iterable[np.ndarray],
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[int, float | None, bool]:
        """Streams float32 waveform chunks to speakers with barge-in support.

        Args:
          chunks: Iterable of 1D float32 audio arrays at ``self._sample_rate``.
          cancel_event: Optional event; when set, aborts synthesis and playback.

        Returns:
          Tuple ``(total_samples, time_to_first_audio_s, completed)``.
        """
        if cancel_event is not None and cancel_event.is_set():
            return 0, None, False
        start = time.perf_counter()
        total_samples = 0
        first_chunk_s: float | None = None
        completed = True

        with self:
            for wav in chunks:
                if self._stopped or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    self.stop()
                    completed = False
                    break
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - start
                total_samples += len(wav)
                if not self.write(AudioProcessor.to_pcm16(wav)):
                    completed = False
                    break
        return total_samples, first_chunk_s, completed

    def play(
        self,
        audio: np.ndarray,
        *,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Plays a complete waveform array; returns False if interrupted."""
        _, _, completed = self.play_stream([audio], cancel_event=cancel_event)
        return completed


class AudioRecorder:
    """Non-blocking PipeWire/ALSA microphone capture with VAD endpointing."""

    _RECORDERS = (
        (
            "parec",
            [
                "--raw",
                "--format=s16le",
                f"--rate={STT_SAMPLE_RATE}",
                "--channels=1",
            ],
        ),
        (
            "arecord",
            [
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(STT_SAMPLE_RATE),
                "-c",
                "1",
            ],
        ),
    )

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()
        self._level = 0.0
        self._started_at: float | None = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[bytes] | None = None

    @property
    def is_recording(self) -> bool:
        """Returns True while a recording session is active."""
        return self._proc is not None and not self._stop_event.is_set()

    def elapsed_s(self) -> float:
        """Returns elapsed recording duration in seconds."""
        if not self.is_recording or self._started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._started_at)

    def current_level(self) -> float:
        """Returns instantaneous normalized microphone RMS level in ``[0.0, 1.0]``."""
        with self._lock:
            return self._level

    @classmethod
    def available_backend(cls) -> str | None:
        """Returns the first available system recording command, or None."""
        for cmd, _ in cls._RECORDERS:
            if shutil.which(cmd):
                return cmd
        return None

    def start(self, *, stream_frames: bool = False) -> None:
        """Starts background microphone recording."""
        if self.is_recording:
            return
        for cmd, args in self._RECORDERS:
            if shutil.which(cmd):
                self._stop_event.clear()
                with self._lock:
                    self._chunks = []
                    self._level = 0.0
                self._frame_queue = queue.Queue() if stream_frames else None
                self._started_at = time.perf_counter()
                self._proc = subprocess.Popen(
                    [cmd, *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()
                return
        raise RuntimeError("No audio recorder found (tried parec, arecord).")

    def _reader_loop(self) -> None:
        """Continuously reads 40 ms PCM frames and updates the live RMS level."""
        frame_bytes = int(STT_SAMPLE_RATE * 0.04) * 2
        assert self._proc is not None and self._proc.stdout is not None
        while not self._stop_event.is_set():
            chunk = self._proc.stdout.read(frame_bytes)
            if not chunk:
                break
            even_len = (len(chunk) // 2) * 2
            if even_len == 0:
                continue
            pcm_chunk = chunk[:even_len]
            samples = np.frombuffer(pcm_chunk, dtype="<i2").astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples**2)))
            scaled = min(1.0, rms * 6.0)
            with self._lock:
                self._chunks.append(pcm_chunk)
                self._level = scaled
            if self._frame_queue is not None:
                self._frame_queue.put(pcm_chunk)

    def stop(self) -> np.ndarray:
        """Stops microphone recording and returns captured 16 kHz mono float32 audio."""
        self._stop_event.set()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        with self._lock:
            raw = b"".join(self._chunks)
            self._chunks = []
            self._level = 0.0
        self._frame_queue = None
        self._started_at = None
        return AudioProcessor.decode_pcm_bytes(raw, sampwidth=2, channels=1)

    def record_seconds(self, duration_s: float) -> np.ndarray:
        """Records synchronously for ``duration_s`` seconds and returns audio."""
        self.start()
        try:
            time.sleep(max(0.1, duration_s))
        finally:
            audio = self.stop()
        return audio

    def listen_utterance(
        self,
        *,
        energy_threshold: float = 0.018,
        silence_duration_s: float = 0.9,
        pre_roll_s: float = 0.3,
        min_speech_s: float = 0.2,
        max_duration_s: float = 30.0,
        wait_timeout_s: float = 15.0,
        cancel_event: threading.Event | None = None,
        on_level: collections.abc.Callable[[float, str], None] | None = None,
    ) -> np.ndarray:
        """Records a single spoken utterance and stops automatically on silence.

        Args:
          energy_threshold: RMS amplitude threshold separating speech from silence.
          silence_duration_s: Trailing silence duration before ending the turn.
          pre_roll_s: Audio preserved ahead of speech onset.
          min_speech_s: Minimum utterance length after trigger before auto-stop.
          max_duration_s: Hard cap on utterance duration after speech onset.
          wait_timeout_s: Maximum seconds to wait before speech begins (0 = wait).
          cancel_event: Optional external cancellation signal.
          on_level: Optional callback ``(normalized_level, state)`` per 40 ms frame.

        Returns:
          16 kHz mono float32 array containing the captured utterance, or empty array
          if timed out or cancelled before speech onset.
        """
        vad = VadSegmenter(
            sample_rate=STT_SAMPLE_RATE,
            frame_ms=40,
            energy_threshold=energy_threshold,
            silence_duration_s=silence_duration_s,
            pre_roll_s=pre_roll_s,
            min_speech_s=min_speech_s,
            max_duration_s=max_duration_s,
            wait_timeout_s=wait_timeout_s,
        )
        self.start(stream_frames=True)
        frame_q = self._frame_queue
        try:
            while frame_q is not None:
                if cancel_event is not None and cancel_event.is_set():
                    return np.zeros(0, dtype=np.float32)
                try:
                    frame = frame_q.get(timeout=0.1)
                except queue.Empty:
                    if not self.is_recording:
                        break
                    continue
                state = vad.process_pcm16(frame)
                if on_level is not None:
                    on_level(self.current_level(), state)
                if state in ("done", "timeout"):
                    break
            return vad.get_audio()
        finally:
            self.stop()
