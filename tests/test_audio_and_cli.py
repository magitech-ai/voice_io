"""Tests for AudioProcessor, AudioPlayer, AudioRecorder, and CLI entry points."""

import io
import json
import runpy
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voice_io import (
    STT_SAMPLE_RATE,
    AudioPlayer,
    AudioProcessor,
    AudioRecorder,
    TranscriptionResult,
    to_pcm16,
    to_wav_bytes,
)
from voice_io.__main__ import VoiceIoCli, main


class _FakePipe:
    """In-memory writable stdin or readable stdout stream for Popen mocks."""

    def __init__(
        self,
        chunks: list[bytes] | None = None,
        *,
        broken_write: bool = False,
        raise_on_close: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        self._chunks = list(chunks or [])
        self._idx = 0
        self._broken_write = broken_write
        self._raise_on_close = raise_on_close
        self._delay_s = delay_s
        self.written = bytearray()
        self.closed = False

    def write(self, data: bytes) -> int:
        if self._broken_write:
            raise BrokenPipeError("broken pipe")
        self.written.extend(data)
        return len(data)

    def flush(self) -> None:
        if self._broken_write:
            raise BrokenPipeError("broken pipe")

    def read(self, size: int = -1) -> bytes:
        del size
        if self._idx >= len(self._chunks):
            return b""
        if self._delay_s > 0 and self._idx == 0:
            time.sleep(self._delay_s)
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def close(self) -> None:
        self.closed = True
        if self._raise_on_close:
            raise OSError("close failed")


class _FakePopen:
    """Configurable subprocess.Popen double for player and recorder tests."""

    def __init__(
        self,
        cmd: list[str],
        *,
        stdout_chunks: list[bytes] | None = None,
        returncode: int = 0,
        broken_write: bool = False,
        raise_on_close: bool = False,
        timeout_on_wait: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        self.args = cmd
        self.stdin = _FakePipe(
            broken_write=broken_write,
            raise_on_close=raise_on_close,
        )
        self.stdout = _FakePipe(
            chunks=stdout_chunks,
            raise_on_close=raise_on_close,
            delay_s=delay_s,
        )
        self.returncode = returncode
        self._timeout_on_wait = timeout_on_wait
        self.killed = False
        self.terminated = False
        self.wait_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._timeout_on_wait and timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True


def _tone_pcm_bytes(amp: float, duration_s: float, sr: int = STT_SAMPLE_RATE) -> bytes:
    n = int(sr * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    wave = (amp * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    return to_pcm16(wave)


def test_audio_processor_edge_cases_and_ffmpeg_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers resample bounds, PCM widths, stereo mixdown, peaks, chunking, ffmpeg."""
    with pytest.raises(ValueError, match="positive"):
        AudioProcessor.resample(np.zeros(16, dtype=np.float32), 0, 16000)

    tiny = np.array([0.1, -0.2], dtype=np.float32)
    assert np.array_equal(AudioProcessor.resample(tiny, 8000, 8000), tiny)
    tiny_up = AudioProcessor.resample(tiny, 8000, 16000)
    assert len(tiny_up) == 4
    assert tiny_up.dtype == np.float32

    t = np.linspace(0.0, 0.1, 800, endpoint=False, dtype=np.float32)
    sine_8k = (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    up = AudioProcessor.resample(sine_8k, 8000, 48000)
    assert len(up) == 4800

    assert len(AudioProcessor.decode_pcm_bytes(b"", sampwidth=2, channels=1)) == 0

    pcm_u8 = bytes([0, 128, 255])
    decoded_u8 = AudioProcessor.decode_pcm_bytes(pcm_u8, sampwidth=1, channels=1)
    assert decoded_u8.dtype == np.float32
    assert len(decoded_u8) == 3

    pcm_24_stereo = b"\x00\x00\x40" + b"\x00\x00\xc0"
    decoded_24 = AudioProcessor.decode_pcm_bytes(pcm_24_stereo, sampwidth=3, channels=2)
    assert len(decoded_24) == 1
    assert abs(float(decoded_24[0])) < 1e-4

    pcm_32 = np.array([2**30, -(2**30)], dtype="<i4").tobytes()
    decoded_32 = AudioProcessor.decode_pcm_bytes(pcm_32, sampwidth=4, channels=1)
    assert len(decoded_32) == 2

    with pytest.raises(ValueError, match="Unsupported WAV sample width"):
        AudioProcessor.decode_pcm_bytes(b"\x00" * 5, sampwidth=5, channels=1)

    empty_decoded, empty_sr = AudioProcessor.load_wav_bytes(b"", STT_SAMPLE_RATE)
    assert empty_sr == STT_SAMPLE_RATE
    assert len(empty_decoded) == 0

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        AudioProcessor.load_wav_bytes(b"NOTAWAVDATA", STT_SAMPLE_RATE)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"bad header"
        ),
    )
    with pytest.raises(ValueError, match="bad header"):
        AudioProcessor.load_wav_bytes(b"NOTAWAVDATA", STT_SAMPLE_RATE)

    fake_pcm = _tone_pcm_bytes(0.25, 0.1)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=fake_pcm, stderr=b""
        ),
    )
    decoded, sr = AudioProcessor.load_wav_bytes(b"OGG_PLACEHOLDER", STT_SAMPLE_RATE)
    assert sr == STT_SAMPLE_RATE
    assert len(decoded) == len(fake_pcm) // 2

    assert (
        AudioProcessor.compute_peaks(np.zeros(10, dtype=np.float32), num_bins=0) == []
    )
    assert (
        AudioProcessor.compute_peaks(np.zeros(0, dtype=np.float32), num_bins=8)
        == [0.0] * 8
    )
    short_peaks = AudioProcessor.compute_peaks(
        np.array([0.0, 0.5, -1.0], dtype=np.float32), num_bins=10
    )
    assert len(short_peaks) == 10
    assert max(short_peaks) == 1.0

    quiet_gaps = np.concatenate(
        [
            np.full(int(STT_SAMPLE_RATE * 0.4), 0.3, dtype=np.float32),
            np.zeros(int(STT_SAMPLE_RATE * 0.2), dtype=np.float32),
            np.full(int(STT_SAMPLE_RATE * 0.4), 0.3, dtype=np.float32),
            np.full(int(STT_SAMPLE_RATE * 0.6), 0.5, dtype=np.float32),
        ]
    )
    chunks = AudioProcessor.split_chunks(
        quiet_gaps, sample_rate=STT_SAMPLE_RATE, max_chunk_s=0.5
    )
    assert len(chunks) >= 3
    assert sum(len(c) for c in chunks) == len(quiet_gaps)


def test_audio_player_backends_error_paths_and_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers backend discovery, write/stop/exit edge cases, play, and cancellation."""
    spawned: list[_FakePopen] = []

    def _spawn_normal(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        proc = _FakePopen(cmd)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _spawn_normal)
    sample = np.full(800, 0.2, dtype=np.float32)
    sample_pcm = to_pcm16(sample)

    for target in ("paplay", "aplay"):
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, t=target: name if name == t else None,
        )
        assert AudioPlayer.available_backend() == target
        with AudioPlayer(sample_rate=16000) as player:
            assert player.write(sample_pcm) is True

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert AudioPlayer.available_backend() is None
    with pytest.raises(RuntimeError, match="No raw audio player"):
        with AudioPlayer(sample_rate=16000):
            pass

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/paplay" if name == "paplay" else None,
    )

    with AudioPlayer(sample_rate=16000) as player:
        assert player.write(sample_pcm) is True
        player.stop()
        assert player.write(sample_pcm) is False

    assert AudioPlayer(sample_rate=16000).play(sample) is True
    assert len(spawned[-1].stdin.written) == len(sample_pcm)

    def _spawn_broken(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        return _FakePopen(
            cmd, broken_write=True, timeout_on_wait=True, raise_on_close=True
        )

    monkeypatch.setattr(subprocess, "Popen", _spawn_broken)
    with AudioPlayer(sample_rate=16000) as broken_player:
        assert broken_player.write(sample_pcm) is False
        broken_player.stop()
        broken_player.stop()

    _, _, empty_then_ok = AudioPlayer(sample_rate=16000).play_stream(
        [np.zeros(0, dtype=np.float32)]
    )
    assert empty_then_ok is False

    monkeypatch.setattr(subprocess, "Popen", _spawn_normal)
    _, _, empty_and_sample = AudioPlayer(sample_rate=16000).play_stream(
        [np.zeros(0, dtype=np.float32), sample]
    )
    assert empty_and_sample is True

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, **_kw: _FakePopen(cmd, raise_on_close=True),
    )
    with AudioPlayer(sample_rate=16000) as close_on_exit_err:
        assert close_on_exit_err.write(sample_pcm) is True

    def _spawn_broken_write_once(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        return _FakePopen(cmd, broken_write=True)

    monkeypatch.setattr(subprocess, "Popen", _spawn_broken_write_once)
    _, _, stopped_on_write = AudioPlayer(sample_rate=16000).play_stream([sample])
    assert stopped_on_write is False

    def _spawn_nonzero(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        return _FakePopen(cmd, returncode=2)

    monkeypatch.setattr(subprocess, "Popen", _spawn_nonzero)
    with pytest.raises(subprocess.CalledProcessError):
        with AudioPlayer(sample_rate=16000) as err_player:
            err_player.write(sample_pcm)

    with pytest.raises(ValueError, match="outer"):
        with AudioPlayer(sample_rate=16000) as err_player2:
            err_player2.write(sample_pcm)
            raise ValueError("outer")

    monkeypatch.setattr(subprocess, "Popen", _spawn_normal)
    cancel = threading.Event()

    def _chunk_iter() -> Iterable[np.ndarray]:
        yield sample
        cancel.set()
        yield sample

    total_samples, ttfa, completed = AudioPlayer(sample_rate=16000).play_stream(
        _chunk_iter(), cancel_event=cancel
    )
    assert total_samples == len(sample)
    assert ttfa is not None
    assert completed is False

    total2, ttfa2, completed2 = AudioPlayer(sample_rate=16000).play_stream([sample])
    assert total2 == len(sample)
    assert ttfa2 is not None
    assert completed2 is True


def test_audio_recorder_backends_capture_and_vad_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers recorder backend discovery, odd-byte framing, timeout, and VAD outcomes."""
    silence_bytes = _tone_pcm_bytes(0.0, 0.04)

    for target in ("parec", "arecord"):
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, t=target: name if name == t else None,
        )
        assert AudioRecorder.available_backend() == target
        monkeypatch.setattr(
            subprocess,
            "Popen",
            lambda cmd, **_kw: _FakePopen(cmd, stdout_chunks=[silence_bytes]),
        )
        rec_backend = AudioRecorder()
        rec_backend.start()
        rec_backend.stop()

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert AudioRecorder.available_backend() is None
    with pytest.raises(RuntimeError):
        AudioRecorder().start()

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/parec" if name == "parec" else None,
    )

    frame_bytes = _tone_pcm_bytes(0.4, 0.04)
    odd_split_chunks = [
        frame_bytes[: len(frame_bytes) - 1],
        frame_bytes[len(frame_bytes) - 1 :],
        frame_bytes,
        frame_bytes,
        silence_bytes,
        silence_bytes,
        silence_bytes,
        silence_bytes,
        silence_bytes,
    ]

    def _spawn_speech(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        return _FakePopen(cmd, stdout_chunks=odd_split_chunks, delay_s=0.06)

    monkeypatch.setattr(subprocess, "Popen", _spawn_speech)
    rec = AudioRecorder()
    rec.start()
    assert rec.is_recording is True
    assert rec.elapsed_s() >= 0.0
    assert rec.current_level() >= 0.0
    rec.start()
    stopped_audio = rec.stop()
    assert len(stopped_audio) >= 0
    assert rec.is_recording is False
    assert rec.elapsed_s() == 0.0
    assert rec.current_level() >= 0.0

    levels: list[tuple[float, str]] = []
    utterance = rec.listen_utterance(
        energy_threshold=0.05,
        silence_duration_s=0.16,
        pre_roll_s=0.08,
        min_speech_s=0.08,
        wait_timeout_s=5.0,
        on_level=lambda lvl, state: levels.append((lvl, state)),
    )
    assert len(utterance) > 0
    assert any(state == "speaking" for _, state in levels)

    def _spawn_silence(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        return _FakePopen(
            cmd,
            stdout_chunks=[silence_bytes] * 60,
            timeout_on_wait=True,
            raise_on_close=True,
        )

    monkeypatch.setattr(subprocess, "Popen", _spawn_silence)
    short = rec.record_seconds(0.05)
    assert len(short) > 0

    cancel = threading.Event()
    cancel.set()
    cancelled = rec.listen_utterance(
        energy_threshold=0.05,
        wait_timeout_s=1.0,
        cancel_event=cancel,
    )
    assert len(cancelled) == 0

    timed_out = rec.listen_utterance(
        energy_threshold=0.05,
        wait_timeout_s=0.12,
    )
    assert len(timed_out) == 0

    def _spawn_slow_queue(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        return _FakePopen(cmd, stdout_chunks=odd_split_chunks, delay_s=0.16)

    monkeypatch.setattr(subprocess, "Popen", _spawn_slow_queue)
    slow_utt = rec.listen_utterance(
        energy_threshold=0.05,
        silence_duration_s=0.16,
        pre_roll_s=0.08,
        min_speech_s=0.08,
        wait_timeout_s=5.0,
    )
    assert len(slow_utt) > 0


def test_cli_speak_listen_transcribe_and_main_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Covers CLI flags for TTS, stdin, transcribe, listen, echo, and python -m."""
    sample_wav = tmp_path / "input.wav"
    tone = np.full(STT_SAMPLE_RATE // 5, 0.25, dtype=np.float32)
    sample_wav.write_bytes(to_wav_bytes(tone, sample_rate=STT_SAMPLE_RATE))
    out_wav = tmp_path / "spoken.wav"

    played_calls: list[tuple[int, int]] = []

    class _FakeTts:
        def __init__(self, **_kwargs: Any) -> None:
            self.sample_rate = 16000

        def __call__(self, text: str, *, seed: int | None = None) -> np.ndarray:
            del text, seed
            return np.full(1600, 0.1, dtype=np.float32)

        def stream(self, text: str, *, seed: int | None = None) -> Iterable[np.ndarray]:
            del text, seed
            yield np.full(800, 0.1, dtype=np.float32)

        def save(self, path: str, wav: np.ndarray) -> None:
            Path(path).write_bytes(to_wav_bytes(wav, sample_rate=16000))

    class _FakeStt:
        def __init__(self, **_kwargs: Any) -> None:
            self.sample_rate = STT_SAMPLE_RATE

        def transcribe_file(self, path: str) -> TranscriptionResult:
            del path
            return TranscriptionResult(
                text="file transcript",
                duration_s=0.2,
                inference_s=0.05,
                rtf=4.0,
                num_tokens=2,
                model_name="tiny",
                peaks=[],
            )

        def transcribe_array(
            self,
            audio: np.ndarray,
            *,
            sample_rate: int = STT_SAMPLE_RATE,
        ) -> TranscriptionResult:
            del audio, sample_rate
            return TranscriptionResult(
                text="mic transcript",
                duration_s=0.2,
                inference_s=0.05,
                rtf=4.0,
                num_tokens=2,
                model_name="tiny",
                peaks=[],
            )

    class _FakeRecorder:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def record_seconds(self, duration_s: float) -> np.ndarray:
            del duration_s
            return np.full(1600, 0.2, dtype=np.float32)

        def listen_utterance(self, **_kwargs: Any) -> np.ndarray:
            return np.full(1600, 0.2, dtype=np.float32)

    class _FakePlayer:
        def __init__(self, sample_rate: int = 16000) -> None:
            self._sr = sample_rate
            self._bytes_written = 0

        def __enter__(self) -> "_FakePlayer":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def write(self, pcm_bytes: bytes) -> bool:
            self._bytes_written += len(pcm_bytes)
            played_calls.append((self._sr, len(pcm_bytes) // 2))
            return True

        def stop(self) -> None:
            pass

        def play_stream(self, chunks: Iterable[np.ndarray]) -> tuple[int, float, bool]:
            total = sum(len(chunk) for chunk in chunks)
            played_calls.append((self._sr, total))
            return total, 0.01, True

    import voice_io.__main__ as cli_mod

    monkeypatch.setattr(cli_mod, "TextToSpeech", _FakeTts)
    monkeypatch.setattr(cli_mod, "SpeechToText", _FakeStt)
    monkeypatch.setattr(cli_mod, "AudioRecorder", _FakeRecorder)
    monkeypatch.setattr(cli_mod, "AudioPlayer", _FakePlayer)

    cli = VoiceIoCli()

    rc_save = cli.run(
        [
            "Hello",
            "-o",
            str(out_wav),
            "--seed",
            "7",
            "--voice",
            "juno",
            "--temp",
            "0.5",
            "--retries",
            "0",
            "--no-markdown",
        ]
    )
    assert rc_save == 0
    assert out_wav.exists()

    rc_play = cli.run(["Hello speakers", "--play"])
    assert rc_play == 0
    assert len(played_calls) >= 1

    monkeypatch.chdir(tmp_path)
    rc_default_out = cli.run(["Default output path"])
    assert rc_default_out == 0
    assert (tmp_path / "out.wav").exists()

    both_wav = tmp_path / "both.wav"
    rc_both = cli.run(["Both output and play", "-o", str(both_wav), "--play"])
    assert rc_both == 0
    assert both_wav.exists()

    with pytest.raises(SystemExit) as no_args_exc:
        cli.run([])
    assert no_args_exc.value.code == 2

    with monkeypatch.context() as mp:
        mp.setattr(_FakePlayer, "write", lambda _self, _b: False)
        rc_stop_play = cli.run(["Stop playback early", "--play"])
        assert rc_stop_play == 0

    monkeypatch.setattr(sys, "stdin", io.StringIO("First line\n\nSecond line\n"))
    rc_stdin = cli.run(["--stdin"])
    assert rc_stdin == 0

    rc_transcribe_echo = cli.run(["--transcribe", str(sample_wav), "--echo"])
    assert rc_transcribe_echo == 0
    assert "file transcript" in capsys.readouterr().out

    rc_listen_dur = cli.run(["--listen", "--duration", "0.2", "--json"])
    assert rc_listen_dur == 0
    dur_payload = json.loads(capsys.readouterr().out)
    assert dur_payload["text"] == "mic transcript"

    rc_listen_vad = cli.run(["--listen", "--silence", "0.2", "--threshold", "0.02"])
    assert rc_listen_vad == 0
    assert "mic transcript" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["voice-io", "--transcribe", str(sample_wav)])
    assert main() == 0
    capsys.readouterr()

    monkeypatch.delitem(sys.modules, "voice_io.__main__", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("voice_io", run_name="__main__")
    assert exc_info.value.code == 0


def test_audio_player_stop_oserror_and_recorder_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers AudioPlayer.stop OSError fallback and recorder stop during empty wait."""

    class _OserrStopPopen:
        def __init__(self) -> None:
            self.stdin = _FakePipe()

        def terminate(self) -> None:
            raise OSError("already dead")

        def kill(self) -> None:
            raise OSError("kill failed")

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: name if name in ("paplay", "parec") else None,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _OserrStopPopen())
    with AudioPlayer() as player:
        player.stop()

    class _DelayedRecPopen:
        def __init__(self) -> None:
            self.stdout = _FakePipe([b"\x00\x00" * 640], delay_s=0.14)

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DelayedRecPopen())
    rec = AudioRecorder()
    timer = threading.Timer(0.02, rec._stop_event.set)
    timer.start()
    try:
        captured = rec.listen_utterance(
            energy_threshold=0.05,
            silence_duration_s=0.2,
            wait_timeout_s=1.0,
        )
    finally:
        timer.cancel()
    assert len(captured) == 0
