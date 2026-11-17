"""Command-line interface for voice_io offline TTS and STT."""

import argparse
import dataclasses
import json
import os
import sys
import time

# Pin single-threaded BLAS before NumPy loads to avoid thread-sync overhead.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from . import (  # noqa: E402
    TTS_SAMPLE_RATE,
    VOICES,
    AudioPlayer,
    AudioRecorder,
    SpeechToText,
    TextToSpeech,
    TranscriptionResult,
    to_pcm16,
)


class VoiceIoCli:
    """Command-line runner for scriptable offline TTS and STT."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Builds the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="voice-io",
            description="Offline CPU bidirectional TTS & STT SDK/CLI.",
        )
        parser.add_argument("text", nargs="?", help="Text to speak via TTS.")
        parser.add_argument(
            "-o",
            "--out",
            default=None,
            help="Output WAV path (defaults to out.wav unless --play/--stdin).",
        )
        parser.add_argument("--voice", default="juno", choices=VOICES)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--temp", type=float, default=0.5)
        parser.add_argument(
            "--retries",
            type=int,
            default=3,
            help="Max retries per piece on stall/runaway (0 for lowest latency).",
        )
        parser.add_argument(
            "--no-markdown",
            action="store_true",
            help="Disable Markdown-to-speech sanitization on TTS input.",
        )
        parser.add_argument(
            "--play",
            action="store_true",
            help="Stream synthesized audio directly to speakers.",
        )
        parser.add_argument(
            "--stdin",
            action="store_true",
            help="Read lines from stdin and speak each one (implies --play).",
        )
        parser.add_argument(
            "--transcribe",
            metavar="FILE",
            default=None,
            help="Transcribe an audio file offline.",
        )
        parser.add_argument(
            "--listen",
            action="store_true",
            help="Record one spoken turn from microphone and transcribe it.",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=0.0,
            help="Fixed seconds to record in --listen mode (0 uses auto VAD stop).",
        )
        parser.add_argument(
            "--silence",
            type=float,
            default=0.9,
            help="Trailing silence seconds to end VAD capture in --listen mode.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.018,
            help="RMS energy threshold for VAD onset/offset in --listen mode.",
        )
        parser.add_argument(
            "--stt-model",
            default="base",
            choices=("tiny", "base"),
            help="Offline ONNX STT model size.",
        )
        parser.add_argument(
            "--echo",
            action="store_true",
            help="Speak back transcribed text via TTS (with --listen/--transcribe).",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit structured JSON STT result on stdout.",
        )
        return parser

    @staticmethod
    def speak(
        tts: TextToSpeech,
        text: str,
        *,
        seed: int,
        play: bool,
        out: str | None,
    ) -> None:
        """Synthesizes ``text``, streaming playback and/or saving to ``out``."""
        start = time.perf_counter()
        first_chunk_elapsed: float | None = None
        chunks: list[np.ndarray] | None = [] if out else None
        total_samples = 0

        if play:
            with AudioPlayer(sample_rate=tts.sample_rate) as player:
                for wav in tts.stream(text, seed=seed):
                    if first_chunk_elapsed is None:
                        first_chunk_elapsed = time.perf_counter() - start
                    total_samples += len(wav)
                    if chunks is not None:
                        chunks.append(wav)
                    player.write(to_pcm16(wav))
        else:
            for wav in tts.stream(text, seed=seed):
                total_samples += len(wav)
                if chunks is not None:
                    chunks.append(wav)

        elapsed = time.perf_counter() - start
        duration = total_samples / TTS_SAMPLE_RATE
        if out and chunks:
            full_wav = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            tts.save(out, full_wav)
            label = out
        else:
            label = "stream"
        ttfa = (
            f", first audio in {first_chunk_elapsed:.2f} s"
            if play and first_chunk_elapsed is not None
            else ""
        )
        print(
            f"{label}: {duration:.2f} s of audio in {elapsed:.2f} s "
            f"({duration / elapsed:.1f}x real time{ttfa})",
            file=sys.stderr,
        )

    @staticmethod
    def _format_stt_output(result: TranscriptionResult, *, as_json: bool) -> None:
        """Prints transcript (clean text or JSON) to stdout and stats to stderr."""
        if as_json:
            payload = dataclasses.asdict(result)
            print(json.dumps(payload))
        else:
            print(result.text)
        print(
            f"[{result.model_name}] "
            f"{result.duration_s:.2f}s audio in {result.inference_s:.2f}s "
            f"({result.rtf:.1f}x real time)",
            file=sys.stderr,
        )

    @classmethod
    def _handle_stt(cls, args: argparse.Namespace) -> int:
        """Handles ``--transcribe`` and ``--listen`` CLI invocations."""
        stt = SpeechToText(model_name=args.stt_model)

        if args.transcribe:
            result = stt.transcribe_file(args.transcribe)
        else:
            recorder = AudioRecorder()
            if args.duration > 0:
                print(
                    f"Recording for {args.duration:.1f}s...",
                    file=sys.stderr,
                    flush=True,
                )
                audio = recorder.record_seconds(args.duration)
            else:
                print(
                    "Listening (auto-stop on silence)...",
                    file=sys.stderr,
                    flush=True,
                )
                audio = recorder.listen_utterance(
                    energy_threshold=args.threshold,
                    silence_duration_s=args.silence,
                )
            result = stt.transcribe_array(audio, sample_rate=stt.sample_rate)

        cls._format_stt_output(result, as_json=args.json)

        if (args.echo or args.play) and result.text:
            tts = TextToSpeech(
                voice=args.voice,
                temp=args.temp,
                retries=args.retries,
                sanitize_markdown=not args.no_markdown,
            )
            cls.speak(tts, result.text, seed=args.seed, play=True, out=args.out)
        return 0

    def run(self, argv: list[str] | None = None) -> int:
        """Parses CLI arguments and executes TTS or STT workflows."""
        parser = self.build_parser()
        args = parser.parse_args(argv)

        if args.transcribe or args.listen:
            return self._handle_stt(args)
        if not args.stdin and args.text is None:
            parser.error("provide text, --stdin, --transcribe, or --listen")

        tts = TextToSpeech(
            voice=args.voice,
            temp=args.temp,
            retries=args.retries,
            sanitize_markdown=not args.no_markdown,
        )
        if args.stdin:
            for line in sys.stdin:
                stripped = line.strip()
                if stripped:
                    self.speak(tts, stripped, seed=args.seed, play=True, out=None)
            return 0

        out = args.out if (args.out is not None or args.play) else "out.wav"
        self.speak(tts, args.text, seed=args.seed, play=args.play, out=out)
        return 0


def main() -> int:
    """Primary CLI entrypoint (``voice-io``)."""
    return VoiceIoCli().run()


if __name__ == "__main__":
    raise SystemExit(main())
