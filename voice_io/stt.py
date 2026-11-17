"""SpeechToText SDK and default offline ONNX encoder-decoder backend."""

import asyncio
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .audio import STT_SAMPLE_RATE, AudioProcessor
from .backends import SttBackend

DEFAULT_STT_REPO = "UsefulSensors/moonshine"
SUPPORTED_STT_MODELS = ("tiny", "base")
SUPPORTED_PRECISIONS = ("float", "quantized")
_MODEL_SPECS: dict[str, tuple[int, int, int]] = {
    "tiny": (6, 8, 36),
    "base": (8, 8, 52),
}


@dataclass(slots=True)
class TranscriptionResult:
    """Container for speech-to-text transcript and performance telemetry."""

    text: str
    duration_s: float
    inference_s: float
    rtf: float
    model_name: str
    num_tokens: int
    peaks: list[float]


class OnnxSttBackend:
    """Offline CPU ONNX encoder-decoder STT backend (Moonshine architecture)."""

    def __init__(
        self,
        model_name: str = "base",
        precision: str = "float",
        num_threads: int = 4,
        models_dir: str | None = None,
        repo_id: str = DEFAULT_STT_REPO,
    ) -> None:
        """Initializes ONNX encoder/decoder sessions and tokenizer.

        Args:
          model_name: Model variant (``"tiny"`` or ``"base"``, with optional prefix).
          precision: Weight precision (``"float"`` or ``"quantized"``).
          num_threads: Intra-op CPU threads for ONNX Runtime.
          models_dir: Optional local directory containing ONNX files and tokenizer.
          repo_id: Hugging Face Hub repository containing ONNX weights.
        """
        short_name = model_name.split("/")[-1].strip().lower()
        if short_name not in _MODEL_SPECS:
            raise ValueError(
                f"Unsupported STT model {model_name!r}; "
                f"expected one of {SUPPORTED_STT_MODELS}."
            )
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision {precision!r}; "
                f"expected one of {SUPPORTED_PRECISIONS}."
            )

        self._model_name = short_name
        self.precision = precision
        self._num_layers, self._num_kv_heads, self._head_dim = _MODEL_SPECS[short_name]
        self._decoder_start_token_id = 1
        self._eos_token_id = 2

        encoder_path, decoder_path, tokenizer_path = self._resolve_assets(
            short_name, precision, models_dir, repo_id
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, num_threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._encoder = ort.InferenceSession(
            encoder_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._decoder = ort.InferenceSession(
            decoder_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._encoder_inputs = {x.name for x in self._encoder.get_inputs()}
        self._decoder_inputs = {x.name for x in self._decoder.get_inputs()}
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        """Expected mono input sample rate in Hz (16 kHz)."""
        return STT_SAMPLE_RATE

    @property
    def model_name(self) -> str:
        """Active model identifier."""
        return self._model_name

    @staticmethod
    def _resolve_assets(
        model_name: str,
        precision: str,
        models_dir: str | None,
        repo_id: str,
    ) -> tuple[str, str, str]:
        """Resolves local or HuggingFace-cached ONNX weights and tokenizer."""
        if models_dir is not None:
            return (
                os.path.join(models_dir, "encoder_model.onnx"),
                os.path.join(models_dir, "decoder_model_merged.onnx"),
                os.path.join(models_dir, "tokenizer.json"),
            )

        subfolder = f"onnx/merged/{model_name}/{precision}"
        encoder = hf_hub_download(repo_id, "encoder_model.onnx", subfolder=subfolder)
        decoder = hf_hub_download(
            repo_id, "decoder_model_merged.onnx", subfolder=subfolder
        )
        tokenizer = hf_hub_download(
            repo_id, "tokenizer.json", subfolder="onnx/merged/base/float"
        )
        return encoder, decoder, tokenizer

    def generate_tokens(
        self, audio_16k: np.ndarray, max_len: int | None = None
    ) -> list[int]:
        """Runs encoder + autoregressive KV-cached decoder on 16 kHz audio."""
        if audio_16k.ndim == 1:
            batch_audio = audio_16k[None, :].astype(np.float32, copy=False)
        else:
            batch_audio = audio_16k.astype(np.float32, copy=False)

        duration_s = batch_audio.shape[1] / STT_SAMPLE_RATE
        if max_len is None:
            max_len = max(24, min(256, int(duration_s * 7.5) + 16))

        audio_mask = np.ones_like(batch_audio, dtype=np.int64)
        encoder_feed: dict[str, np.ndarray] = {"input_values": batch_audio}
        if "attention_mask" in self._encoder_inputs:
            encoder_feed["attention_mask"] = audio_mask

        with self._lock:
            last_hidden_state = self._encoder.run(None, encoder_feed)[0]

            past_kv: dict[str, np.ndarray] = {
                f"past_key_values.{layer}.{kind}.{kv}": np.zeros(
                    (0, self._num_kv_heads, 1, self._head_dim), dtype=np.float32
                )
                for layer in range(self._num_layers)
                for kind in ("decoder", "encoder")
                for kv in ("key", "value")
            }

            tokens = [self._decoder_start_token_id]
            input_ids = np.array([[self._decoder_start_token_id]], dtype=np.int64)

            for step in range(max_len):
                use_cache = step > 0
                decoder_feed: dict[str, np.ndarray] = {
                    "input_ids": input_ids,
                    "encoder_hidden_states": last_hidden_state,
                    "use_cache_branch": np.array([use_cache], dtype=bool),
                    **past_kv,
                }
                if "encoder_attention_mask" in self._decoder_inputs:
                    decoder_feed["encoder_attention_mask"] = audio_mask

                outputs = self._decoder.run(None, decoder_feed)
                logits = outputs[0]
                present_kv = outputs[1:]
                next_token = int(np.argmax(logits[0, -1]))
                tokens.append(next_token)
                if next_token == self._eos_token_id:
                    break

                input_ids = np.array([[next_token]], dtype=np.int64)
                for key, val in zip(past_kv.keys(), present_kv):
                    if not use_cache or "decoder" in key:
                        past_kv[key] = val

        return tokens

    def transcribe_pcm(self, audio: np.ndarray) -> tuple[str, int]:
        """Transcribes one chunk of 16 kHz mono audio and decodes tokens."""
        tokens = self.generate_tokens(audio)
        text = self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return text, len(tokens)


class SpeechToText:
    """High-level offline speech-to-text recognizer with pluggable backend."""

    def __init__(
        self,
        *,
        backend: SttBackend | None = None,
        model_name: str = "base",
        precision: str = "float",
        num_threads: int = 4,
        models_dir: str | None = None,
    ) -> None:
        """Initializes SpeechToText.

        Args:
          backend: Optional custom :class:`SttBackend`. Defaults to
            :class:`OnnxSttBackend`.
          model_name: Default ONNX model size (``"tiny"`` or ``"base"``).
          precision: Weight precision (``"float"`` or ``"quantized"``).
          num_threads: Intra-op CPU threads for ONNX Runtime.
          models_dir: Optional directory containing local ONNX model assets.
        """
        self.backend: SttBackend = (
            backend
            if backend is not None
            else OnnxSttBackend(
                model_name=model_name,
                precision=precision,
                num_threads=num_threads,
                models_dir=models_dir,
            )
        )
        self.sample_rate = self.backend.sample_rate

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = STT_SAMPLE_RATE
    ) -> TranscriptionResult:
        """Transcribes a 1D waveform array and returns text with telemetry."""
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_target = AudioProcessor.resample(mono, sample_rate, self.sample_rate)
        duration_s = audio_target.size / self.sample_rate
        peaks = AudioProcessor.compute_peaks(audio_target)

        if duration_s < 0.08 or float(np.max(np.abs(audio_target))) < 1e-4:
            return TranscriptionResult(
                text="",
                duration_s=duration_s,
                inference_s=0.0,
                rtf=0.0,
                model_name=self.backend.model_name,
                num_tokens=0,
                peaks=peaks,
            )

        min_samples = int(0.25 * self.sample_rate)
        if audio_target.size < min_samples:
            audio_target = np.pad(audio_target, (0, min_samples - audio_target.size))

        t0 = time.perf_counter()
        chunks = AudioProcessor.split_chunks(audio_target, self.sample_rate)
        decoded_parts: list[str] = []
        total_tokens = 0
        for chunk in chunks:
            text, num_tokens = self.backend.transcribe_pcm(chunk)
            total_tokens += num_tokens
            if text:
                decoded_parts.append(text)
        inference_s = max(1e-6, time.perf_counter() - t0)
        full_text = " ".join(decoded_parts).strip()
        return TranscriptionResult(
            text=full_text,
            duration_s=duration_s,
            inference_s=inference_s,
            rtf=duration_s / inference_s,
            model_name=self.backend.model_name,
            num_tokens=total_tokens,
            peaks=peaks,
        )

    def transcribe_bytes(self, data: bytes) -> TranscriptionResult:
        """Decodes WAV/FFmpeg-supported audio bytes and transcribes speech."""
        audio, sr = AudioProcessor.load_wav_bytes(data, self.sample_rate)
        return self.transcribe_array(audio, sample_rate=sr)

    def transcribe_file(self, path: str) -> TranscriptionResult:
        """Loads an audio file from disk and transcribes speech."""
        audio = AudioProcessor.load_file(path, self.sample_rate)
        return self.transcribe_array(audio, sample_rate=self.sample_rate)

    async def atranscribe_array(
        self, audio: np.ndarray, sample_rate: int = STT_SAMPLE_RATE
    ) -> TranscriptionResult:
        """Asynchronously transcribes a 1D waveform array in a worker thread."""
        return await asyncio.to_thread(
            self.transcribe_array, audio, sample_rate=sample_rate
        )

    async def atranscribe_bytes(self, data: bytes) -> TranscriptionResult:
        """Asynchronously decodes audio bytes and transcribes speech."""
        return await asyncio.to_thread(self.transcribe_bytes, data)

    async def atranscribe_file(self, path: str) -> TranscriptionResult:
        """Asynchronously loads an audio file from disk and transcribes speech."""
        return await asyncio.to_thread(self.transcribe_file, path)
