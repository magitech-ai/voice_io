"""Pure-NumPy int8 flow TTS inference backend.

Implements a transformer speech model, one-step flow head, and Mimi-adapted
audio decoder using only NumPy and the Python standard library.
"""

import json
import math
import threading
import time
import wave
from typing import Any, cast

import numpy as np

SAMPLE_RATE = 24000
SR = SAMPLE_RATE
FLOAT32 = np.float32
EPS = float(np.finfo(np.float32).eps)

CHARS = list(
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,!?;:'\"-()&/%$€£@#+=*_<>[]"
) + ["…", "—"]
VOCAB = {ch: idx + 3 for idx, ch in enumerate(CHARS)}


def _layer_norm(hidden: np.ndarray) -> np.ndarray:
    """Applies parameter-free layer normalization along the last axis."""
    mean = hidden.mean(-1, keepdims=True)
    centered = hidden - mean
    return centered / np.sqrt(
        (centered * centered).mean(-1, keepdims=True) + FLOAT32(1e-5)
    )


def _silu(x: np.ndarray) -> np.ndarray:
    """Applies SiLU activation."""
    return x / (FLOAT32(1) + np.exp(-x))


def _erf(x: np.ndarray) -> np.ndarray:
    """Approximates erf(x) via Abramowitz-Stegun 7.1.26 (|err| <= 1.5e-7)."""
    abs_x = np.abs(x)
    t = FLOAT32(1) / (FLOAT32(1) + FLOAT32(0.3275911) * abs_x)
    poly = (
        ((FLOAT32(1.061405429) * t - FLOAT32(1.453152027)) * t + FLOAT32(1.421413741))
        * t
        - FLOAT32(0.284496736)
    ) * t + FLOAT32(0.254829592)
    y = FLOAT32(1) - poly * t * np.exp(-abs_x * abs_x)
    return cast(np.ndarray, np.copysign(y, x))


def _gelu(x: np.ndarray) -> np.ndarray:
    """Applies GELU activation."""
    return FLOAT32(0.5) * x * (FLOAT32(1) + _erf(x * FLOAT32(1 / math.sqrt(2))))


def _elu_inplace(x: np.ndarray) -> np.ndarray:
    """Applies ELU activation in-place."""
    neg = np.minimum(x, FLOAT32(0))
    np.expm1(neg, out=neg)
    np.maximum(x, FLOAT32(0), out=x)
    x += neg
    return x


class TorchRandn:
    """PyTorch-compatible CPU MT19937 Box-Muller normal RNG without PyTorch."""

    def __init__(self, seed: int = 0) -> None:
        """Initializes the RNG with ``seed``."""
        self._bitgen = np.random.MT19937(0)
        self.seed(seed)

    def seed(self, seed: int) -> None:
        """Reseeds the underlying MT19937 generator."""
        self._bitgen = np.random.MT19937(0)
        self._bitgen._legacy_seeding(seed)  # pylint: disable=protected-access

    def randn(self, count: int) -> np.ndarray:
        """Draws ``count`` standard normal float32 values (``count % 16 == 0``)."""
        uniform = (self._bitgen.random_raw(count) & 0xFFFFFF).astype(
            np.float64
        ) * 2.0**-24
        uniform = uniform.reshape(-1, 2, 8)
        radius = np.sqrt(-2.0 * np.log(1.0 - uniform[:, 0]))
        theta = 2.0 * np.pi * uniform[:, 1]
        return (
            np.stack([radius * np.cos(theta), radius * np.sin(theta)], 1)
            .reshape(-1)
            .astype(FLOAT32)
        )


class WeightLoader:
    """Utilities for unpacking and dequantizing int8 NPZ weight archives."""

    @staticmethod
    def unflatten(npz: Any) -> Any:
        """Reconstructs the nested weight dictionary from an npz archive."""

        def build(node: Any) -> Any:
            if isinstance(node, dict):
                if "@" in node and len(node) == 1:
                    return npz[node["@"]]
                return {key: build(val) for key, val in node.items()}
            if isinstance(node, list):
                return [build(val) for val in node]
            return node

        return build(json.loads(str(npz["__tree__"])))

    @staticmethod
    def load_linear_weight(
        entry: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Dequantizes a speech-model linear layer entry to float32."""
        if "q" in entry:
            weight = entry["q"].astype(FLOAT32) * entry["s"].astype(FLOAT32)[:, None]
        else:
            weight = entry["w"].astype(FLOAT32)
        bias = entry.get("b")
        return (
            np.ascontiguousarray(weight),
            None if bias is None else bias.astype(FLOAT32),
        )

    @staticmethod
    def dequantize(tensor: dict[str, Any] | np.ndarray) -> np.ndarray:
        """Dequantizes a grouped-int8 decoder tensor or casts float to float32."""
        if not isinstance(tensor, dict):
            return tensor.astype(FLOAT32)
        q8 = tensor["q8"]
        scale = tensor["s"].astype(FLOAT32)
        shape = tuple(tensor["shape"])
        axis = tensor["axis"]
        group_size = tensor["g"]
        cols, num_groups = q8.shape[1], scale.shape[1]
        matrix = np.zeros((q8.shape[0], num_groups * group_size), FLOAT32)
        matrix[:, :cols] = q8
        matrix = (
            matrix.reshape(q8.shape[0], num_groups, group_size) * scale[..., None]
        ).reshape(q8.shape[0], -1)[:, :cols]
        rest = [shape[dim] for dim in range(len(shape)) if dim != axis]
        return np.ascontiguousarray(
            np.moveaxis(matrix.reshape(shape[axis], *rest), 0, axis)
        )


class SpeechModel:
    """Autoregressive transformer speech model with one-step flow head."""

    def __init__(
        self,
        state: dict[str, Any],
        rng: TorchRandn | None = None,
        max_len: int = 1100,
    ) -> None:
        """Initializes the speech model from unpacked weights."""
        cfg = state["cfg"]
        self.num_layers = cfg["L"]
        self.hidden_dim = cfg["d"]
        self.num_heads = cfg["H"]
        self.ff_dim = cfg["ff"]
        self.head_width = cfg["hw"]
        self.latent_dim = cfg["latent"]
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_len = max_len
        self.depth = cfg["depth"]
        self._rng = rng if rng is not None else TorchRandn(0)

        self._layers = [
            [WeightLoader.load_linear_weight(entry) for entry in layer]
            for layer in state["layers"]
        ]
        self._ce = WeightLoader.load_linear_weight(state["ce"])
        self._ada = WeightLoader.load_linear_weight(state["ada"])
        self._hblocks = [
            [WeightLoader.load_linear_weight(entry) for entry in block]
            for block in state["hblocks"]
        ]
        self._lat_in = WeightLoader.load_linear_weight(state["lat_in"])
        self._h_inp = WeightLoader.load_linear_weight(state["h_inp"])
        self._h_out = WeightLoader.load_linear_weight(state["h_out"])
        self._text_emb = state["text_emb"].astype(FLOAT32)
        self._bos = state["bos"].astype(FLOAT32)

        inv_freq = (
            1.0
            / (
                cfg["rope_base"]
                ** (
                    np.arange(0, self.head_dim, 2, dtype=FLOAT32)
                    / FLOAT32(self.head_dim)
                )
            )
        ).astype(FLOAT32)
        angles = np.arange(max_len, dtype=FLOAT32)[:, None] * inv_freq
        cos_half, sin_half = np.cos(angles), np.sin(angles)
        self._cos = np.concatenate([cos_half, cos_half], -1).astype(FLOAT32)
        self._sin = np.concatenate([-sin_half, sin_half], -1).astype(FLOAT32)
        self._kv_cache = np.zeros(
            (self.num_layers, 2, self.num_heads, max_len, self.head_dim), FLOAT32
        )
        self._scale = FLOAT32(self.head_dim**-0.5)

    @staticmethod
    def encode_text(text: str) -> list[int]:
        """Encodes a normalized string into token IDs."""
        return [1] + [VOCAB.get(ch, 2) for ch in text]

    @staticmethod
    def _rms_inv(x: np.ndarray) -> np.ndarray:
        """Computes inverse RMS norm factor for a 1D vector."""
        return cast(
            np.ndarray,
            FLOAT32(1.0) / np.sqrt(np.dot(x, x) / FLOAT32(x.shape[-1]) + FLOAT32(EPS)),
        )

    def _prefix_block(
        self, layer_idx: int, x: np.ndarray, prefix_len: int
    ) -> np.ndarray:
        """Runs one transformer block over the full prefix sequence."""
        (wqkv, _), (wo, _), (w1, _), (w2, _) = self._layers[layer_idx]
        num_heads, head_dim = self.num_heads, self.head_dim
        half_dim = head_dim // 2

        x_norm = x / np.sqrt((x * x).mean(-1, keepdims=True) + FLOAT32(EPS))
        qkv = (x_norm @ wqkv.T).reshape(prefix_len, 3, num_heads, head_dim)
        qk = qkv[:, :2]
        qk = (
            qk * self._cos[:prefix_len, None, None]
            + np.concatenate([qk[..., half_dim:], qk[..., :half_dim]], -1)
            * self._sin[:prefix_len, None, None]
        )
        keys, values = self._kv_cache[layer_idx, 0], self._kv_cache[layer_idx, 1]
        keys[:, :prefix_len] = qk[:, 1].transpose(1, 0, 2)
        values[:, :prefix_len] = qkv[:, 2].transpose(1, 0, 2)

        queries = qk[:, 0].transpose(1, 0, 2)
        scores = (
            np.matmul(queries, keys[:, :prefix_len].transpose(0, 2, 1)) * self._scale
        )
        scores[:, np.triu(np.ones((prefix_len, prefix_len), bool), 1)] = -np.inf
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)

        attn = (
            np.matmul(scores, values[:, :prefix_len])
            .transpose(1, 0, 2)
            .reshape(prefix_len, self.hidden_dim)
        )
        x = x + attn @ wo.T
        x_norm = x / np.sqrt((x * x).mean(-1, keepdims=True) + FLOAT32(EPS))
        gate_up = x_norm @ w1.T
        return cast(
            np.ndarray,
            x + (_silu(gate_up[:, : self.ff_dim]) * gate_up[:, self.ff_dim :]) @ w2.T,
        )

    def _step_block(self, layer_idx: int, x: np.ndarray, pos: int) -> np.ndarray:
        """Runs one autoregressive transformer step at position ``pos``."""
        (wqkv, _), (wo, _), (w1, _), (w2, _) = self._layers[layer_idx]
        num_heads, head_dim = self.num_heads, self.head_dim

        qkv = (wqkv @ x) * self._rms_inv(x)
        qkv = qkv.reshape(3, num_heads, head_dim)
        qk = qkv[:2]
        qk_rot = qk.reshape(2, num_heads, 2, head_dim // 2)[:, :, ::-1].reshape(
            2, num_heads, head_dim
        )
        qk = qk * self._cos[pos] + qk_rot * self._sin[pos]

        keys, values = self._kv_cache[layer_idx, 0], self._kv_cache[layer_idx, 1]
        keys[:, pos] = qk[1]
        values[:, pos] = qkv[2]

        scores = np.matmul(keys[:, : pos + 1], qk[0][:, :, None])[:, :, 0] * self._scale
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
        attn = np.matmul(scores[:, None, :], values[:, : pos + 1])[:, 0]
        x = x + wo @ attn.reshape(-1)

        gate_up = (w1 @ x) * self._rms_inv(x)
        return cast(
            np.ndarray,
            x + w2 @ (_silu(gate_up[: self.ff_dim]) * gate_up[self.ff_dim :]),
        )

    def _condition(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Computes flow-head conditioning vector and EOS logit."""
        weight, bias = self._ce
        cond_eos = (weight @ x) * self._rms_inv(x) + bias
        return _silu(cond_eos[: self.head_width]), float(cond_eos[self.head_width])

    def _flow_head(self, cond: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Runs the one-step flow matching head."""
        hw = self.head_width
        ada = self._ada[0] @ cond + self._ada[1]
        hidden = self._h_inp[0] @ noise + self._h_inp[1]
        for idx, ((w1, b1), (w2, b2)) in enumerate(self._hblocks):
            offset = 3 * hw * idx
            y = (
                ada[offset : offset + hw]
                + _layer_norm(hidden) * ada[offset + hw : offset + 2 * hw]
            )
            hidden = hidden + ada[offset + 2 * hw : offset + 3 * hw] * (
                w2 @ _silu(w1 @ y + b1) + b2
            )
        offset = 3 * hw * self.depth
        return cast(
            np.ndarray,
            self._h_out[0]
            @ (
                ada[offset : offset + hw]
                + _layer_norm(hidden) * ada[offset + hw : offset + 2 * hw]
            )
            + self._h_out[1],
        )

    def generate(
        self,
        voice: np.ndarray,
        text_ids: list[int],
        max_frames: int = 750,
        temp: float = 0.5,
        eos_thresh: float = 0.9,
        min_frames_per_char: float = 0.0,
        force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generates acoustic latents for ``text_ids``."""
        if force is not None:
            max_frames, eos_thresh = len(force), 1.0 - 1e-12
        x = np.concatenate(
            [voice, self._text_emb[np.asarray(text_ids)], self._bos[None]], 0
        ).astype(FLOAT32)
        prefix_len = x.shape[0]
        max_frames = min(max_frames, self.max_len - prefix_len)
        min_frames = int(min_frames_per_char * len(text_ids))

        for layer_idx in range(self.num_layers):
            x = self._prefix_block(layer_idx, x, prefix_len)
        x = np.ascontiguousarray(x[-1])
        cond, eos_logit = self._condition(x)
        eos_cutoff = math.log(eos_thresh / (1 - eos_thresh))

        out = np.empty((max_frames, self.latent_dim), FLOAT32)
        num_frames = 0
        w_in, b_in = self._lat_in
        for frame_idx in range(max_frames):
            if frame_idx > max(3, min_frames) and eos_logit > eos_cutoff:
                break
            noise = self._rng.randn(self.latent_dim) * FLOAT32(temp)
            latent = noise + self._flow_head(cond, noise)
            out[frame_idx] = latent
            num_frames = frame_idx + 1
            x = w_in @ (latent if force is None else force[frame_idx]) + b_in
            for layer_idx in range(self.num_layers):
                x = self._step_block(layer_idx, x, prefix_len + frame_idx)
            cond, eos_logit = self._condition(x)
        return out[:num_frames].copy()


class Conv1dLayer:
    """Causal streaming 1D convolution layer."""

    def __init__(self, state: dict[str, Any], lazy: bool) -> None:
        self._state_dict = state
        self._lazy = lazy
        weight_meta = state["w"]
        shape = (
            weight_meta["shape"] if isinstance(weight_meta, dict) else weight_meta.shape
        )
        self._kernel_size = shape[2]
        self._in_channels = shape[1]
        self._bias = state["b"].astype(FLOAT32)
        self._weights_list = None if lazy else self._load_weights()
        self._state = np.zeros((self._kernel_size - 1, self._in_channels), FLOAT32)

    def _load_weights(self) -> list[np.ndarray]:
        weight = WeightLoader.dequantize(self._state_dict["w"])
        return [
            np.ascontiguousarray(weight[:, :, tap].T)
            for tap in range(self._kernel_size)
        ]

    def reset(self) -> None:
        """Resets the causal convolution buffer."""
        self._state = np.zeros((self._kernel_size - 1, self._in_channels), FLOAT32)

    def __call__(self, x: np.ndarray, res: np.ndarray | None = None) -> np.ndarray:
        """Applies causal 1D convolution to ``x`` with optional residual."""
        weights = (
            self._weights_list
            if self._weights_list is not None
            else self._load_weights()
        )
        num_steps = x.shape[0]
        if self._kernel_size > 1:
            x = np.concatenate([self._state, x])
            self._state = x[num_steps:].copy()
        out = x[:num_steps] @ weights[0]
        out += self._bias if res is None else res + self._bias
        for tap in range(1, self._kernel_size):
            out += x[tap : tap + num_steps] @ weights[tap]
        return cast(np.ndarray, out)


class ConvTranspose1dLayer:
    """Streaming 1D transposed convolution layer."""

    def __init__(self, state: dict[str, Any], lazy: bool) -> None:
        self._state_dict = state
        self._lazy = lazy
        self._stride = state["stride"]
        weight_meta = state["w"]
        shape = (
            weight_meta["shape"] if isinstance(weight_meta, dict) else weight_meta.shape
        )
        _, self._out_channels, _ = shape
        self._stride_channels = self._stride * self._out_channels
        self._bias = np.tile(state["b"].astype(FLOAT32), self._stride)
        self._weight = None if lazy else self._load_weights()
        self._carry = np.zeros(self._stride_channels, FLOAT32)

    def _load_weights(self) -> np.ndarray:
        weight = WeightLoader.dequantize(self._state_dict["w"])
        in_ch, out_ch, kernel_size = weight.shape
        return np.ascontiguousarray(
            weight.transpose(0, 2, 1).reshape(in_ch, kernel_size * out_ch)
        )

    def reset(self) -> None:
        """Resets the overlap-add carry buffer."""
        self._carry = np.zeros(self._stride_channels, FLOAT32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies streaming 1D transposed convolution to ``x``."""
        weight = self._weight if self._weight is not None else self._load_weights()
        num_steps = x.shape[0]
        proj = x @ weight
        out = proj[:, : self._stride_channels] + self._bias
        out[0] += self._carry
        out[1:] += proj[:-1, self._stride_channels :]
        self._carry = proj[-1, self._stride_channels :].copy()
        return cast(
            np.ndarray, out.reshape(num_steps * self._stride, self._out_channels)
        )


class AudioDecoder:
    """Mimi-adapted transformer and SEANet convolutional audio decoder."""

    def __init__(
        self,
        state: dict[str, Any],
        chunk: int = 8,
        split: int = 6,
        mode: str = "fp32",
    ) -> None:
        """Initializes the audio decoder."""
        cfg = state["cfg"]
        self._num_heads = cfg["heads"]
        self._head_dim = cfg["head_dim"]
        self._window = cfg["window"]
        self._eps = FLOAT32(cfg["eps"])
        self._chunk = chunk
        self._split = split
        self._lazy = mode == "int8"

        self._inp_w = WeightLoader.dequantize(state["inp"][0]).T.copy()
        self._inp_b = WeightLoader.dequantize(state["inp"][1])
        self._inv_freq = state["inv_freq"].astype(FLOAT32)
        self._tf_state = state["tf"]
        self._tf_layers = (
            None
            if self._lazy
            else [self._load_tf_layer(layer) for layer in state["tf"]]
        )
        self._dim = self._num_heads * self._head_dim
        self._ops: list[tuple[str, Any]] = []
        for idx, (kind, params) in enumerate(state["seanet"]):
            lazy = self._lazy and idx < split
            op: Any
            if kind == "conv":
                op = Conv1dLayer(params, lazy)
            elif kind == "convt":
                op = ConvTranspose1dLayer(params, lazy)
            elif kind == "res":
                op = (Conv1dLayer(params[0], lazy), Conv1dLayer(params[1], lazy))
            else:
                op = None
            self._ops.append((kind, op))

    @staticmethod
    def _load_tf_layer(layer: dict[str, Any]) -> dict[str, Any]:
        return {
            "ln1": [WeightLoader.dequantize(t) for t in layer["ln1"]],
            "ln2": [WeightLoader.dequantize(t) for t in layer["ln2"]],
            **{
                key: np.ascontiguousarray(WeightLoader.dequantize(layer[key]).T)
                for key in ("qkv", "o", "fc1", "fc2")
            },
        }

    def _layer_norm(self, hidden: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
        mean = hidden.mean(-1, keepdims=True)
        centered = hidden - mean
        normed = centered / np.sqrt(
            (centered * centered).mean(-1, keepdims=True) + self._eps
        )
        return cast(np.ndarray, normed * weights[0] + weights[1])

    def transformer(self, hidden: np.ndarray) -> np.ndarray:
        """Runs the windowed causal transformer over decoded latents."""
        num_steps = hidden.shape[0]
        freqs = np.arange(num_steps, dtype=FLOAT32)[:, None] * self._inv_freq
        emb = np.concatenate([freqs, freqs], -1)
        cos_emb, sin_emb = np.cos(emb), np.sin(emb)
        indices = np.arange(num_steps)
        masked = (indices[None] > indices[:, None]) | (
            indices[None] <= indices[:, None] - self._window
        )
        half_dim = self._head_dim // 2

        layers = self._tf_layers if self._tf_layers is not None else self._tf_state
        for layer_state in layers:
            layer = (
                layer_state
                if self._tf_layers is not None
                else self._load_tf_layer(layer_state)
            )
            normed = self._layer_norm(hidden, layer["ln1"])
            qkv = (
                (normed @ layer["qkv"])
                .reshape(num_steps, 3, self._num_heads, self._head_dim)
                .transpose(1, 2, 0, 3)
            )
            qk = qkv[:2]
            qk_rot = np.concatenate([-qk[..., half_dim:], qk[..., :half_dim]], -1)
            qk = qk * cos_emb + qk_rot * sin_emb
            scores = np.matmul(qk[0], qk[1].transpose(0, 2, 1)) * FLOAT32(
                1 / math.sqrt(self._head_dim)
            )
            scores[:, masked] = -np.inf
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            scores /= scores.sum(-1, keepdims=True)
            attn = (
                np.matmul(scores, qkv[2])
                .transpose(1, 0, 2)
                .reshape(num_steps, self._dim)
            )
            hidden = hidden + attn @ layer["o"]
            normed = self._layer_norm(hidden, layer["ln2"])
            hidden = hidden + _gelu(normed @ layer["fc1"]) @ layer["fc2"]
        return hidden

    @staticmethod
    def _run_ops(x: np.ndarray, ops: list[tuple[str, Any]]) -> np.ndarray:
        for kind, op in ops:
            if kind == "elu":
                x = _elu_inplace(x)
            elif kind == "res":
                x = op[1](_elu_inplace(op[0](_elu_inplace(x.copy()))), res=x)
            else:
                x = op(x)
        return x

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        """Decodes latent frames into 24 kHz mono float32 waveform samples."""
        hidden = self.transformer(latents @ self._inp_w + self._inp_b)
        for kind, op in self._ops:
            if kind == "res":
                op[0].reset()
                op[1].reset()
            elif op is not None:
                op.reset()
        x = self._run_ops(hidden, self._ops[: self._split])
        rows = self._chunk * x.shape[0] // hidden.shape[0]
        chunks = [
            self._run_ops(
                np.ascontiguousarray(x[idx : idx + rows]),
                self._ops[self._split :],
            )
            for idx in range(0, x.shape[0], rows)
        ]
        return np.concatenate(chunks)[:, 0]


class LocalInt8FlowBackend:
    """Default pure-NumPy int8 flow TTS backend."""

    def __init__(
        self,
        weights_path: str,
        voice: str = "juno",
        dec_mode: str = "fp32",
    ) -> None:
        """Loads quantized weights and initializes speech and decoder models."""
        with np.load(weights_path) as archive:
            state = WeightLoader.unflatten(archive)
        available = tuple(sorted(state["voices"].keys()))
        if voice not in state["voices"]:
            raise ValueError(
                f"Unsupported voice {voice!r}; expected one of {available}."
            )
        self._voices = available
        self._lock = threading.Lock()
        self.rng = TorchRandn(0)
        self.speech = SpeechModel(state["speech"], rng=self.rng)
        self.decoder = AudioDecoder(state["decoder"], mode=dec_mode)
        self.mean = state["mean"].astype(FLOAT32)
        self.std = state["std"].astype(FLOAT32)
        self.voice = state["voices"][voice].astype(FLOAT32)
        self.frames_max = state["max_frames"]
        self.fps = state["fps"]
        self.t_speech = 0.0
        self.t_dec = 0.0

    @property
    def sample_rate(self) -> int:
        """Native output sample rate in Hz (24 kHz)."""
        return SAMPLE_RATE

    @property
    def voices(self) -> tuple[str, ...]:
        """Available speaker voices stored in the weight archive."""
        return self._voices

    @property
    def max_frames(self) -> int:
        """Maximum latent frame budget per piece."""
        return int(self.frames_max)

    def seed(self, seed_value: int) -> None:
        """Reseeds the engine's random number generator."""
        with self._lock:
            self.rng.seed(seed_value)

    def latents(
        self,
        text: str,
        temp: float = 0.5,
        eos: float = 0.9,
        min_fpc: float = 0.0,
    ) -> np.ndarray:
        """Generates acoustic latents for ``text``."""
        return self.speech.generate(
            self.voice,
            SpeechModel.encode_text(text),
            self.frames_max,
            temp,
            eos,
            min_fpc * self.fps / 12.5,
        )

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decodes unnormalized latents into waveform samples."""
        return self.decoder(latents * self.std + self.mean)

    def synthesize_raw(
        self,
        text: str,
        *,
        temp: float = 0.5,
        eos: float = 0.9,
    ) -> tuple[np.ndarray, int]:
        """Synthesizes ``text`` to ``(waveform_1d_float32, frame_count)``."""
        with self._lock:
            t0 = time.perf_counter()
            lat = self.latents(text, temp=temp, eos=eos)
            t1 = time.perf_counter()
            wav = self.decode(lat)
            self.t_speech += t1 - t0
            self.t_dec += time.perf_counter() - t1
            return wav, int(len(lat))

    def __call__(self, text: str, **kwargs: Any) -> np.ndarray:
        """Synthesizes ``text`` directly to waveform samples."""
        with self._lock:
            t0 = time.perf_counter()
            lat = self.latents(text, **kwargs)
            t1 = time.perf_counter()
            wav = self.decode(lat)
            self.t_speech += t1 - t0
            self.t_dec += time.perf_counter() - t1
            return wav

    @staticmethod
    def to_pcm16(audio: np.ndarray) -> bytes:
        """Converts float32 mono audio in [-1, 1] to little-endian 16-bit PCM."""
        return cast(
            bytes,
            (np.clip(audio, -1, 1) * 32767).round().astype("<i2").tobytes(),
        )

    @classmethod
    def write_wav(cls, path: str, audio: np.ndarray) -> None:
        """Writes float32 mono audio to a 24 kHz 16-bit PCM WAV file."""
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(cls.to_pcm16(audio))


def to_pcm16(audio: np.ndarray) -> bytes:
    """Converts float32 mono audio in [-1, 1] to little-endian 16-bit PCM."""
    return LocalInt8FlowBackend.to_pcm16(audio)


def write_wav(path: str, audio: np.ndarray) -> None:
    """Writes float32 mono audio to a 24 kHz 16-bit PCM WAV file."""
    LocalInt8FlowBackend.write_wav(path, audio)
