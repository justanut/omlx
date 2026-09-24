# SPDX-License-Identifier: Apache-2.0
"""MLX port of MERaLiON-3 ASR (``MERaLiON/MERaLiON-3-3B-ASR``).

Architecture (see upstream ``modeling_meralion3.py``):

    30 s log-mel (128 bins, 3000 frames)
      -> Whisper-large-v3 encoder            (speech_encoder.*)   1500 x 1280
      -> LayerNorm                           (ln_speech.*)
      -> fold 5 frames, Linear+SiLU, Linear  (speech_audio_adapter.*) 300 x 2304
      -> scattered into the prompt at the ``<SpeechHere>`` token positions
      -> Gemma-2 2B decoder                  (text_decoder.*)

Every <= 30 s chunk is decoded independently with the fixed ASR prompt and the
per-chunk transcripts are concatenated, matching the reference
``meralion-3-asr`` package (greedy, ``max_new_tokens=512``,
``no_repeat_ngram_size=6``).

The module follows the mlx-audio STT model contract (``Model``,
``ModelConfig``, ``sanitize``, ``post_load_hook``, ``generate`` returning
``STTOutput``) so it loads through ``mlx_audio.stt.utils.load_model``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE

ASR_CONTENT = (
    "Instruction: Please transcribe this speech. \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)
# Rendered form of the model's chat template for a single user turn. Used
# only when the tokenizer has no chat template of its own.
_FALLBACK_PROMPT = (
    "<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
)
_SPEAKER_TAGS = ("<Speaker1>:", "<Speaker1>", "Speaker 1:")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SpeechConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ModelConfig:
    model_type: str = "meralion3_asr"
    speech_config: SpeechConfig = field(default_factory=SpeechConfig)
    text_config: Dict[str, Any] = field(default_factory=dict)
    speech_mlp_scale_factor: int = 5
    speech_mlp_use_projection: bool = False
    speech_token_index: int = 255999
    fixed_speech_embeds_length: int = 300
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(
            model_type=d.get("model_type", "meralion3_asr"),
            speech_config=SpeechConfig.from_dict(d.get("speech_config", {})),
            text_config=dict(d.get("text_config", {})),
            speech_mlp_scale_factor=d.get("speech_mlp_scale_factor", 5),
            speech_mlp_use_projection=d.get("speech_mlp_use_projection", False),
            speech_token_index=d.get("speech_token_index", 255999),
            fixed_speech_embeds_length=d.get("fixed_speech_embeds_length", 300),
            model_path=d.get("model_path"),
        )


# ---------------------------------------------------------------------------
# Whisper encoder (HF weight names: speech_encoder.*)
# ---------------------------------------------------------------------------


class WhisperAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class WhisperEncoderLayer(nn.Module):
    def __init__(self, cfg: SpeechConfig):
        super().__init__()
        d = cfg.d_model
        self.self_attn = WhisperAttention(d, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class WhisperEncoder(nn.Module):
    def __init__(self, cfg: SpeechConfig):
        super().__init__()
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d)
        self.layers = [WhisperEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        self.layer_norm = nn.LayerNorm(d)

    def __call__(self, mel: mx.array) -> mx.array:
        """``mel``: (B, 3000, n_mels) channels-last."""
        x = nn.gelu(self.conv1(mel))
        x = nn.gelu(self.conv2(x))
        x = x + self.embed_positions.weight[: x.shape[1]].astype(x.dtype)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# Speech adapter (speech_audio_adapter.*)
# ---------------------------------------------------------------------------


class SpeechAudioAdapter(nn.Module):
    """``MERaLiON3SpeechAudioAdaperLarge``: fold, Linear+SiLU, (gate), Linear."""

    def __init__(self, config: ModelConfig, text_hidden: int):
        super().__init__()
        c = config.speech_config.d_model
        self.scale_factor = config.speech_mlp_scale_factor
        self.use_projection = config.speech_mlp_use_projection
        # nn.Sequential(Linear, SiLU, Dropout) -> keys mlp_adapter.0.*
        self.mlp_adapter = [nn.Linear(c * self.scale_factor, c * 5)]
        if self.use_projection:
            self.gate_proj = nn.Linear(c * 5, c * 5)
            self.pool_proj = nn.Linear(c * 5, c * 5)
        self.out_proj = nn.Linear(c * 5, text_hidden)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        x = x.reshape(B, T // self.scale_factor, C * self.scale_factor)
        x = nn.silu(self.mlp_adapter[0](x))
        if self.use_projection:
            x = nn.silu(self.gate_proj(x)) * self.pool_proj(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


def _build_text_decoder(text_config: Dict[str, Any]) -> nn.Module:
    from mlx_lm.models.gemma2 import Model as Gemma2Model
    from mlx_lm.models.gemma2 import ModelArgs as Gemma2Args

    cfg = dict(text_config)
    cfg.setdefault("model_type", "gemma2")
    return Gemma2Model(Gemma2Args.from_dict(cfg))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.speech_encoder = WhisperEncoder(config.speech_config)
        self.ln_speech = nn.LayerNorm(config.speech_config.d_model)
        self.text_decoder = _build_text_decoder(config.text_config)
        text_hidden = self.text_decoder.args.hidden_size
        self.speech_audio_adapter = SpeechAudioAdapter(config, text_hidden)

        self._tokenizer = None
        self._feature_extractor = None
        self._prompt_ids: Optional[List[int]] = None
        self.eos_token_ids: set = {1, 107}  # <eos>, <end_of_turn>

    # -- loading -----------------------------------------------------------

    @property
    def layers(self):
        return self.text_decoder.layers

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            if k.startswith("text_decoder.lm_head."):
                continue  # tied to embed_tokens
            if k.endswith("num_batches_tracked"):
                continue
            if k in ("speech_encoder.conv1.weight", "speech_encoder.conv2.weight"):
                # PyTorch Conv1d (out, in, k) -> MLX (out, k, in). Only
                # transpose checkpoints that are still in torch layout.
                if v.ndim == 3 and v.shape[1] != 3 and v.shape[2] == 3:
                    v = v.transpose(0, 2, 1)
            out[k] = v
        return out

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        # Only the Gemma-2 decoder is ever quantized; the audio path stays
        # in its original precision.
        return path.startswith("text_decoder.")

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Union[str, Path]) -> "Model":
        model._load_processors(Path(model_path))
        return model

    def _load_processors(self, model_path: Path) -> None:
        from transformers import WhisperFeatureExtractor

        try:
            # Load the Gemma tokenizer class directly: AutoTokenizer would
            # consult config.json, whose auto_map points at remote code and
            # triggers an interactive trust_remote_code prompt.
            from transformers import GemmaTokenizerFast as _Tok
        except ImportError:  # pragma: no cover - very old/new transformers
            from transformers import AutoTokenizer as _Tok

        try:
            self._tokenizer = _Tok.from_pretrained(str(model_path))
        except Exception:  # noqa: BLE001
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=False
            )
        self._feature_extractor = WhisperFeatureExtractor(
            feature_size=self.config.speech_config.num_mel_bins,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            chunk_length=CHUNK_SECONDS,
            n_fft=400,
            padding_value=0.0,
        )
        for tok in ("<eos>", "<end_of_turn>"):
            tid = self._tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                self.eos_token_ids.add(tid)
        self._prompt_ids = self._build_prompt_ids()

    def _build_prompt_ids(self) -> List[int]:
        tok = self._tokenizer
        prompt = None
        if getattr(tok, "chat_template", None):
            try:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": ASR_CONTENT}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                prompt = None
        if not prompt:
            prompt = _FALLBACK_PROMPT.format(content=ASR_CONTENT)
        ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
        n_speech = ids.count(self.config.speech_token_index)
        if n_speech != 1:
            raise RuntimeError(
                f"MERaLiON-3 prompt must contain exactly one <SpeechHere> token "
                f"(id {self.config.speech_token_index}); found {n_speech}"
            )
        i = ids.index(self.config.speech_token_index)
        n = self.config.fixed_speech_embeds_length
        return ids[:i] + [self.config.speech_token_index] * n + ids[i + 1 :]

    # -- features ----------------------------------------------------------

    def _mel(self, chunk: np.ndarray) -> mx.array:
        """Whisper log-mel with zero-mean/unit-var waveform normalisation,
        padded to 30 s — identical to the upstream processor."""
        feats = self._feature_extractor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            do_normalize=True,
            return_tensors="np",
        )["input_features"]  # (1, n_mels, 3000)
        return mx.array(feats.transpose(0, 2, 1))

    def encode_speech(self, mel: mx.array) -> mx.array:
        """mel (B, 3000, n_mels) -> speech embeddings (B, 300, hidden)."""
        dtype = self.speech_encoder.conv1.weight.dtype
        x = self.speech_encoder(mel.astype(dtype))
        x = self.ln_speech(x)
        return self.speech_audio_adapter(x)

    # -- decoder -----------------------------------------------------------

    def _decode(self, h: mx.array, cache) -> mx.array:
        """Run Gemma-2 on input embeddings ``h`` (unscaled)."""
        from mlx_lm.models.base import create_attention_mask

        lm = self.text_decoder
        h = h * (lm.args.hidden_size**0.5)
        mask = create_attention_mask(h, cache[0], return_array=True)
        for layer, c in zip(lm.model.layers, cache):
            h = layer(h, mask, c)
        h = lm.model.norm(h)
        logits = lm.model.embed_tokens.as_linear(h[:, -1:, :])
        cap = lm.final_logit_softcapping
        return mx.tanh(logits / cap) * cap

    def _embed_prompt(self, speech: mx.array) -> mx.array:
        ids = mx.array(self._prompt_ids)[None]
        embeds = self.text_decoder.model.embed_tokens(ids)
        speech = speech.astype(embeds.dtype)
        mask = np.array(self._prompt_ids) == self.config.speech_token_index
        start = int(np.argmax(mask))
        n = int(mask.sum())
        return mx.concatenate(
            [embeds[:, :start], speech[:, :n], embeds[:, start + n :]], axis=1
        )

    def _generate_tokens(
        self,
        speech: mx.array,
        max_new_tokens: int,
        no_repeat_ngram_size: int,
        repetition_penalty: float,
    ) -> Generator[int, None, None]:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.text_decoder)
        logits = self._decode(self._embed_prompt(speech), cache)
        history = list(self._prompt_ids)
        ngrams: Dict[tuple, set] = {}
        n = no_repeat_ngram_size
        if n > 0:
            for i in range(len(history) - n + 1):
                ngrams.setdefault(tuple(history[i : i + n - 1]), set()).add(
                    history[i + n - 1]
                )
        generated: List[int] = []

        for _ in range(max_new_tokens):
            scores = logits[0, -1].astype(mx.float32)
            if repetition_penalty and repetition_penalty != 1.0 and generated:
                idx = mx.array(sorted(set(generated)))
                sel = scores[idx]
                sel = mx.where(
                    sel > 0, sel / repetition_penalty, sel * repetition_penalty
                )
                scores[idx] = sel
            banned = ngrams.get(tuple(history[-(n - 1):])) if n > 1 else None
            if banned:
                scores[mx.array(sorted(banned))] = -mx.inf
            token = int(mx.argmax(scores).item())
            if token in self.eos_token_ids:
                break
            generated.append(token)
            history.append(token)
            if n > 0 and len(history) >= n:
                ngrams.setdefault(tuple(history[-n:-1]), set()).add(token)
            yield token
            logits = self._decode(
                self.text_decoder.model.embed_tokens(mx.array([[token]])), cache
            )

    # -- audio -------------------------------------------------------------

    @staticmethod
    def _load_audio(audio: Union[str, Path, np.ndarray, mx.array]) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio

            audio = load_audio(str(audio), sr=SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                audio = audio.T
            audio = audio.mean(axis=-1)
        if audio.ndim != 1:
            raise ValueError(f"Unsupported audio shape {audio.shape}")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _chunks(audio: np.ndarray) -> List[tuple]:
        if audio.shape[0] <= CHUNK_SAMPLES:
            return [(audio, 0.0)]
        return [
            (audio[s : s + CHUNK_SAMPLES], s / SAMPLE_RATE)
            for s in range(0, audio.shape[0], CHUNK_SAMPLES)
        ]

    @staticmethod
    def _postprocess(text: str) -> str:
        text = text.strip()
        for tag in _SPEAKER_TAGS:
            if text.startswith(tag):
                text = text[len(tag) :].lstrip()
        return text

    # -- public API --------------------------------------------------------

    def generate(
        self,
        audio: Union[str, Path, np.ndarray, mx.array],
        *,
        max_new_tokens: int = 512,
        max_tokens: Optional[int] = None,
        no_repeat_ngram_size: int = 6,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Transcribe ``audio`` (path, or 16 kHz mono samples).

        ``language`` and other mlx-audio kwargs are accepted and ignored: the
        model detects the language itself and the prompt is fixed.
        """
        if self._tokenizer is None:
            if self.config.model_path is None:
                raise RuntimeError("MERaLiON-3 tokenizer not loaded")
            self._load_processors(Path(self.config.model_path))
        if max_tokens is not None and max_tokens > 0:
            # oMLX forwards request/per-model ``max_tokens`` (which can be very
            # large for other STT backends). Here it bounds each 30 s chunk,
            # and never beyond the reference per-chunk budget.
            max_new_tokens = min(int(max_tokens), max_new_tokens)
        gen = self._run(
            audio, max_new_tokens, no_repeat_ngram_size, repetition_penalty, verbose
        )
        if stream:
            return self._stream(gen)
        final = None
        for item in gen:
            final = item
        return final

    def _stream(self, gen: Iterable) -> Generator:
        from mlx_audio.stt.models.base import STTOutput

        for item in gen:
            if isinstance(item, str):
                yield STTOutput(text=item)

    def _run(
        self,
        audio,
        max_new_tokens: int,
        no_repeat_ngram_size: int,
        repetition_penalty: float,
        verbose: bool,
    ) -> Generator:
        """Yield text deltas (str) while decoding, then a final STTOutput."""
        from mlx_audio.stt.models.base import STTOutput

        t0 = time.perf_counter()
        samples = self._load_audio(audio)
        chunks = self._chunks(samples)
        segments = []
        texts: List[str] = []
        prompt_tokens = 0
        gen_tokens = 0
        tok = self._tokenizer

        for idx, (chunk, offset) in enumerate(chunks):
            speech = self.encode_speech(self._mel(chunk))
            prompt_tokens += len(self._prompt_ids)
            ids: List[int] = []
            emitted = ""
            sep = " " if texts else ""
            for token in self._generate_tokens(
                speech, max_new_tokens, no_repeat_ngram_size, repetition_penalty
            ):
                ids.append(token)
                text = self._postprocess(tok.decode(ids, skip_special_tokens=True))
                # Hold back incomplete multi-byte characters.
                if text.endswith("�") or not text.startswith(emitted):
                    continue
                delta = text[len(emitted) :]
                if delta:
                    yield (sep + delta) if not emitted else delta
                    emitted = text
            gen_tokens += len(ids)
            text = self._postprocess(tok.decode(ids, skip_special_tokens=True))
            if text.startswith(emitted) and len(text) > len(emitted):
                tail = text[len(emitted) :]
                yield (sep + tail) if not emitted else tail
            if verbose:
                print(f"[{offset:7.2f}s] {text}")
            if text:
                texts.append(text)
            segments.append(
                {
                    "id": idx,
                    "start": offset,
                    "end": offset + chunk.shape[0] / SAMPLE_RATE,
                    "text": text,
                }
            )
            mx.clear_cache()

        total = time.perf_counter() - t0
        yield STTOutput(
            text=" ".join(texts),
            segments=segments,
            language=None,
            prompt_tokens=prompt_tokens,
            generation_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
            generation_tps=gen_tokens / total if total > 0 else 0.0,
            total_time=total,
        )
