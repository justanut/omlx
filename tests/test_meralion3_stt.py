# SPDX-License-Identifier: Apache-2.0
"""Tests for MERaLiON-3 ASR support (omlx/patches/mlx_audio_meralion3).

Discovery tests only parse config.json. Model tests build a tiny randomly
initialised MERaLiON-3 (same topology, toy sizes) and run it through
mlx-audio's real ``load_model`` path; they need mlx, mlx-lm, mlx-audio and
transformers but no downloaded checkpoint.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from omlx.model_discovery import detect_model_type, discover_models

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _write_config(path: Path, config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))


_SPEECH_CONFIG = {
    "model_type": "whisper",
    "architectures": ["WhisperForConditionalGeneration"],
    "d_model": 1280,
    "encoder_layers": 32,
    "num_mel_bins": 128,
}
_TEXT_CONFIG = {
    "model_type": "gemma2",
    "architectures": ["Gemma2ForCausalLM"],
    "hidden_size": 2304,
}


@pytest.mark.parametrize(
    "model_type,arch",
    [
        ("meralion3", "MERaLiON3ForConditionalGeneration"),
        ("meralion3_asr", "MERaLiON3ASRForConditionalGeneration"),
    ],
)
def test_meralion3_detected_as_stt(tmp_path, model_type, arch):
    _write_config(
        tmp_path,
        {
            "model_type": model_type,
            "architectures": [arch],
            "speech_config": _SPEECH_CONFIG,
            "text_config": _TEXT_CONFIG,
        },
    )
    assert detect_model_type(tmp_path) == "audio_stt"


@pytest.mark.parametrize("model_type", ["meralion3", "meralion3_asr"])
def test_meralion3_model_type_without_architectures(tmp_path, model_type):
    _write_config(tmp_path, {"model_type": model_type})
    assert detect_model_type(tmp_path) == "audio_stt"


def test_meralion3_discovered_with_stt_engine(tmp_path):
    model_dir = tmp_path / "MERaLiON-3-3B-ASR-MLX-8bit"
    _write_config(
        model_dir,
        {
            "model_type": "meralion3",
            "architectures": ["MERaLiON3ForConditionalGeneration"],
            "speech_config": _SPEECH_CONFIG,
            "text_config": _TEXT_CONFIG,
        },
    )
    (model_dir / "model.safetensors").write_bytes(b"0" * 1000)
    models = discover_models(tmp_path)
    info = models["MERaLiON-3-3B-ASR-MLX-8bit"]
    assert info.model_type == "audio_stt"
    assert info.engine_type == "audio_stt"


# ---------------------------------------------------------------------------
# Model (tiny random weights)
# ---------------------------------------------------------------------------

SPEECH_TOKEN = 255999
N_SPEECH = 300

_TINY_CONFIG = {
    "model_type": "meralion3",
    "architectures": ["MERaLiON3ForConditionalGeneration"],
    "fixed_speech_embeds_length": N_SPEECH,
    "speech_mlp_scale_factor": 5,
    "speech_mlp_use_projection": False,
    "speech_token_index": SPEECH_TOKEN,
    "speech_config": {
        "model_type": "whisper",
        "d_model": 64,
        "encoder_layers": 2,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 128,
        "num_mel_bins": 128,
        "max_source_positions": 1500,
    },
    "text_config": {
        "model_type": "gemma2",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 256000,
        "rms_norm_eps": 1e-6,
        "query_pre_attn_scalar": 16,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
    },
    "quantization": {"group_size": 64, "bits": 8},
}

# <bos><start_of_turn>user\n ... <SpeechHere><end_of_turn>\n<start_of_turn>model\n
_PROMPT_IDS = [2, 106, 1645, 108, 37854, 235292, 235248, SPEECH_TOKEN, 107, 108, 106, 2516, 108]


class _StubTokenizer:
    """Minimal tokenizer: token id -> 'w<id> '."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"w{i} " for i in ids)


@pytest.fixture(scope="module")
def mlx_modules():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("mlx_audio.stt.utils")
    pytest.importorskip("transformers")
    return mx


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory, mlx_modules):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from omlx.patches.mlx_audio_meralion3.meralion3 import Model, ModelConfig

    mx.random.seed(0)
    path = tmp_path_factory.mktemp("meralion3_tiny")
    model = Model(ModelConfig.from_dict(_TINY_CONFIG))
    nn.quantize(model.text_decoder, group_size=64, bits=8)
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if k in ("speech_encoder.conv1.weight", "speech_encoder.conv2.weight"):
            v = v.transpose(0, 2, 1)  # store in PyTorch (out, in, k) layout
        weights[k] = v
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(_TINY_CONFIG))
    return path, dict(tree_flatten(model.parameters()))


def _load(path):
    from mlx_audio.stt.utils import load_model

    from omlx.patches.mlx_audio_meralion3.meralion3 import Model

    original = Model.post_load_hook
    Model.post_load_hook = classmethod(lambda cls, m, p: m)
    try:
        model = load_model(str(path))
    finally:
        Model.post_load_hook = original

    from transformers import WhisperFeatureExtractor

    model._tokenizer = _StubTokenizer()
    model._feature_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400
    )
    i = _PROMPT_IDS.index(SPEECH_TOKEN)
    model._prompt_ids = (
        _PROMPT_IDS[:i] + [SPEECH_TOKEN] * N_SPEECH + _PROMPT_IDS[i + 1 :]
    )
    return model


def test_patch_registers_both_model_types(mlx_modules):
    import sys

    import mlx_audio.stt.utils as stt_utils

    from omlx.patches.mlx_audio_meralion3 import apply_mlx_audio_meralion3_patch

    assert apply_mlx_audio_meralion3_patch()
    for name in ("meralion3", "meralion3_asr"):
        assert stt_utils.MODEL_REMAPPING[name] == name
        assert f"mlx_audio.stt.models.{name}" in sys.modules


def test_sanitize_transposes_torch_conv_only():
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_audio_meralion3.meralion3 import Model

    torch_layout = {
        "speech_encoder.conv1.weight": mx.zeros((1280, 128, 3)),
        "speech_encoder.conv2.weight": mx.zeros((1280, 1280, 3)),
        "text_decoder.lm_head.weight": mx.zeros((4, 4)),
    }
    out = Model.sanitize(torch_layout)
    assert out["speech_encoder.conv1.weight"].shape == (1280, 3, 128)
    assert out["speech_encoder.conv2.weight"].shape == (1280, 3, 1280)
    assert "text_decoder.lm_head.weight" not in out
    # Already-converted weights are left alone.
    again = Model.sanitize(out)
    assert again["speech_encoder.conv1.weight"].shape == (1280, 3, 128)


def test_load_quantizes_decoder_only(tiny_checkpoint):
    import mlx.core as mx
    import mlx.nn as nn

    from omlx.patches.mlx_audio_meralion3 import apply_mlx_audio_meralion3_patch

    apply_mlx_audio_meralion3_patch()
    path, reference = tiny_checkpoint
    model = _load(path)

    assert isinstance(model.text_decoder.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.speech_encoder.layers[0].fc1, nn.Linear)
    assert not isinstance(model.speech_encoder.layers[0].fc1, nn.QuantizedLinear)
    from mlx.utils import tree_flatten

    loaded = dict(tree_flatten(model.parameters()))
    for k, v in reference.items():
        assert mx.array_equal(loaded[k], v), k


def test_encode_speech_shape(tiny_checkpoint):
    import mlx.core as mx

    from omlx.patches.mlx_audio_meralion3 import apply_mlx_audio_meralion3_patch

    apply_mlx_audio_meralion3_patch()
    model = _load(tiny_checkpoint[0])
    speech = model.encode_speech(mx.zeros((2, 3000, 128)))
    assert speech.shape == (2, N_SPEECH, 64)


def test_generate_chunks_and_stream(tiny_checkpoint):
    from omlx.patches.mlx_audio_meralion3 import apply_mlx_audio_meralion3_patch

    apply_mlx_audio_meralion3_patch()
    model = _load(tiny_checkpoint[0])
    audio = np.random.RandomState(0).randn(16000 * 35).astype(np.float32) * 0.1

    out = model.generate(audio, max_new_tokens=4)
    assert [(s["start"], s["end"]) for s in out.segments] == [(0.0, 30.0), (30.0, 35.0)]
    assert out.prompt_tokens == 2 * len(model._prompt_ids)
    assert 0 < out.generation_tokens <= 8
    assert out.text == " ".join(s["text"] for s in out.segments if s["text"])

    streamed = list(model.generate(audio, max_new_tokens=4, stream=True))
    assert "".join(c.text for c in streamed) == out.text


def test_no_repeat_ngram_blocks_repeats(tiny_checkpoint):
    """With n=2, no bigram may repeat in the generated tokens."""
    from omlx.patches.mlx_audio_meralion3 import apply_mlx_audio_meralion3_patch

    apply_mlx_audio_meralion3_patch()
    model = _load(tiny_checkpoint[0])
    import mlx.core as mx

    speech = model.encode_speech(model._mel(np.zeros(16000, np.float32)))
    toks = list(model._generate_tokens(speech, 40, 2, 1.0))
    seq = model._prompt_ids + toks
    bigrams = list(zip(seq, seq[1:]))
    # Only bigrams introduced by generated tokens are constrained.
    gen_bigrams = bigrams[len(model._prompt_ids) - 1 :]
    assert len(gen_bigrams) == len(set(gen_bigrams))
    assert all(b not in set(bigrams[: len(model._prompt_ids) - 1]) for b in gen_bigrams)
    mx.clear_cache()


def test_postprocess_strips_speaker_tag():
    pytest.importorskip("mlx.core")
    from omlx.patches.mlx_audio_meralion3.meralion3 import Model

    assert Model._postprocess("  <Speaker1>: hello lah ") == "hello lah"
    assert Model._postprocess("Speaker 1: ok") == "ok"


def test_max_tokens_and_repetition_penalty(tiny_checkpoint):
    from omlx.patches.mlx_audio_meralion3 import apply_mlx_audio_meralion3_patch

    apply_mlx_audio_meralion3_patch()
    model = _load(tiny_checkpoint[0])
    audio = np.zeros(16000, np.float32)
    out = model.generate(audio, max_tokens=3, repetition_penalty=1.05)
    assert out.generation_tokens <= 3
    # A huge oMLX max_tokens never exceeds the per-chunk reference budget.
    out = model.generate(audio, max_tokens=65536, max_new_tokens=5)
    assert out.generation_tokens <= 5
