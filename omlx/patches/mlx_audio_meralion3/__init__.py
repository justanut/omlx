# SPDX-License-Identifier: Apache-2.0
"""Register the MERaLiON-3 ASR model with mlx-audio's STT loader.

mlx-audio resolves a checkpoint's ``model_type`` through
``mlx_audio.stt.utils.MODEL_REMAPPING`` and then imports
``mlx_audio.stt.models.<name>``. MERaLiON-3 is not part of mlx-audio, so this
patch exposes oMLX's MLX port under both model_type spellings used by
published checkpoints:

* ``meralion3``      – original upstream revision and the majentik MLX quants
* ``meralion3_asr``  – current upstream revision (renamed 2026-06-06)
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

MODEL_TYPES = ("meralion3", "meralion3_asr")
ARCHITECTURES = (
    "MERaLiON3ForConditionalGeneration",
    "MERaLiON3ASRForConditionalGeneration",
)

_APPLIED = False


def apply_mlx_audio_meralion3_patch() -> bool:
    """Make ``mlx_audio.stt.utils.load_model`` able to load MERaLiON-3.

    Returns True when the patch is active (idempotent).
    """
    global _APPLIED
    if _APPLIED:
        return True

    try:
        import mlx_audio.stt.models  # noqa: F401
        import mlx_audio.stt.utils as stt_utils

        from . import meralion3 as module
    except Exception as exc:  # noqa: BLE001
        logger.debug("MERaLiON-3 mlx-audio patch unavailable: %s", exc)
        return False

    for name in MODEL_TYPES:
        sys.modules.setdefault(f"mlx_audio.stt.models.{name}", module)
        remapping = getattr(stt_utils, "MODEL_REMAPPING", None)
        if isinstance(remapping, dict):
            remapping.setdefault(name, name)

    _APPLIED = True
    logger.debug("MERaLiON-3 mlx-audio STT patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "ARCHITECTURES",
    "MODEL_TYPES",
    "apply_mlx_audio_meralion3_patch",
    "is_applied",
]
