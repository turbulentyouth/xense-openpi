"""Serving wrapper that applies the tactile refiner on top of a trained VLA policy.

Pipeline per ``infer`` call:

1. Inner policy (absolute action chunk, e.g. [50, 20] bi-Flexiv delta-cartesian).
2. Gripper gating: the refiner only acts while the right gripper is closed
   (``state[19] < gripper_threshold`` -- grasp segment). Outside a grasp the
   inner output passes through unchanged and all refiner state is cleared.
3. Action-space round trip: absolute chunk --(DeltaActions + Normalize)--> the
   normalised delta space the refiner was trained on --> refiner -->
   (Unnormalize + AbsoluteActions)--> absolute again. Norm stats are restricted
   to the ``actions`` key so the raw state used by the delta/absolute conversion
   is never (un)normalised by accident.
4. Safety shaping of the correction: padding dims (20-31) are always masked,
   ``dim_mask`` optionally restricts the valid dims that may be corrected, and
   direction hysteresis locks the sign of the dim-9 (right_tcp.x) correction for
   the duration of a grasp -- a locked direction never flips within a segment;
   a disagreeing frame gets its dim-9 correction zeroed instead.

Tactile history: the last ``HISTORY`` (4) frames of the four tactile views,
encoded by the frozen ImageNet FastViT-T12 (``checkpoints/params.safetensors``,
deliberately NOT the collapsed in-model encoder) into 1024-dim features and
standardised with the train statistics stored in the refiner pickle. Until four
frames have been seen the oldest frame is repeated, matching the frame-clamp
used when the training windows were built. The websocket server never calls
``reset``; the gripper gate is what bounds an episode (gripper open clears the
history and the direction lock).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import logging
import pathlib
from typing import Any, override

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from xense_client import base_policy as _base_policy
from xense_client import image_tools

from openpi import transforms as _transforms
from openpi.models import tactile_refiner as _refiner
from openpi.models.tactile_encoders import build_tactile_encoder

logger = logging.getLogger(__name__)

# Client camera key -> encoder view index, same order as
# openpi.policies.bi_flexiv_policy.BiFlexivTactileInputs (0=left_top,
# 1=left_bottom, 2=right_top, 3=right_bottom).
TACTILE_VIEW_KEYS: tuple[str, ...] = (
    "left_tactile_top",
    "left_tactile_bottom",
    "right_tactile_top",
    "right_tactile_bottom",
)

RIGHT_GRIPPER_DIM = 19  # state layout: right_gripper.pos
DIRECTION_DIM = 9  # action layout: right_tcp.x (delta)
VALID_ACTION_DIMS = 20  # dims 20-31 of the padded model action are padding
IMAGE_SIZE = 224


class TactileRefinerPolicy(_base_policy.BasePolicy):
    """Wraps an inner policy with the tactile refiner (see module docstring)."""

    def __init__(
        self,
        inner: _base_policy.BasePolicy,
        *,
        refiner_params: str | pathlib.Path,
        fastvit_params: str | pathlib.Path,
        norm_stats: dict[str, _transforms.NormStats],
        use_quantile_norm: bool,
        delta_action_mask: Sequence[bool] | None,
        gripper_threshold: float = 0.48,
        dim_mask: Sequence[int] | None = None,
        direction_lock: bool = True,
    ) -> None:
        self._inner = inner
        self._gripper_threshold = gripper_threshold
        self._direction_lock = direction_lock
        if dim_mask is not None:
            bad = [d for d in dim_mask if not 0 <= d < VALID_ACTION_DIMS]
            if bad:
                raise ValueError(f"dim_mask entries must be in [0, {VALID_ACTION_DIMS}), got {bad}")
        self._dim_mask = None if dim_mask is None else sorted(set(int(d) for d in dim_mask))

        # Refiner (frozen) + feature standardisation from the training pickle.
        model, meta = _refiner.load_refiner(refiner_params)
        self._tac_mean = meta["tac_mean"]
        self._tac_std = meta["tac_std"]
        ref_graphdef, self._ref_state = nnx.split(model)

        def refine(state, tac, a):
            return nnx.merge(ref_graphdef, state)(tac, a)

        self._refine = jax.jit(refine)

        # Frozen ImageNet FastViT, same build as the refiner-data pipeline.
        encoder = build_tactile_encoder("fastvit_t12", rngs=nnx.Rngs(0), pretrained_path=str(fastvit_params))
        enc_graphdef, self._enc_state = nnx.split(encoder)

        def encode(state, imgs):
            return nnx.merge(enc_graphdef, state)(imgs)

        self._encode = jax.jit(encode)

        # Action-space round-trip transforms. Only the "actions" stats are used:
        # the state must stay raw for the delta/absolute conversion.
        action_stats = {"actions": norm_stats["actions"]}
        self._to_refiner_space = _transforms.compose(
            [
                _transforms.DeltaActions(delta_action_mask),
                _transforms.Normalize(action_stats, use_quantiles=use_quantile_norm),
            ]
        )
        self._from_refiner_space = _transforms.compose(
            [
                _transforms.Unnormalize(action_stats, use_quantiles=use_quantile_norm),
                _transforms.AbsoluteActions(delta_action_mask),
            ]
        )

        # Episode state (cleared on reset and whenever the gripper opens).
        self._tac_frames: deque[np.ndarray] = deque(maxlen=_refiner.HISTORY)
        self._locked_sign: int = 0
        self._n_refined = 0
        self._n_direction_suppressed = 0
        self._warned_missing_tactile = False

        # Trigger JIT compilation of the two frozen networks at startup instead
        # of inside the first control-loop iteration.
        self._warmup()

    # ------------------------------------------------------------------ #
    # BasePolicy interface                                                #
    # ------------------------------------------------------------------ #

    @override
    def infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        outputs = self._inner.infer(obs, **kwargs)

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        gripper = float(state[RIGHT_GRIPPER_DIM])
        if gripper >= self._gripper_threshold:
            if self._tac_frames or self._locked_sign:
                logger.info(
                    "refiner deactivated (gripper %.3f >= %.3f); refined %d frames this grasp, "
                    "direction-suppressed %d total",
                    gripper,
                    self._gripper_threshold,
                    self._n_refined,
                    self._n_direction_suppressed,
                )
            self._clear_episode_state()
            return outputs

        tac = self._extract_tactile(obs)
        if tac is None:
            return outputs
        self._tac_frames.append(tac)

        # 4-frame history; repeat the oldest frame until the deque is full
        # (same clamp as the training windows: frame max(t-3, 0)).
        frames = list(self._tac_frames)
        while len(frames) < _refiner.HISTORY:
            frames.insert(0, frames[0])
        hist = np.stack(frames)  # [T, views, H, W, C]
        t, nv, h, w, c = hist.shape
        feats = np.asarray(self._encode(self._enc_state, jnp.asarray(hist.reshape(t * nv, h, w, c))), dtype=np.float32)
        tac_hist = (feats.reshape(1, t, nv, -1) - self._tac_mean) / self._tac_std

        # absolute -> normalised delta (the refiner's training space).
        actions_abs = np.asarray(outputs["actions"], dtype=np.float32)
        steps, act_dim = actions_abs.shape
        if steps * 32 != _refiner.ACTION_DIMS or act_dim != VALID_ACTION_DIMS:
            raise RuntimeError(
                f"unexpected inner action shape {actions_abs.shape}; "
                f"expected [steps, {VALID_ACTION_DIMS}] with steps*32 == {_refiner.ACTION_DIMS}"
            )
        fwd = self._to_refiner_space({"state": state.copy(), "actions": actions_abs.copy()})
        a_vla = np.zeros((steps, 32), dtype=np.float32)
        a_vla[:, :VALID_ACTION_DIMS] = fwd["actions"]
        a_vla_flat = a_vla.reshape(1, -1)

        a_new, logits = self._refine(self._ref_state, jnp.asarray(tac_hist), jnp.asarray(a_vla_flat))
        a_new = np.asarray(a_new, dtype=np.float32)
        delta = (a_new - a_vla_flat).reshape(steps, 32)

        # Correction shaping: padding dims always off; optional dim mask;
        # direction hysteresis on the right_tcp.x correction.
        delta[:, VALID_ACTION_DIMS:] = 0.0
        if self._dim_mask is not None:
            keep = np.zeros(VALID_ACTION_DIMS, dtype=bool)
            keep[self._dim_mask] = True
            delta[:, :VALID_ACTION_DIMS] *= keep[None, :]
        suppressed = False
        if self._direction_lock:
            d = delta[:, DIRECTION_DIM]
            sign = int(np.sign(d.sum()))
            if self._locked_sign == 0:
                if sign != 0:
                    self._locked_sign = sign
                    logger.info("refiner direction locked: dim%d sign %+d", DIRECTION_DIM, sign)
            elif sign != 0 and sign != self._locked_sign:
                delta[:, DIRECTION_DIM] = 0.0
                suppressed = True
                self._n_direction_suppressed += 1

        a_new = a_vla + delta
        inv = self._from_refiner_space(
            {"state": state.copy(), "actions": a_new[:, :VALID_ACTION_DIMS].copy()}
        )
        refined_abs = np.asarray(inv["actions"], dtype=np.float32)
        if not np.isfinite(refined_abs).all():
            logger.warning("refiner produced non-finite actions; passing inner output through")
            return outputs

        prob_water = float(1.0 / (1.0 + np.exp(-(logits[0, 1] - logits[0, 0]))))
        self._n_refined += 1
        outputs["actions_vla"] = actions_abs
        outputs["actions"] = refined_abs
        outputs["refiner"] = {
            "active": True,
            "gripper": gripper,
            "delta_l2": float(np.linalg.norm(delta[:, :VALID_ACTION_DIMS])),
            "aux_water_prob": prob_water,
            "direction_sign": self._locked_sign,
            "direction_suppressed": suppressed,
        }
        return outputs

    @override
    def reset(self) -> None:
        self._clear_episode_state()
        self._inner.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._inner, "metadata", {})

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _clear_episode_state(self) -> None:
        self._tac_frames.clear()
        self._locked_sign = 0
        self._n_refined = 0

    def _extract_tactile(self, obs: dict) -> np.ndarray | None:
        """Client obs -> [views, 224, 224, 3] float32 in [-1, 1], or None if unavailable."""
        images = obs.get("images")
        if not isinstance(images, dict) or any(k not in images for k in TACTILE_VIEW_KEYS):
            if not self._warned_missing_tactile:
                logger.warning(
                    "tactile views %s not all present in obs images (%s); refiner passes through",
                    TACTILE_VIEW_KEYS,
                    None if not isinstance(images, dict) else sorted(images),
                )
                self._warned_missing_tactile = True
            return None
        views = []
        for key in TACTILE_VIEW_KEYS:
            img = np.asarray(images[key])
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] != 3:
                img = np.moveaxis(img, 0, -1)  # CHW -> HWC
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
                img = np.asarray(image_tools.resize_with_pad(img[None], IMAGE_SIZE, IMAGE_SIZE)[0])
            views.append(img.astype(np.float32) / 255.0 * 2.0 - 1.0)
        return np.stack(views)

    def _warmup(self) -> None:
        dummy_img = jnp.zeros((_refiner.HISTORY * len(TACTILE_VIEW_KEYS), IMAGE_SIZE, IMAGE_SIZE, 3), jnp.float32)
        feats = np.asarray(self._encode(self._enc_state, dummy_img), dtype=np.float32)
        tac = (feats.reshape(1, _refiner.HISTORY, len(TACTILE_VIEW_KEYS), -1) - self._tac_mean) / self._tac_std
        a_new, _ = self._refine(self._ref_state, jnp.asarray(tac), jnp.zeros((1, _refiner.ACTION_DIMS), jnp.float32))
        jax.block_until_ready(a_new)
        logger.info("tactile refiner warmup done (FastViT + refiner JIT compiled)")
