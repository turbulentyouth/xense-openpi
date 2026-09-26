"""Stage A: slim expert-only tactile training (zero changes under src/openpi/).

Trains ONLY the action expert + tactile branch of the pi05 FastViT tactile model,
with the VLM prefix (2B backbone / SigLIP / prompt) dropped entirely:

  - ``SlimGemmaModule``: ``gemma.Module`` with ``configs=[gemma_300m]`` and no
    token embedder. Expert param names come out WITHOUT the ``_1`` suffix, which
    is what makes the 60k checkpoint mapping (and merge-back) a pure rename.
  - ``SlimTactileExpert``: the suffix half of ``Pi0.embed_suffix`` (pi05 path:
    action_in_proj + time MLP -> adaRMS cond) plus the tactile half of
    ``Pi0TactileFastVit.embed_suffix`` (4 tactile images -> frozen FastViT ->
    tactile_proj -> 4 prepended tokens, adaRMS cond zero-padded), plus a
    2-class aux head on the mean tactile-proj output (water / no_water).
  - Loss: training-time RTC action loss (pi0.py:311-360 suffix half) +
    0.1 * class-weighted CE, CE masked out where the per-frame aux label is -1
    (outside grasp segments / ambiguous release position).
  - Weights: expert + action projections from checkpoints/ckpt59999_bf16/params_npy
    (bf16 bit-pattern npy, decoded leaf by leaf); tactile_encoder from
    checkpoints/params.safetensors (ImageNet init, FROZEN -- the 60k-trained
    encoder collapsed, see plan); tactile_proj from the 60k npy; aux head random.
  - Data: script-local subclass of LeRobotBiFlexivTactileDataConfig whose repack
    keeps episode_index/frame_index and injects the per-frame label from
    outputs/stage_a/aux_labels.npz (compute_aux_labels.py).
  - Optimizer: optax.clip_by_global_norm(1.0) -> multi_transform(
    tactile (tactile_proj + aux head) = 10x base lr, default = 0.1x base lr).

Modes:
  --mode fidelity   compare slim suffix construction vs full-model embed_suffix
  --mode scan       memory scan, batch 2/4/8 x N steps, report peak GPU memory
  --mode train      train (use --steps 100 for the smoke short-run)

Env discipline: run with ~/miniforge3/envs/lerobot-xense/bin/python and
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85. NEVER set XLA_FLAGS=--xla_gpu_autotune_level=0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import pickle
import re
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
from openpi.models.tactile_encoders import build_tactile_encoder
import openpi.policies.bi_flexiv_policy as bi_flexiv_policy
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger("train_stage_a")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_NAME = "pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100"
CKPT_DIR = REPO_ROOT / "checkpoints" / "ckpt59999_bf16"
NPY_DIR = CKPT_DIR / "params_npy"
FASTVIT_SAFETENSORS = REPO_ROOT / "checkpoints" / "params.safetensors"
AUX_LABELS = REPO_ROOT / "outputs" / "stage_a" / "aux_labels.npz"
OUT_DIR = REPO_ROOT / "outputs" / "stage_a"

# Deployment prefix: 3 camera views x 256 SigLIP tokens + 200 prompt tokens.
# Verified in the fidelity smoke: this dataset always has all 3 views + prompt.
PREFIX_LEN = 968
NUM_TACTILE = 4

TACTILE_GROUP_RE = re.compile(r"(^|/)tactile_(proj|aux_head)(/|$)")
FREEZE_RE = nnx_utils.PathRegex(r"(^|/)tactile_encoder(/|$).*")

_EXPERT_PARAM_MODULES = ("q_einsum", "kv_einsum", "attn_vec_einsum", "mlp", "pre_attention_norm", "pre_ffw_norm")
_PROJ_MODULES = ("action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj", "tactile_proj")


# --------------------------------------------------------------------------- #
# Slim gemma (expert only, no embedder)                                        #
# --------------------------------------------------------------------------- #


class SlimGemmaModule(_gemma.Module):
    """``gemma.Module`` without the token embedder, for suffix-only training.

    ``setup`` is a copy of gemma.py:593-630 minus the ``Embedder`` (which serves
    the backbone's prompt tokens and is never called on the suffix path).
    ``init`` likewise skips the ``embed`` warm-up call. Everything else --
    scanned blocks, final norms, ``__call__`` -- is inherited unchanged, so with
    ``configs=[gemma_300m]`` the expert parameters are named exactly like the
    full tree's ``*_1`` leaves minus the suffix.
    """

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        block_cls = nn.remat(
            _gemma.Block,
            prevent_cse=False,
            static_argnums=(5, 7),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            use_cudnn_attention=self.use_cudnn_attention,
            cudnn_attention_dtype=self.cudnn_attention_dtype,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [_gemma.RMSNorm(name=_gemma._name("final_norm", i)) for i in range(len(self.configs))]

    def init(self, use_adarms):
        # Same as gemma.Module.init but without the embed() call (no embedder).
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


# --------------------------------------------------------------------------- #
# Slim tactile expert model                                                    #
# --------------------------------------------------------------------------- #


class SlimTactileExpert(nnx.Module):
    """Action expert + tactile branch of Pi0TactileFastVit, nothing else."""

    def __init__(self, config: Pi0TactileFastVitConfig, rngs: nnx.Rngs) -> None:
        if not config.pi05:
            raise ValueError("SlimTactileExpert only implements the pi05 suffix path")
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self._max_delay = config.max_delay

        expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            SlimGemmaModule(
                configs=[expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                use_cudnn_attention=config.use_cudnn_attention,
                cudnn_attention_dtype=config.cudnn_attention_dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[True])
        self.PaliGemma = nnx.Dict(llm=llm)

        width = expert_config.width
        self.action_in_proj = nnx.Linear(config.action_dim, width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(width, width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(width, width, rngs=rngs)
        self.action_out_proj = nnx.Linear(width, config.action_dim, rngs=rngs)

        compute_dtype = jnp.dtype(config.tactile_compute_dtype)
        self.tactile_encoder = build_tactile_encoder(
            config.tactile_encoder_name,
            rngs=rngs,
            pretrained_path=config.tactile_pretrained_path,
            compute_dtype=compute_dtype,
        )
        self.tactile_proj = nnx.Linear(self.tactile_encoder.feature_dim, width, rngs=rngs)
        self.tactile_aux_head = nnx.Linear(width, 2, rngs=rngs)

        self._tactile_keys: tuple[str, ...] = tuple(config.tactile_image_keys)
        self._num_tactile = len(self._tactile_keys)

    # ------------------------------------------------------------------ #
    # Suffix construction: pi0.py:210-248 (pi05) + pi0_tactile_fastvit.py:100-139
    # ------------------------------------------------------------------ #

    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep,
    ):
        """Mirror Pi0TactileFastVit.embed_suffix. Returns (tokens, mask, ar, adarms_cond)."""
        tokens, mask, ar, cond, _ = self._embed_suffix_impl(obs, noisy_actions, timestep)
        return tokens, mask, ar, cond

    def _embed_suffix_impl(self, obs, noisy_actions, timestep):
        # ---- tactile branch (pi0_tactile_fastvit.py:100-115) ----
        with jax.named_scope("suffix/tactile/stack"):
            tactile_imgs = jnp.stack([obs.images[key] for key in self._tactile_keys], axis=1)
            tactile_mask = jnp.stack([obs.image_masks[key] for key in self._tactile_keys], axis=1)
            b, n, h, w, c = tactile_imgs.shape
        with jax.named_scope("suffix/tactile/fastvit"):
            feats = self.tactile_encoder(tactile_imgs.reshape(b * n, h, w, c))
        with jax.named_scope("suffix/tactile/proj"):
            tactile_feats = self.tactile_proj(feats)
            tactile_tokens = tactile_feats.reshape(b, n, -1)
        tactile_ar = jnp.asarray([True] + [False] * (self._num_tactile - 1))

        # ---- base pi05 suffix (pi0.py:210-248) ----
        with jax.named_scope("suffix/action_in_proj"):
            action_tokens = self.action_in_proj(noisy_actions)
        with jax.named_scope("suffix/time_embed"):
            time_emb = _pi0.posemb_sincos(
                timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
        base_tokens = action_tokens
        base_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        base_ar = jnp.asarray([True] + [False] * (self.action_horizon - 1))

        with jax.named_scope("suffix/concat"):
            tokens = jnp.concatenate([tactile_tokens, base_tokens], axis=1)
            input_mask = jnp.concatenate([tactile_mask, base_mask], axis=1)
            ar_mask = jnp.concatenate([tactile_ar, base_ar], axis=0)
            if adarms_cond is not None and adarms_cond.ndim == 3:
                tactile_cond = jnp.zeros(
                    (adarms_cond.shape[0], self._num_tactile, adarms_cond.shape[-1]),
                    dtype=adarms_cond.dtype,
                )
                adarms_cond = jnp.concatenate([tactile_cond, adarms_cond], axis=1)
        return tokens, input_mask, ar_mask, adarms_cond, tactile_feats

    # ------------------------------------------------------------------ #
    # RTC suffix loss + aux CE (pi0.py:311-360 suffix half)                #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        rng,
        obs: _model.Observation,
        actions,
        tac_labels,
        class_weights,
        *,
        train: bool = True,
        aux_weight: float = 0.1,
    ):
        prep_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        with jax.named_scope("loss/preprocess"):
            obs = _model.preprocess_observation_tactile(
                prep_rng, obs, train=train, image_keys=_model.IMAGE_KEYS_TACTILE_4
            )

        with jax.named_scope("loss/rng_noise_time_delay"):
            b, ah, ad = actions.shape
            time = jax.random.uniform(time_rng, (b,))
            noise = jax.random.normal(noise_rng, (b, ah, ad))
            delay = jax.random.randint(delay_rng, (b,), 0, self._max_delay)
            action_prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]
            time_masked = jnp.where(action_prefix_mask, 0.0, time[:, None])
            x_t = time_masked[:, :, None] * noise + (1 - time_masked[:, :, None]) * actions
            u_t = noise - actions

        with jax.named_scope("loss/embed_suffix"):
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond, tactile_feats = self._embed_suffix_impl(
                obs, x_t, time_masked
            )
        with jax.named_scope("loss/attn_mask"):
            attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar)
            positions = PREFIX_LEN + jnp.cumsum(suffix_mask, axis=-1) - 1
        with jax.named_scope("loss/llm_forward"):
            outs, _ = self.PaliGemma.llm(
                [suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[adarms_cond],
            )
            suffix_out = outs[0]
        with jax.named_scope("loss/action_out"):
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            loss = (v_t - u_t) ** 2
            action_postfix_mask = jnp.logical_not(action_prefix_mask)[:, :, None]
            per_token = jnp.sum(loss * action_postfix_mask, axis=-1) / (
                jnp.sum(action_postfix_mask, axis=-1) + 1e-8
            )
            action_loss = jnp.mean(per_token)

        with jax.named_scope("loss/aux_ce"):
            logits = self.tactile_aux_head(jnp.mean(tactile_feats.reshape(b, self._num_tactile, -1), axis=1))
            valid = tac_labels >= 0
            safe = jnp.maximum(tac_labels, 0).astype(jnp.int32)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), safe)
            ce = ce * class_weights[safe]
            n_valid = jnp.maximum(jnp.sum(valid), 1)
            aux_loss = jnp.sum(jnp.where(valid, ce, 0.0)) / n_valid
            pred = jnp.argmax(logits, axis=-1)
            hit = (pred == safe) & valid
            aux_acc = jnp.sum(hit) / n_valid
            acc0 = jnp.sum(hit & (safe == 0)) / jnp.maximum(jnp.sum(valid & (safe == 0)), 1)
            acc1 = jnp.sum(hit & (safe == 1)) / jnp.maximum(jnp.sum(valid & (safe == 1)), 1)

        total = action_loss + aux_weight * aux_loss
        metrics = {
            "action_loss": action_loss,
            "aux_loss": aux_loss,
            "aux_acc": aux_acc,
            "aux_acc_no_water": acc0,
            "aux_acc_water": acc1,
            "frac_labeled": jnp.mean(valid.astype(jnp.float32)),
        }
        return total, metrics


# --------------------------------------------------------------------------- #
# Weight loading (leaf-by-leaf from the bf16 npy dir)                           #
# --------------------------------------------------------------------------- #


def decode_bf16_npy(path: pathlib.Path) -> np.ndarray:
    """uint16 bf16 bit pattern -> float32."""
    raw = np.load(path)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def encode_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> uint16 bf16 bit pattern (RNE via jnp, same as convert_checkpoint_bf16.py)."""
    return np.asarray(jnp.asarray(x, dtype=jnp.bfloat16).view(jnp.uint16))


def _flatten(tree: dict, prefix: tuple = ()) -> dict[tuple, Any]:
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flatten(v, (*prefix, k)))
        else:
            out[(*prefix, k)] = v
    return out


def _unflatten(flat: dict[tuple, Any]) -> dict:
    out: dict = {}
    for path, v in flat.items():
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = v
    return out


def npy_leaf_for_slim_path(dotted: str, *, encoder_source: str) -> str | None:
    """Map a slim state path to its leaf file in the 60k params_npy dir (None = keep init)."""
    if dotted.startswith("PaliGemma.llm.layers."):
        rest = dotted[len("PaliGemma.llm.layers.") :]
        parts = rest.split(".")
        if parts[0] == "attn":
            mapped = f"attn.{parts[1]}_1." + ".".join(parts[2:])
        elif parts[0] in _EXPERT_PARAM_MODULES:
            mapped = f"{parts[0]}_1." + ".".join(parts[1:])
        else:
            raise ValueError(f"unmapped slim llm.layers path: {dotted}")
        return f"params.PaliGemma.llm.layers.{mapped}.value.bf16.npy"
    if dotted.startswith("PaliGemma.llm.final_norm."):
        rest = dotted[len("PaliGemma.llm.final_norm.") :]
        return f"params.PaliGemma.llm.final_norm_1.{rest}.value.bf16.npy"
    top = dotted.split(".")[0]
    if top in _PROJ_MODULES:
        return f"params.{dotted}.value.bf16.npy"
    if top == "tactile_encoder":
        if encoder_source == "npy":
            return f"params.{dotted}.value.bf16.npy"
        return None  # keep the safetensors ImageNet init
    if top == "tactile_aux_head":
        return None  # random init
    raise ValueError(f"unmapped slim path: {dotted}")


def load_slim_weights(
    model: SlimTactileExpert,
    npy_dir: pathlib.Path,
    *,
    encoder_source: str = "safetensors",
) -> SlimTactileExpert:
    """Load 60k weights into the slim tree, leaf by leaf, with shape checks."""
    graphdef, state = nnx.split(model)
    flat = _flatten(state.to_pure_dict())
    loaded, kept = [], []
    new_flat = {}
    for path, cur in flat.items():
        dotted = ".".join(str(p) for p in path)
        leaf = npy_leaf_for_slim_path(dotted, encoder_source=encoder_source)
        if leaf is None:
            kept.append(dotted)
            new_flat[path] = cur
            continue
        src = npy_dir / leaf
        if not src.exists():
            raise FileNotFoundError(f"{dotted} -> missing {src}")
        arr = decode_bf16_npy(src)
        cur_shape = tuple(cur.shape) if hasattr(cur, "shape") else None
        if cur_shape != arr.shape:
            raise ValueError(f"shape mismatch at {dotted}: slim {cur_shape} vs npy {arr.shape} ({src.name})")
        new_flat[path] = arr
        loaded.append(dotted)
    state.replace_by_pure_dict(_unflatten(new_flat))
    logger.info("loaded %d leaves from %s; kept init for %d leaves", len(loaded), npy_dir, len(kept))
    return nnx.merge(graphdef, state)


def save_slim_params(model_or_state, path: pathlib.Path) -> None:
    state = model_or_state if isinstance(model_or_state, nnx.State) else nnx.state(model_or_state)
    pure = jax.tree.map(lambda x: np.asarray(x), state.to_pure_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pure, f)
    logger.info("saved slim params -> %s", path)


def load_slim_params(model: SlimTactileExpert, path: pathlib.Path) -> SlimTactileExpert:
    with open(path, "rb") as f:
        pure = pickle.load(f)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pure)
    return nnx.merge(graphdef, state)


# --------------------------------------------------------------------------- #
# Data (script-local config subclass + label transform)                         #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class InjectTactileLabel:
    """Attach the per-frame water/no_water label from aux_labels.npz.

    Sits in repack_transforms (training only, like InjectTactileReference). The
    npz is loaded lazily inside each dataloader worker and cached there.
    """

    labels_path: str
    _cache: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    def _labels(self):
        if "labels" not in self._cache:
            with np.load(self.labels_path) as z:
                self._cache["labels"] = {k: z[k] for k in z.files}
        return self._cache["labels"]

    def __call__(self, data: dict) -> dict:
        ep = int(np.asarray(data["episode_index"]))
        frame = int(np.asarray(data["frame_index"]))
        data["tactile_label"] = np.asarray(self._labels()[f"ep{ep}"][frame], dtype=np.int64)
        return data


@dataclasses.dataclass(frozen=True)
class StageABiFlexivTactileInputs(bi_flexiv_policy.BiFlexivTactileInputs):
    """BiFlexivTactileInputs that keeps episode_index/frame_index.

    The stock transform rebuilds the sample dict and drops every unknown key,
    but the aux-label lookup needs the (episode, frame) key afterwards.
    """

    def __call__(self, data: dict) -> dict:
        ep, frame = data["episode_index"], data["frame_index"]
        out = super().__call__(data)
        out["episode_index"] = ep
        out["frame_index"] = frame
        return out


@dataclasses.dataclass(frozen=True)
class StageADataConfig(_config.LeRobotBiFlexivTactileDataConfig):
    """LeRobotBiFlexivTactileDataConfig + per-frame aux label injection."""

    aux_labels_path: str = str(AUX_LABELS)

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        **self.repack_transforms.inputs[0].structure,
                        # episode_index/frame_index must survive the repack so the label
                        # lookup has a key (same pattern as the Diff config, config.py:604-613).
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                    }
                ),
            ]
        )
        # The label is injected at the END of the data transforms: BiFlexivTactileInputs
        # and DeltaActions both consume the repacked dict, and Normalize / the model
        # transforms pass unknown keys through, so tactile_label reaches the batch.
        data_transforms = _transforms.Group(
            inputs=[StageABiFlexivTactileInputs()],
            outputs=[bi_flexiv_policy.BiFlexivOutputs()],
        )
        if self.use_delta_cartesian_actions:
            delta_action_mask = _transforms.make_bool_mask(18, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        data_transforms = data_transforms.push(inputs=[InjectTactileLabel(self.aux_labels_path)])
        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def build_train_config(args) -> _config.TrainConfig:
    loaded = _config.get_config(CONFIG_NAME)
    src_data = loaded.data
    base_config = dataclasses.replace(src_data.base_config or _config.DataConfig(), tactile=True)
    stage_data = StageADataConfig(
        repo_id=src_data.repo_id,
        base_config=base_config,
        assets=_config.AssetsConfig(assets_dir=str(CKPT_DIR / "assets")),
        default_prompt=src_data.default_prompt,
        aux_labels_path=str(AUX_LABELS),
    )
    return dataclasses.replace(
        loaded,
        data=stage_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def compute_class_weights(labels_path: pathlib.Path) -> np.ndarray:
    with np.load(labels_path) as z:
        all_labels = np.concatenate([z[k] for k in z.files])
    n0 = int((all_labels == 0).sum())
    n1 = int((all_labels == 1).sum())
    total = n0 + n1
    # inverse-frequency balancing: w_c = total / (2 * n_c)
    return np.asarray([total / (2 * n0), total / (2 * n1)], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Optimizer                                                                     #
# --------------------------------------------------------------------------- #


def build_tx(trainable_params: nnx.State, *, steps: int, warmup: int) -> optax.GradientTransformation:
    """clip(1.0) -> multi_transform(tactile=10x, default=0.1x of the 60k base lr)."""
    warmup = min(warmup, max(steps - 1, 1))  # optax requires decay_steps > warmup_steps

    def cosine(peak: float) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=peak / (warmup + 1),
            peak_value=peak,
            warmup_steps=warmup,
            decay_steps=steps,
            end_value=peak / 10,
        )

    def adamw(lr) -> optax.GradientTransformation:
        return optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)

    base_peak = 2.5e-5  # production 60k peak lr
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"tactile": adamw(cosine(base_peak * 10)), "default": adamw(cosine(base_peak * 0.1))},
            _group_labels(trainable_params),
        ),
    )
    return tx


def _keypath_str(kp) -> str:
    parts = []
    for e in kp:
        parts.append(str(getattr(e, "key", getattr(e, "idx", e))))
    return "/".join(parts)


def _group_labels(params: nnx.State):
    return jax.tree_util.tree_map_with_path(
        lambda kp, _: "tactile" if TACTILE_GROUP_RE.search(_keypath_str(kp)) else "default",
        params,
    )


# --------------------------------------------------------------------------- #
# Train step (jit, graphdef closed over; state in/out like layer_probe)         #
# --------------------------------------------------------------------------- #


def make_step_fn(graphdef, tx, trainable_filter, class_weights):
    def loss_fn(model, rng, obs, actions, tac_labels):
        return model.compute_loss(rng, obs, actions, tac_labels, class_weights, train=True)

    def step(state, opt_state, rng, obs, actions, tac_labels):
        model = nnx.merge(graphdef, state)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter), has_aux=True)(
            model, rng, obs, actions, tac_labels
        )
        params = state.filter(trainable_filter)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_state = nnx.state(model)
        tactile_grads = grads.filter(nnx_utils.PathRegex(r".*tactile_(proj|aux_head).*"))
        metrics = {
            **metrics,
            "grad_norm": optax.global_norm(grads),
            "grad_norm_tactile": optax.global_norm(tactile_grads),
        }
        return new_state, new_opt_state, loss, metrics

    return jax.jit(step)


def batch_to_model_inputs(batch):
    obs = _model.Observation.from_dict(batch)
    return obs, batch["actions"], batch["tactile_label"]


# --------------------------------------------------------------------------- #
# Modes                                                                         #
# --------------------------------------------------------------------------- #


def build_slim_model(model_config: Pi0TactileFastVitConfig, *, encoder_source: str, seed: int = 0):
    """Build the slim model and load 60k weights.

    encoder_source="safetensors": tactile_encoder keeps the ImageNet init loaded
    from config.tactile_pretrained_path (the training configuration; frozen).
    encoder_source="npy": load the 60k encoder leaves too (fidelity check only --
    the full model carries the collapsed 60k encoder, so the suffix comparison
    needs matching encoder weights).
    """
    if encoder_source == "safetensors":
        if not pathlib.Path(str(model_config.tactile_pretrained_path)).expanduser().exists():
            raise FileNotFoundError(
                f"tactile_pretrained_path {model_config.tactile_pretrained_path} missing; "
                f"expected {FASTVIT_SAFETENSORS}"
            )
    model = SlimTactileExpert(model_config, rngs=nnx.Rngs(seed))
    return load_slim_weights(model, NPY_DIR, encoder_source=encoder_source)


def resolve_model_config() -> Pi0TactileFastVitConfig:
    train_config = _config.get_config(CONFIG_NAME)
    model_config = train_config.model
    if not isinstance(model_config, Pi0TactileFastVitConfig):
        raise TypeError(type(model_config).__name__)
    return dataclasses.replace(model_config, tactile_pretrained_path=str(FASTVIT_SAFETENSORS))


def mode_fidelity(args) -> None:
    """Compare slim suffix construction against the full model's embed_suffix."""
    from test.tactile_counterfactual import layer_probe

    setup = layer_probe.load_setup(CONFIG_NAME, CKPT_DIR)
    full = setup.model
    model_config = resolve_model_config()
    slim = build_slim_model(model_config, encoder_source="npy")  # match the full model's 60k encoder
    # Cast every slim float leaf to bf16 so both sides run identical dtypes. This
    # must include BatchStat leaves: with bf16 compute, fp32 BN running stats
    # promote the BN to fp32 while the full model computes it fully in bf16, which
    # alone shifts the tactile tokens by ~1e-2 (measured). nnx.Param-only casting
    # is NOT enough.
    graphdef, state = nnx.split(slim)

    def _bf16_leaf(_, v):
        val = v.value if hasattr(v, "value") else v
        if hasattr(val, "dtype") and jnp.issubdtype(val.dtype, jnp.floating):
            return v.replace(val.astype(jnp.bfloat16))
        return v

    state = state.map(_bf16_leaf)
    slim = nnx.merge(graphdef, state)

    sample = setup.dataset.get_sample(args.fidelity_episode, args.fidelity_frame)
    obs = setup.dataset.observation_from_sample(sample)
    rng = jax.random.key(123)
    b = obs.state.shape[0]
    x_t = jax.random.normal(rng, (b, full.action_horizon, full.action_dim))

    results = {}
    for name, timestep in (
        ("time_1d", jnp.full((b,), 0.5)),
        ("time_2d", jnp.linspace(0.0, 1.0, full.action_horizon)[None, :].repeat(b, axis=0)),
    ):
        obs_full = full._preprocess_observation(None, obs, train=False)
        f_tokens, f_mask, f_ar, f_cond = full.embed_suffix(obs_full, x_t, timestep)
        obs_slim = _model.preprocess_observation_tactile(
            None, obs, train=False, image_keys=_model.IMAGE_KEYS_TACTILE_4
        )
        s_tokens, s_mask, s_ar, s_cond = slim.embed_suffix(obs_slim, x_t, timestep)

        tok_diff = jnp.abs(f_tokens.astype(jnp.float32) - s_tokens.astype(jnp.float32))
        denom = jnp.maximum(jnp.abs(f_tokens.astype(jnp.float32)), 1e-6)
        entry = {
            "tokens_shape": list(f_tokens.shape),
            "tokens_max_abs_diff": float(jnp.max(tok_diff)),
            "tokens_max_rel_diff": float(jnp.max(tok_diff / denom)),
            "mask_equal": bool(jnp.all(f_mask == s_mask)),
            "ar_equal": bool(jnp.all(f_ar == s_ar)),
        }
        if f_cond is not None:
            cond_diff = jnp.abs(f_cond.astype(jnp.float32) - s_cond.astype(jnp.float32))
            entry["cond_shape"] = list(f_cond.shape)
            entry["cond_max_abs_diff"] = float(jnp.max(cond_diff))
            entry["cond_max_rel_diff"] = float(jnp.max(cond_diff / jnp.maximum(jnp.abs(f_cond.astype(jnp.float32)), 1e-6)))
        results[name] = entry
        logger.info("fidelity[%s]: %s", name, entry)

    ok = all(
        e["mask_equal"] and e["ar_equal"] and e["tokens_max_rel_diff"] < 1e-5 and e.get("cond_max_rel_diff", 0.0) < 1e-5
        for e in results.values()
    )
    logger.info("fidelity check %s", "PASSED" if ok else "FAILED (see diffs above)")
    (OUT_DIR / "fidelity_report.json").write_text(json.dumps(results, indent=2))
    if not ok:
        raise SystemExit(1)


def mode_scan(args) -> None:
    """Memory scan: batch 2/4/8, N steps each, report peak GPU memory."""
    model_config = resolve_model_config()
    slim = build_slim_model(model_config, encoder_source="safetensors")
    graphdef, state = nnx.split(slim)
    state = nnx_utils.state_map(state, FREEZE_RE, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    trainable_filter = nnx.All(nnx.Param, nnx.Not(FREEZE_RE))
    class_weights = jnp.asarray(compute_class_weights(AUX_LABELS))
    rng = jax.random.key(args.seed)

    rows = []
    for bs in (2, 4, 8):
        args.batch_size = bs
        train_cfg = build_train_config(args)
        loader = _data_loader.create_data_loader(train_cfg, shuffle=True)
        tx = build_tx(state.filter(trainable_filter), steps=args.steps, warmup=100)
        opt_state = tx.init(state.filter(trainable_filter))
        step_fn = make_step_fn(graphdef, tx, trainable_filter, class_weights)
        it = iter(loader._data_loader)
        t0 = time.monotonic()
        losses = []
        for i in range(args.scan_steps):
            batch = next(it)
            obs, actions, tac_labels = batch_to_model_inputs(batch)
            rng, step_rng = jax.random.split(rng)
            state, opt_state, loss, metrics = step_fn(state, opt_state, step_rng, obs, actions, tac_labels)
            losses.append(float(loss))
            if i == 0:
                logger.info("bs=%d first step (incl. compile) %.1fs", bs, time.monotonic() - t0)
        stats = jax.devices()[0].memory_stats()
        peak = stats.get("peak_bytes_in_use", 0) / 2**30 if stats else float("nan")
        dt = time.monotonic() - t0
        rows.append({"batch_size": bs, "steps": args.scan_steps, "peak_gib": peak,
                     "steps_per_s": args.scan_steps / dt, "last_loss": losses[-1]})
        logger.info("bs=%d done: %s", bs, rows[-1])
        loader.close()
        del loader, it, step_fn, opt_state, tx
        jax.clear_caches()
    logger.info("memory scan summary: %s", json.dumps(rows, indent=2))
    (OUT_DIR / "memory_scan.json").write_text(json.dumps(rows, indent=2))


def mode_train(args) -> None:
    model_config = resolve_model_config()
    slim = build_slim_model(model_config, encoder_source="safetensors")
    graphdef, state = nnx.split(slim)
    state = nnx_utils.state_map(state, FREEZE_RE, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    trainable_filter = nnx.All(nnx.Param, nnx.Not(FREEZE_RE))

    class_weights = jnp.asarray(compute_class_weights(AUX_LABELS))
    logger.info("CE class weights [no_water, water]: %s", np.asarray(class_weights))

    train_cfg = build_train_config(args)
    loader = _data_loader.create_data_loader(train_cfg, shuffle=True)
    tx = build_tx(state.filter(trainable_filter), steps=args.steps, warmup=args.warmup)
    opt_state = tx.init(state.filter(trainable_filter))
    step_fn = make_step_fn(graphdef, tx, trainable_filter, class_weights)

    # Sanity: all image masks true (the PREFIX_LEN=968 assumption).
    it = iter(loader._data_loader)
    first = next(it)
    mask_check = {k: bool(np.all(np.asarray(v))) for k, v in first["image_mask"].items()}
    if not all(mask_check.values()):
        raise RuntimeError(f"image_mask not all true; fixed PREFIX_LEN={PREFIX_LEN} is invalid: {mask_check}")
    batches = [first]

    rng = jax.random.key(args.seed)
    log_every = 10 if args.steps <= 200 else 100
    history = []
    t0 = time.monotonic()
    for step_i in range(args.steps):
        if not batches:
            batches.append(next(it))
        batch = batches.pop()
        obs, actions, tac_labels = batch_to_model_inputs(batch)
        rng, step_rng = jax.random.split(rng)
        state, opt_state, loss, metrics = step_fn(state, opt_state, step_rng, obs, actions, tac_labels)
        if step_i % log_every == 0 or step_i == args.steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            m["loss"] = float(loss)
            m["step"] = step_i
            history.append(m)
            logger.info("step %d: %s", step_i, m)
        if not np.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss at step {step_i}")
    dt = time.monotonic() - t0
    logger.info("trained %d steps in %.1fs (%.2f steps/s)", args.steps, dt, args.steps / dt)

    # Shut the dataloader workers down BEFORE serializing: 4 spawned workers each
    # holding the dataset + prefetch buffer plus the host copy of the params
    # overruns the ~14 GB host RAM budget (observed OOM-kill during save).
    loader.close()
    save_path = pathlib.Path(args.save_slim)
    save_slim_params(state, save_path)
    (OUT_DIR / f"train_history_{args.steps}.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fidelity", "scan", "train"], required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scan-steps", type=int, default=20)
    p.add_argument("--save-slim", default=str(OUT_DIR / "slim_params_final.pkl"))
    p.add_argument("--fidelity-episode", type=int, default=0)
    p.add_argument("--fidelity-frame", type=int, default=100)
    p.add_argument("--print-paths", action="store_true", help="debug: print slim state paths and exit")
    args = p.parse_args()

    if args.print_paths:
        model_config = resolve_model_config()
        model = SlimTactileExpert(model_config, rngs=nnx.Rngs(0))
        flat = _flatten(nnx.state(model).to_pure_dict())
        for path, v in sorted(flat.items()):
            if not path[0] == "tactile_encoder":
                print(".".join(str(x) for x in path), getattr(v, "shape", None), getattr(v, "dtype", None))
        n_enc = sum(1 for k in flat if k[0] == "tactile_encoder")
        print(f"... plus {n_enc} tactile_encoder leaves")
        return

    {"fidelity": mode_fidelity, "scan": mode_scan, "train": mode_train}[args.mode](args)


if __name__ == "__main__":
    main()
