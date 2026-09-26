"""Slim loader utilities for the attention-dump tooling.

Extracted from the probe runner of commit dc83417 (the full runner was removed in
7fd05d1 together with the probe orchestration). Only the pieces the attention dump
needs are kept: data-config resolution with norm stats, and checkpoint loading with
a tactile-weights presence check.

One deliberate change from dc83417: ``load_model`` restores params as bfloat16 (the
same dtype the production inference path uses, see
``openpi.policies.policy_config.create_trained_policy``). The old code restored
float32, which transiently needs ~13 GB of host RAM for this checkpoint and gets
OOM-killed on 16 GB-RAM machines.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
import openpi.shared.normalize as _normalize
import openpi.training.config as _config

logger = logging.getLogger("tactile_counterfactual")


def resolve_data_config(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
) -> _config.DataConfig:
    """DataConfig with norm stats guaranteed present.

    Priority: config assets dir (compute_norm_stats.py output), then the
    checkpoint's own assets (saved by train.py at save time). Missing norm
    stats raise a clear error — silently skipping normalization would produce
    observations inconsistent with training.
    """
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if data_config.norm_stats is not None:
        return data_config

    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError(
            "Norm stats are required but the data config has no asset_id; "
            "cannot locate normalization stats."
        )

    # 1) checkpoint assets: <checkpoint_dir>/assets/<asset_id>/
    ckpt_assets = checkpoint_dir / "assets" / asset_id
    if ckpt_assets.is_dir():
        norm_stats = _normalize.load(ckpt_assets)
        logger.info("Loaded norm stats from checkpoint assets: %s", ckpt_assets)
        return dataclasses.replace(data_config, norm_stats=norm_stats)

    # 2) config assets dir: <assets>/<config_name>/<asset_id>/
    config_assets = train_config.assets_dirs / asset_id
    if config_assets.is_dir():
        norm_stats = _normalize.load(config_assets)
        logger.info("Loaded norm stats from config assets: %s", config_assets)
        return dataclasses.replace(data_config, norm_stats=norm_stats)

    raise FileNotFoundError(
        f"Norm stats not found for asset_id={asset_id!r}. Looked in:\n"
        f"  {ckpt_assets}\n  {config_assets}\n"
        "Run scripts/compute_norm_stats.py for this config, or point the dump "
        "at a checkpoint that contains its assets."
    )


def load_model(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[_model.BaseModel, dict[str, Any]]:
    """Load the checkpoint into a model; verify tactile weights are present."""
    model_config = train_config.model
    if not isinstance(model_config, Pi0TactileFastVitConfig):
        raise TypeError(
            f"Probe requires a Pi0TactileFastVitConfig model, got {type(model_config).__name__}. "
            "The attention dump is only defined for tactile models."
        )

    # The pretrained FastViT path on the training machine may not exist here;
    # weights come from the checkpoint anyway, so drop the path (the encoder
    # structure is identical).
    if (
        model_config.tactile_pretrained_path is not None
        and not pathlib.Path(model_config.tactile_pretrained_path).expanduser().exists()
    ):
        logger.warning(
            "tactile_pretrained_path %s does not exist locally; ignoring it (weights will come from the checkpoint).",
            model_config.tactile_pretrained_path,
        )
        model_config = dataclasses.replace(model_config, tactile_pretrained_path=None)

    # A training checkpoint step dir holds params/, train_state/ and assets/; the
    # weights themselves live in params/ (same path policy_config.create_trained_policy
    # loads). Accept a params dir directly too, for convenience. A params_npy/
    # directory (scripts/convert_checkpoint_bf16.py) takes precedence: it reads
    # leaf by leaf and halves both host RAM and load time on small machines.
    params_dir = checkpoint_dir / "params"
    npy_dir = checkpoint_dir / "params_npy"
    if not params_dir.is_dir() and not npy_dir.is_dir():
        params_dir = checkpoint_dir
    if npy_dir.is_dir():
        params = load_npy_params(npy_dir)
    else:
        params = _model.restore_params(params_dir, dtype=dtype)
    flat = _flatten_keys(params)
    missing_tactile = [k for k in ("tactile_encoder", "tactile_proj") if not any(k in key for key in flat)]
    if missing_tactile:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} has no tactile weights ({missing_tactile}); "
            "cannot dump tactile attention. Available top-level keys: "
            f"{sorted({k.split('/')[0] for k in flat})}"
        )

    logger.info("Loading model from checkpoint %s", checkpoint_dir)
    model = model_config.load(params)
    logger.info("Model class: %s", type(model).__name__)
    return model, {"checkpoint_tactile_keys": [k for k in flat if "tactile" in k]}


def load_npy_params(npy_dir: pathlib.Path) -> dict[str, Any]:
    """Load a bf16 npy directory written by scripts/convert_checkpoint_bf16.py.

    Files are named ``<dotted.path>.bf16.npy`` and hold the uint16 bit pattern of
    bfloat16 values. Reads leaf by leaf (bounded host RAM), returns the params
    pytree with the training-time ``value`` suffix stripped, like
    ``model.restore_params``.
    """
    import ml_dtypes

    import numpy as np

    def strip_value(node):
        # nnx.State saves every leaf as {...: {"value": array}}; undo that.
        if isinstance(node, dict):
            if set(node.keys()) == {"value"}:
                return node["value"]
            return {k: strip_value(v) for k, v in node.items()}
        return node

    files = sorted(npy_dir.glob("*.bf16.npy"))
    if not files:
        raise FileNotFoundError(f"no *.bf16.npy files in {npy_dir}")
    tree: dict[str, Any] = {}
    total = 0
    for f in files:
        dotted = f.name[: -len(".bf16.npy")]
        parts = dotted.split(".")
        assert parts[0] == "params", f"unexpected leaf prefix in {f.name}"
        x = np.load(f).view(ml_dtypes.bfloat16)
        total += x.nbytes
        node = tree
        for p in parts[1:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(x)
    logger.info("loaded %d bf16 leaves (%.2f GiB) from %s", len(files), total / 2**30, npy_dir)
    return strip_value(tree)


def _flatten_keys(tree: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flatten_keys(v, key + "/"))
        else:
            out.append(key)
    return out
