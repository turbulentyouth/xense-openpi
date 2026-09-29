"""Tactile refiner (v1.1): FiLM residual MLP + tactile history, serving-side home.

The refiner corrects a VLA action chunk in the policy's INTERNAL action space
(normalised delta-cartesian, flattened [action_horizon * action_dim]) from a
short tactile feature history. It was developed and validated under
``test/tactile_counterfactual/train_refiner_v11.py``; the classes here are the
single source of truth for both training and serving (the training script
imports them from this module).

Weights live in a pickle produced by the training script:
``outputs/refiner_v11/refiner_params.pkl`` with keys
    state    -- nnx state as a pure dict (RefinerV11 parameter tree)
    tac_mean -- per-feature-dim train mean of the FastViT features, [TAC_DIM]
    tac_std  -- per-feature-dim train std (+1e-8), [TAC_DIM]
    config   -- architecture dict: history/action_dims/tac_dim/frame_dim/
                gru_dim/h_dim/delta_scale (+ training-only dropout rates)

Inputs at inference:
    tac_hist [B, HISTORY, 4 views, TAC_DIM] -- FastViT-T12 features of the four
        tactile views over the last HISTORY frames (t-HISTORY+1 .. t),
        standardised with tac_mean/tac_std. Views are mean-pooled inside the
        model; the time axis is reduced by a small GRU.
    a_vla    [B, ACTION_DIMS] -- the flattened normalised-delta action chunk.

Output: (a_vla + delta, aux_logits) where delta is bounded to
``DELTA_SCALE * tanh(.)`` and aux_logits is the 2-way water/no-water head.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

ACTION_DIMS = 1600  # 50 steps x 32 padded action dims, flattened
TAC_DIM = 1024  # FastViT-T12 feature dim per view per frame
FRAME_DIM = 256
GRU_DIM = 128
H_DIM = 512
DELTA_SCALE = 2.0
HISTORY = 4


class TinyGRU(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, din: int, dh: int):
        self.x2g = nnx.Linear(din, 3 * dh, rngs=rngs)
        self.h2g = nnx.Linear(dh, 3 * dh, use_bias=False, rngs=rngs)
        self.dh = dh

    def __call__(self, xs: jax.Array) -> jax.Array:
        """xs [B, T, D] -> final hidden state [B, dh]."""
        h = jnp.zeros((xs.shape[0], self.dh), dtype=xs.dtype)
        for t in range(xs.shape[1]):
            xr, xz, xn = jnp.split(self.x2g(xs[:, t]), 3, axis=-1)
            hr, hz, hn = jnp.split(self.h2g(h), 3, axis=-1)
            r = jax.nn.sigmoid(xr + hr)
            z = jax.nn.sigmoid(xz + hz)
            n = jnp.tanh(xn + r * hn)
            h = (1 - z) * n + z * h
        return h


class RefinerV11(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, history: str = "gru"):
        self.history = history
        self.frame_fc = nnx.Linear(TAC_DIM, FRAME_DIM, rngs=rngs)
        z_dim = GRU_DIM if history == "gru" else FRAME_DIM
        if history == "gru":
            self.gru = TinyGRU(rngs, FRAME_DIM, GRU_DIM)
        self.film = nnx.Linear(z_dim, 2 * H_DIM, rngs=rngs)
        self.aux = nnx.Linear(z_dim, 2, rngs=rngs)
        self.a_fc = nnx.Linear(ACTION_DIMS, H_DIM, rngs=rngs)
        self.mlp1 = nnx.Linear(H_DIM, H_DIM, rngs=rngs)
        self.mlp2 = nnx.Linear(H_DIM, ACTION_DIMS, rngs=rngs)

    def __call__(self, tac_hist: jax.Array, a_vla: jax.Array) -> tuple[jax.Array, jax.Array]:
        """tac_hist [B, T, views, 1024] (standardised), a_vla [B, 1600]."""
        x = tac_hist.mean(axis=2)  # pool the 4 views -> [B, T, 1024]
        x = nnx.gelu(self.frame_fc(x))  # shared per-frame projection -> [B, T, 256]
        z = self.gru(x) if self.history == "gru" else x.mean(axis=1)
        gamma, beta = jnp.split(self.film(z), 2, axis=-1)
        logits = self.aux(z)
        h = nnx.gelu(self.a_fc(a_vla))
        h = gamma * h + beta
        out = self.mlp2(nnx.gelu(self.mlp1(h)))
        delta = DELTA_SCALE * jnp.tanh(out / DELTA_SCALE)
        return a_vla + delta, logits


def load_refiner(
    pkl_path: str | pathlib.Path,
    *,
    rngs: nnx.Rngs | None = None,
) -> tuple[RefinerV11, dict[str, Any]]:
    """Load a trained refiner from its pickle.

    Returns (model, meta) where meta carries:
        tac_mean / tac_std -- feature standardisation arrays [TAC_DIM]
        config             -- the architecture dict stored by the training script
    """
    pkl_path = pathlib.Path(pkl_path).expanduser().resolve()
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    config = pkl["config"]
    model = RefinerV11(rngs=rngs or nnx.Rngs(0), history=config["history"])
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pkl["state"])
    model = nnx.merge(graphdef, state)
    meta = {
        "tac_mean": np.asarray(pkl["tac_mean"], dtype=np.float32),
        "tac_std": np.asarray(pkl["tac_std"], dtype=np.float32),
        "config": config,
    }
    return model, meta
