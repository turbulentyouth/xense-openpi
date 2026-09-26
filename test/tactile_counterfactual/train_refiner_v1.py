"""Train the v1 tactile refiner (FiLM residual MLP) and run the five acceptance tests.

Model (~2.4M params, JAX nnx):

    tac [B,4,1024] --mean-pool--> Linear(1024->256) GELU --> z_tac
    z_tac --> Linear(256->512) --> FiLM (gamma, beta)   (256->512*2 halves)
    z_tac --> Linear(256->2)   --> aux water/no-water logits
    a_vla flattened [B,1600] --> Linear(1600->512) GELU --> h
    h' = gamma * h + beta
    h' --> Linear(512->512) GELU --> Linear(512->1600)
    delta = s * tanh(out / s), s = 2.0;  a_new = a_vla + delta

Loss: weighted mean_i w_i * ||a_new_i - a_demo_i||^2  +  0.1 * CE(aux, label),
w_i = 3 for in_window frames else 1; CE class weights = inverse train class
frequency; a_vla input dropout p=0.1 (whole-chunk zeroing, configurable).

Split: episode-grouped 128/32 train/val episodes (fixed seed). tac standardised
per-dim with train statistics; a_vla / a_demo left untouched.

Acceptance (val set):
  1. direction probe -- release_tcp_x thresholded at 0.7 into far/near; small
     logistic probes (episode-grouped 3-fold, protocol reused from
     tactile_linear_probe) on flattened a_new vs flattened a_vla.
  2. swap causality -- tac replaced by a random opposite-class val sample's tac:
     rel_l2 change of a_new, plus direction-prediction flip rate.
  3. L2 gain/loss -- ||a_new - a_demo|| vs ||a_vla - a_demo||, all / in_window.
  4. degradation -- tac zeroed: rel_l2(a_new, a_vla) (expected < 1%).
  5. temporal stability -- same-episode adjacent sampled frames: direction
     prediction agreement rate.

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/train_refiner_v1.py
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from test.tactile_counterfactual.tactile_linear_probe import logistic_fit  # noqa: E402

logger = logging.getLogger("train_refiner_v1")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

ACTION_DIMS = 1600  # 50 x 32 flattened
TAC_DIM = 1024
Z_DIM = 256
H_DIM = 512
DELTA_SCALE = 2.0
AUX_WEIGHT = 0.1
WINDOW_WEIGHT = 3.0
FAR_THRESHOLD = 0.7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"))
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "refiner_v1"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--a-vla-dropout", type=float, default=0.1, help="whole-chunk zeroing probability")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-episodes", type=int, default=32)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


class RefinerV1(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.tac_fc = nnx.Linear(TAC_DIM, Z_DIM, rngs=rngs)
        self.film = nnx.Linear(Z_DIM, 2 * H_DIM, rngs=rngs)
        self.aux = nnx.Linear(Z_DIM, 2, rngs=rngs)
        self.a_fc = nnx.Linear(ACTION_DIMS, H_DIM, rngs=rngs)
        self.mlp1 = nnx.Linear(H_DIM, H_DIM, rngs=rngs)
        self.mlp2 = nnx.Linear(H_DIM, ACTION_DIMS, rngs=rngs)

    def __call__(self, tac: jax.Array, a_vla: jax.Array) -> tuple[jax.Array, jax.Array]:
        """tac [B,4,1024] (standardised), a_vla [B,1600] -> (a_new [B,1600], aux logits [B,2])."""
        z = nnx.gelu(self.tac_fc(tac.mean(axis=1)))
        gamma, beta = jnp.split(self.film(z), 2, axis=-1)
        logits = self.aux(z)
        h = nnx.gelu(self.a_fc(a_vla))
        h = gamma * h + beta
        out = self.mlp2(nnx.gelu(self.mlp1(h)))
        delta = DELTA_SCALE * jnp.tanh(out / DELTA_SCALE)
        return a_vla + delta, logits


def count_params(state) -> int:
    return int(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(state) if hasattr(v, "shape")))


# --------------------------------------------------------------------------- #
# Train step / eval (graphdef closed over, state explicit)                     #
# --------------------------------------------------------------------------- #


def make_step_fn(graphdef, tx, class_weights: np.ndarray, aux_weight: float, a_vla_dropout: float):
    cw = jnp.asarray(class_weights, dtype=jnp.float32)

    def loss_fn(model, rng, tac, a_vla, a_demo, labels, weights):
        keep = jax.random.bernoulli(rng, 1.0 - a_vla_dropout, (a_vla.shape[0], 1))
        a_in = jnp.where(keep, a_vla, jnp.zeros_like(a_vla))
        a_new, logits = model(tac, a_in)
        per_sample = jnp.sum((a_new - a_demo) ** 2, axis=-1)
        action_loss = jnp.sum(weights * per_sample) / jnp.maximum(jnp.sum(weights), 1e-8)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), labels)
        ce = ce * cw[labels]
        aux_loss = jnp.mean(ce)
        acc = jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32))
        return action_loss + aux_weight * aux_loss, (action_loss, aux_loss, acc)

    def step(state, opt_state, rng, tac, a_vla, a_demo, labels, weights):
        model = nnx.merge(graphdef, state)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, rng, tac, a_vla, a_demo, labels, weights)
        params = state.filter(nnx.Param)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        action_loss, aux_loss, acc = aux
        return nnx.state(model), new_opt_state, loss, {
            "action_loss": action_loss,
            "aux_loss": aux_loss,
            "aux_acc": acc,
            "grad_norm": optax.global_norm(grads),
        }

    return jax.jit(step)


def make_eval_fn(graphdef):
    def eval_fn(state, tac, a_vla):
        model = nnx.merge(graphdef, state)
        return model(tac, a_vla)

    return jax.jit(eval_fn)


def batched_predict(eval_fn, state, tac, a_vla, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    outs, logits = [], []
    for i in range(0, tac.shape[0], chunk):
        o, l = eval_fn(state, jnp.asarray(tac[i : i + chunk]), jnp.asarray(a_vla[i : i + chunk]))
        outs.append(np.asarray(o, dtype=np.float32))
        logits.append(np.asarray(l, dtype=np.float32))
    return np.concatenate(outs), np.concatenate(logits)


def eval_metrics(a_new, logits, a_demo, labels, in_window) -> dict:
    per_sample = np.sqrt(np.sum((a_new - a_demo) ** 2, axis=-1))
    w = in_window.astype(bool)
    return {
        "l2_all": float(per_sample.mean()),
        "l2_window": float(per_sample[w].mean()) if w.any() else None,
        "aux_acc": float((logits.argmax(-1) == labels).mean()),
    }


# --------------------------------------------------------------------------- #
# Probe helpers (protocol reused from tactile_linear_probe)                    #
# --------------------------------------------------------------------------- #


def episode_grouped_oof_predictions(feats: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Out-of-fold episode-grouped 3-fold logistic predictions + accuracy."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_of_group[perm] = np.arange(len(uniq)) % 3
    folds = fold_of_group[np.searchsorted(uniq, groups)]
    pred = np.empty(len(labels), dtype=int)
    accs = []
    for f in range(3):
        tr, te = folds != f, folds == f
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-8
        w, b = logistic_fit((feats[tr] - mu) / sd, labels[tr].astype(np.float64))
        pred[te] = ((feats[te] - mu) / sd @ w + b > 0).astype(int)
        accs.append(float((pred[te] == labels[te]).mean()))
    return pred, {"fold_acc": accs, "mean_acc": float(np.mean(accs))}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.data)
    n = z["label"].shape[0]
    label = z["label"].astype(np.int64)
    episode = z["episode"].astype(np.int64)
    frame = z["frame"].astype(np.int64)
    in_window = z["in_window"].astype(bool)
    release_x = z["release_tcp_x"].astype(np.float32)
    logger.info("loaded %d frames from %s", n, args.data)

    # Episode-grouped split (fixed seed): 128/32 episodes.
    rng = np.random.default_rng(args.seed)
    uniq_eps = np.unique(episode)
    perm = rng.permutation(len(uniq_eps))
    val_eps = set(uniq_eps[perm[: args.val_episodes]].tolist())
    is_val = np.asarray([ep in val_eps for ep in episode])
    is_tr = ~is_val
    logger.info("split: %d train / %d val frames over %d/%d episodes", is_tr.sum(), is_val.sum(), len(uniq_eps) - len(val_eps), len(val_eps))

    # tac standardisation with train statistics.
    tac = z["tac"].astype(np.float32)
    mu = tac[is_tr].reshape(-1, TAC_DIM).mean(axis=0)
    sd = tac[is_tr].reshape(-1, TAC_DIM).std(axis=0) + 1e-8
    tac = (tac - mu) / sd
    a_vla = z["a_vla"].astype(np.float32).reshape(n, ACTION_DIMS)
    a_demo = z["a_demo"].astype(np.float32).reshape(n, ACTION_DIMS)

    weights = np.where(in_window, WINDOW_WEIGHT, 1.0).astype(np.float32)

    # CE class weights = inverse train class frequency (normalised to mean 1).
    n0, n1 = int((label[is_tr] == 0).sum()), int((label[is_tr] == 1).sum())
    cw = np.asarray([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)], dtype=np.float32)
    logger.info("train class counts no_water %d / water %d -> CE weights %.3f / %.3f", n0, n1, cw[0], cw[1])

    model = RefinerV1(rngs=nnx.Rngs(args.seed))
    graphdef, state = nnx.split(model)
    logger.info("refiner params: %d", count_params(state))

    tr_idx = np.nonzero(is_tr)[0]
    steps_per_epoch = int(np.ceil(len(tr_idx) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    tx = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = tx.init(state.filter(nnx.Param))
    step_fn = make_step_fn(graphdef, tx, cw, AUX_WEIGHT, args.a_vla_dropout)
    eval_fn = make_eval_fn(graphdef)

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + 1000 + epoch).permutation(tr_idx)
        for b0 in range(0, len(order), args.batch_size):
            bi = order[b0 : b0 + args.batch_size]
            step_rng = jax.random.key(args.seed * 1_000_003 + epoch * 10_000 + b0)
            state, opt_state, loss, m = step_fn(
                state,
                opt_state,
                step_rng,
                jnp.asarray(tac[bi]),
                jnp.asarray(a_vla[bi]),
                jnp.asarray(a_demo[bi]),
                jnp.asarray(label[bi]),
                jnp.asarray(weights[bi]),
            )
        # Full-set metrics each epoch.
        a_new_tr, logits_tr = batched_predict(eval_fn, state, tac[is_tr], a_vla[is_tr])
        a_new_va, logits_va = batched_predict(eval_fn, state, tac[is_val], a_vla[is_val])
        entry = {
            "epoch": epoch,
            "train": eval_metrics(a_new_tr, logits_tr, a_demo[is_tr], label[is_tr], in_window[is_tr]),
            "val": eval_metrics(a_new_va, logits_va, a_demo[is_val], label[is_val], in_window[is_val]),
            "elapsed_s": time.time() - t0,
        }
        history.append(entry)
        logger.info(
            "epoch %d: train L2 %.4f (win %.4f) acc %.3f | val L2 %.4f (win %.4f) acc %.3f",
            epoch,
            entry["train"]["l2_all"],
            entry["train"]["l2_window"] or float("nan"),
            entry["train"]["aux_acc"],
            entry["val"]["l2_all"],
            entry["val"]["l2_window"] or float("nan"),
            entry["val"]["aux_acc"],
        )

    # ------------------------------------------------------------------ #
    # Acceptance tests (val set)                                           #
    # ------------------------------------------------------------------ #
    tac_va, avla_va, ademo_va = tac[is_val], a_vla[is_val], a_demo[is_val]
    label_va, ep_va, fr_va, win_va = label[is_val], episode[is_val], frame[is_val], in_window[is_val]
    relx_va = release_x[is_val]
    a_new_va, logits_va = batched_predict(eval_fn, state, tac_va, avla_va)

    acceptance: dict = {}

    # 1. Direction probe: far (release x >= 0.7) vs near.
    dir_label = (relx_va >= FAR_THRESHOLD).astype(np.int64)
    pred_new, probe_new = episode_grouped_oof_predictions(a_new_va, dir_label, ep_va, seed=args.seed)
    pred_vla, probe_vla = episode_grouped_oof_predictions(avla_va, dir_label, ep_va, seed=args.seed)
    acceptance["direction_probe"] = {
        "far_threshold": FAR_THRESHOLD,
        "val_far_frac": float(dir_label.mean()),
        "a_new": probe_new,
        "a_vla": probe_vla,
    }
    logger.info("acc1 direction probe: a_new %.4f vs a_vla %.4f", probe_new["mean_acc"], probe_vla["mean_acc"])

    # 2. Swap causality: tac <- random opposite-class val sample's tac.
    swap_rng = np.random.default_rng(args.seed + 7)
    swap_src = np.empty(len(tac_va), dtype=np.int64)
    for c in (0, 1):
        idx_c = np.nonzero(label_va == c)[0]
        idx_other = np.nonzero(label_va != c)[0]
        swap_src[idx_c] = idx_other[swap_rng.integers(0, len(idx_other), len(idx_c))]
    a_new_swap, _ = batched_predict(eval_fn, state, tac_va[swap_src], avla_va)
    swap_rel = np.linalg.norm(a_new_swap - a_new_va, axis=-1) / np.maximum(np.linalg.norm(a_new_va, axis=-1), 1e-12)
    pred_swap, probe_swap = episode_grouped_oof_predictions(a_new_swap, dir_label, ep_va, seed=args.seed)
    flip = float((pred_swap != pred_new).mean())
    acceptance["swap_causality"] = {
        "rel_l2_mean": float(swap_rel.mean()),
        "rel_l2_median": float(np.median(swap_rel)),
        "direction_flip_rate": flip,
        "reference_vla_internal_swap_rel_l2": 0.0018,
        "a_new_swapped_probe": probe_swap,
    }
    logger.info("acc2 swap: rel_l2 %.5f, flip rate %.4f", swap_rel.mean(), flip)

    # 3. L2 gain/loss vs baseline.
    l2_new = np.sqrt(np.sum((a_new_va - ademo_va) ** 2, axis=-1))
    l2_vla = np.sqrt(np.sum((avla_va - ademo_va) ** 2, axis=-1))
    acceptance["l2_gain"] = {
        "all": {"a_new": float(l2_new.mean()), "a_vla": float(l2_vla.mean()), "ratio": float(l2_new.mean() / l2_vla.mean())},
        "in_window": {
            "a_new": float(l2_new[win_va].mean()),
            "a_vla": float(l2_vla[win_va].mean()),
            "ratio": float(l2_new[win_va].mean() / l2_vla[win_va].mean()),
        },
    }
    logger.info("acc3 L2: all %.4f->%.4f (x%.3f), window %.4f->%.4f (x%.3f)",
                l2_vla.mean(), l2_new.mean(), l2_new.mean() / l2_vla.mean(),
                l2_vla[win_va].mean(), l2_new[win_va].mean(), l2_new[win_va].mean() / l2_vla[win_va].mean())

    # 4. Degradation: tac zeroed.
    a_new_zero, _ = batched_predict(eval_fn, state, np.zeros_like(tac_va), avla_va)
    zero_rel = np.linalg.norm(a_new_zero - avla_va, axis=-1) / np.maximum(np.linalg.norm(avla_va, axis=-1), 1e-12)
    acceptance["degradation_zero_tac"] = {
        "rel_l2_mean": float(zero_rel.mean()),
        "rel_l2_max": float(zero_rel.max()),
        "expectation": "< 0.01",
    }
    logger.info("acc4 zero-tac: rel_l2 mean %.5f max %.5f", zero_rel.mean(), zero_rel.max())

    # 5. Temporal stability: adjacent sampled frames in the same episode.
    agree_new, agree_vla, npairs = 0, 0, 0
    order = np.lexsort((fr_va, ep_va))
    ep_s, fr_s = ep_va[order], fr_va[order]
    for i in range(len(order) - 1):
        if ep_s[i] == ep_s[i + 1] and fr_s[i + 1] - fr_s[i] == 2:
            npairs += 1
            agree_new += int(pred_new[order[i]] == pred_new[order[i + 1]])
            agree_vla += int(pred_vla[order[i]] == pred_vla[order[i + 1]])
    acceptance["temporal_stability"] = {
        "adjacent_pairs": npairs,
        "agree_rate_a_new": agree_new / max(npairs, 1),
        "agree_rate_a_vla": agree_vla / max(npairs, 1),
    }
    logger.info("acc5 temporal: %d pairs, agree a_new %.4f / a_vla %.4f", npairs, agree_new / max(npairs, 1), agree_vla / max(npairs, 1))

    # ------------------------------------------------------------------ #
    # Artifacts                                                            #
    # ------------------------------------------------------------------ #
    import pickle

    with open(out_dir / "refiner_params.pkl", "wb") as f:
        pickle.dump(
            {
                "state": state.to_pure_dict(),
                "tac_mean": mu,
                "tac_std": sd,
                "config": {
                    "action_dims": ACTION_DIMS,
                    "tac_dim": TAC_DIM,
                    "z_dim": Z_DIM,
                    "h_dim": H_DIM,
                    "delta_scale": DELTA_SCALE,
                    "aux_weight": AUX_WEIGHT,
                    "window_weight": WINDOW_WEIGHT,
                    "a_vla_dropout": args.a_vla_dropout,
                },
            },
            f,
        )
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=1))
    summary = {
        "args": vars(args),
        "n_frames": int(n),
        "n_train": int(is_tr.sum()),
        "n_val": int(is_val.sum()),
        "val_episodes": sorted(val_eps),
        "ce_class_weights": cw.tolist(),
        "n_params": count_params(state),
        "final": history[-1],
        "acceptance": acceptance,
    }
    (out_dir / "acceptance.json").write_text(json.dumps(summary, indent=1))
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
