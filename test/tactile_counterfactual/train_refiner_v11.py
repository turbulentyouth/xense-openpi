"""Refiner v1.1: FiLM residual MLP + tactile history + swap augmentation.

Differences vs train_refiner_v1.py (v1 artifacts are NOT touched):

1. tac dropout p=0.1: the tactile window is replaced by the TRAIN MEAN (= 0 in
   the standardised feature space) AND the sample's target becomes a_vla
   itself -- "uninformative tactile -> identity correction". This is what makes
   the degradation check (tac <- train mean at eval) meaningful; in v1 the
   model was never taught to be identity there (acc4 measured 8.3%).
   The two dropouts are mutually exclusive per sample: p=0.1 tac-drop
   (identity target), p=0.1 a_vla-drop (v1 behaviour), else normal.
2. Tactile history: input is tac_hist [B, 4, 1024, ...] -> per time step the 4
   views are mean-pooled and passed through a SHARED Linear(1024->256)+GELU;
   the time axis is then reduced by a small GRU (--history gru, default) or by
   mean-pooling (--history mean, ablation).
3. Early stopping on val in-window L2, patience 5 (best state restored).
4. Swap augmentation (per swap_augment_research.py, option A): every eligible
   train sample (phase bin in delta_phase.npz `eligible`, i.e. phase 10-99
   with ||Delta||>=0.5) may additionally enter the epoch as a counterfactual
   copy: tac window replaced by a same-phase-bin OPPOSITE-class train donor's
   window, aux label flipped, and target = a_demo +/- Delta(bin) (SNR>2-masked
   class-difference table). a_vla is kept (the true counterfactual a_vla differs
   by only ~0.2%, per the VLA-internal swap measurements). The copy probability
   is set so counterfactuals are ~--cf-fraction of the training mixture.

Acceptance = v1's five tests + in/out-window breakdowns; the degradation test
now uses train-mean tac (expectation < 2%).

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/train_refiner_v11.py [--history gru|mean]
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

logger = logging.getLogger("train_refiner_v11")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SEGMENTS = REPO_ROOT / "outputs" / "stage_a" / "aux_labels.report.json"

ACTION_DIMS = 1600
TAC_DIM = 1024
FRAME_DIM = 256
GRU_DIM = 128
H_DIM = 512
DELTA_SCALE = 2.0
AUX_WEIGHT = 0.1
WINDOW_WEIGHT = 3.0
FAR_THRESHOLD = 0.7
HISTORY = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"))
    p.add_argument("--tac-history", default=str(REPO_ROOT / "outputs" / "refiner_data" / "tac_history.npz"))
    p.add_argument("--delta", default=str(REPO_ROOT / "outputs" / "refiner_v11" / "delta_phase.npz"))
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "refiner_v11"))
    p.add_argument("--history", choices=["gru", "mean"], default="gru")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--a-vla-dropout", type=float, default=0.1)
    p.add_argument("--tac-dropout", type=float, default=0.1)
    p.add_argument("--cf-fraction", type=float, default=0.4, help="target counterfactual share of the mixture")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-episodes", type=int, default=32)
    p.add_argument("--tag", default="", help="artifact subdir suffix (ablations)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Train step / eval                                                           #
# --------------------------------------------------------------------------- #


def make_step_fn(graphdef, tx, class_weights, a_vla_dropout, tac_dropout):
    cw = jnp.asarray(class_weights, dtype=jnp.float32)

    def loss_fn(model, rng, tac, a_vla, target, labels, weights):
        b = a_vla.shape[0]
        u = jax.random.uniform(rng, (b,))
        drop_tac = u < tac_dropout  # -> mean tac + identity target
        drop_avla = (u >= tac_dropout) & (u < tac_dropout + a_vla_dropout)
        tac_in = jnp.where(drop_tac[:, None, None, None], jnp.zeros_like(tac), tac)
        a_in = jnp.where(drop_avla[:, None], jnp.zeros_like(a_vla), a_vla)
        tgt = jnp.where(drop_tac[:, None], a_vla, target)
        a_new, logits = model(tac_in, a_in)
        per_sample = jnp.sum((a_new - tgt) ** 2, axis=-1)
        action_loss = jnp.sum(weights * per_sample) / jnp.maximum(jnp.sum(weights), 1e-8)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), labels)
        ce = ce * cw[labels]
        aux_loss = jnp.mean(ce)
        acc = jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32))
        return action_loss + AUX_WEIGHT * aux_loss, (action_loss, aux_loss, acc)

    def step(state, opt_state, rng, tac, a_vla, target, labels, weights):
        model = nnx.merge(graphdef, state)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, rng, tac, a_vla, target, labels, weights)
        params = state.filter(nnx.Param)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
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


def batched_predict(eval_fn, state, tac, a_vla, chunk: int = 4096):
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
# Probe helpers (same protocol as v1)                                          #
# --------------------------------------------------------------------------- #


def episode_grouped_oof_predictions(feats, labels, groups, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_of_group[perm] = np.arange(len(uniq)) % 3
    folds = fold_of_group[np.searchsorted(uniq, groups)]
    pred = np.empty(len(labels), dtype=int)
    for f in range(3):
        tr, te = folds != f, folds == f
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-8
        w, b = logistic_fit((feats[tr] - mu) / sd, labels[tr].astype(np.float64))
        pred[te] = ((feats[te] - mu) / sd @ w + b > 0).astype(int)
    acc = float((pred == labels).mean())
    return pred, {"mean_acc": acc}


def windowed_acc(pred, labels, win) -> dict:
    win = win.astype(bool)
    return {
        "all": float((pred == labels).mean()),
        "in_window": float((pred[win] == labels[win]).mean()) if win.any() else None,
        "out_window": float((pred[~win] == labels[~win]).mean()) if (~win).any() else None,
    }


# --------------------------------------------------------------------------- #
# Phase bins / augmentation table                                              #
# --------------------------------------------------------------------------- #


def compute_phase_bins(episodes: np.ndarray, frames: np.ndarray) -> np.ndarray:
    segs = {int(k): v for k, v in json.loads(SEGMENTS.read_text())["segments"].items()}
    phase = np.empty(len(frames), dtype=np.int64)
    for e, ss in segs.items():
        m = episodes == e
        if not m.any():
            continue
        starts = np.asarray([s[0] for s in ss])
        ends = np.asarray([s[1] for s in ss])
        idx = np.searchsorted(ends, frames[m], side="right")
        phase[m] = frames[m] - starts[idx]
    return phase


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    if args.tag:
        out_dir = out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.data)
    zh = np.load(args.tac_history)
    zd = np.load(args.delta)
    n = z["label"].shape[0]
    label = z["label"].astype(np.int64)
    episode = z["episode"].astype(np.int64)
    frame = z["frame"].astype(np.int64)
    in_window = z["in_window"].astype(bool)
    release_x = z["release_tcp_x"].astype(np.float32)
    tac_hist = zh["tac_hist"].astype(np.float32)  # [N, 4, 4, 1024]
    assert tac_hist.shape[0] == n
    logger.info("loaded %d frames + history %s", n, tac_hist.shape)

    bin_width = int(zd["bin_width"])
    phase = compute_phase_bins(episode, frame)
    phase_bin = np.minimum(phase // bin_width, zd["delta"].shape[0] - 1)
    eligible = zd["eligible"]
    delta_table = zd["delta"].astype(np.float32)
    is_eligible = eligible[phase_bin]
    logger.info("eligible frames for augmentation: %d / %d", is_eligible.sum(), n)

    # Same episode-grouped split as v1 (seed 0, first 32 permuted eps -> val).
    rng = np.random.default_rng(args.seed)
    uniq_eps = np.unique(episode)
    perm = rng.permutation(len(uniq_eps))
    val_eps = set(uniq_eps[perm[: args.val_episodes]].tolist())
    is_val = np.asarray([ep in val_eps for ep in episode])
    is_tr = ~is_val
    logger.info("split: %d train / %d val frames", is_tr.sum(), is_val.sum())

    # Standardise history with train stats over all time slots.
    flat = tac_hist[is_tr].reshape(-1, TAC_DIM)
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-8
    tac_hist = (tac_hist - mu) / sd
    a_vla = z["a_vla"].astype(np.float32).reshape(n, ACTION_DIMS)
    a_demo = z["a_demo"].astype(np.float32).reshape(n, ACTION_DIMS)
    weights = np.where(in_window, WINDOW_WEIGHT, 1.0).astype(np.float32)

    n0, n1 = int((label[is_tr] == 0).sum()), int((label[is_tr] == 1).sum())
    cw = np.asarray([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)], dtype=np.float32)
    logger.info("CE weights %.3f / %.3f", cw[0], cw[1])

    # Counterfactual copies: for every eligible TRAIN sample, a donor of the
    # opposite class from the same phase bin (train only); copy probability set
    # so counterfactuals are ~cf-fraction of the training mixture.
    tr_idx = np.nonzero(is_tr)[0]
    tr_elig = tr_idx[is_eligible[tr_idx]]
    p_copy = min(1.0, (args.cf_fraction / max(1.0 - args.cf_fraction, 1e-9)) * (len(tr_idx) / max(len(tr_elig), 1)))
    aug_rng = np.random.default_rng(args.seed + 11)
    copy_mask = aug_rng.random(len(tr_elig)) < p_copy
    aug_src = tr_elig[copy_mask]
    donor = np.empty(len(aug_src), dtype=np.int64)
    by_bin_class: dict[tuple[int, int], np.ndarray] = {}
    for c in (0, 1):
        for b in np.unique(phase_bin[tr_idx]):
            sel = tr_idx[(phase_bin[tr_idx] == b) & (label[tr_idx] == c)]
            if len(sel):
                by_bin_class[(int(b), c)] = sel
    for k, i in enumerate(aug_src):
        c, b = int(label[i]), int(phase_bin[i])
        pool = by_bin_class.get((b, 1 - c))
        if pool is None:  # nearest bin with an opposite-class donor
            for off in range(1, int(zd["delta"].shape[0])):
                pool = by_bin_class.get((b - off, 1 - c)) or by_bin_class.get((b + off, 1 - c))
                if pool is not None:
                    break
        donor[k] = pool[aug_rng.integers(0, len(pool))]
    sign = np.where(label[aug_src] == 0, 1.0, -1.0).astype(np.float32)  # no_water->+Delta, water->-Delta
    cf_tac = tac_hist[donor]
    cf_avla = a_vla[aug_src]
    cf_target = a_demo[aug_src] + sign[:, None] * delta_table.reshape(delta_table.shape[0], -1)[phase_bin[aug_src]]
    cf_label = 1 - label[aug_src]
    cf_weights = weights[aug_src]
    mix_tac = np.concatenate([tac_hist[tr_idx], cf_tac])
    mix_avla = np.concatenate([a_vla[tr_idx], cf_avla])
    mix_target = np.concatenate([a_demo[tr_idx], cf_target])
    mix_label = np.concatenate([label[tr_idx], cf_label])
    mix_weights = np.concatenate([weights[tr_idx], cf_weights])
    cf_frac = len(aug_src) / len(mix_label)
    logger.info(
        "augmentation: %d counterfactual copies (p_copy=%.2f) -> mixture %d, cf fraction %.3f",
        len(aug_src),
        p_copy,
        len(mix_label),
        cf_frac,
    )

    model = RefinerV11(rngs=nnx.Rngs(args.seed), history=args.history)
    graphdef, state = nnx.split(model)
    n_params = int(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(state) if hasattr(v, "shape")))
    logger.info("refiner v1.1 (%s) params: %d", args.history, n_params)

    steps_per_epoch = int(np.ceil(len(mix_label) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    tx = optax.adamw(optax.cosine_decay_schedule(args.lr, total_steps), weight_decay=args.weight_decay)
    opt_state = tx.init(state.filter(nnx.Param))
    step_fn = make_step_fn(graphdef, tx, cw, args.a_vla_dropout, args.tac_dropout)
    eval_fn = make_eval_fn(graphdef)

    history_log = []
    best = {"val_window_l2": float("inf"), "epoch": -1, "state": None}
    bad = 0
    t0 = time.time()
    stop_epoch = args.epochs
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + 1000 + epoch).permutation(len(mix_label))
        for b0 in range(0, len(order), args.batch_size):
            bi = order[b0 : b0 + args.batch_size]
            step_rng = jax.random.key(args.seed * 1_000_003 + epoch * 10_000 + b0)
            state, opt_state, loss, m = step_fn(
                state,
                opt_state,
                step_rng,
                jnp.asarray(mix_tac[bi]),
                jnp.asarray(mix_avla[bi]),
                jnp.asarray(mix_target[bi]),
                jnp.asarray(mix_label[bi]),
                jnp.asarray(mix_weights[bi]),
            )
        a_new_tr, logits_tr = batched_predict(eval_fn, state, tac_hist[is_tr], a_vla[is_tr])
        a_new_va, logits_va = batched_predict(eval_fn, state, tac_hist[is_val], a_vla[is_val])
        entry = {
            "epoch": epoch,
            "train": eval_metrics(a_new_tr, logits_tr, a_demo[is_tr], label[is_tr], in_window[is_tr]),
            "val": eval_metrics(a_new_va, logits_va, a_demo[is_val], label[is_val], in_window[is_val]),
            "elapsed_s": time.time() - t0,
        }
        history_log.append(entry)
        vw = entry["val"]["l2_window"]
        if vw < best["val_window_l2"] - 1e-4:
            best = {"val_window_l2": vw, "epoch": epoch,
                    "state": jax.tree.map(lambda x: np.asarray(x).copy(), state.to_pure_dict())}
            bad = 0
        else:
            bad += 1
        logger.info(
            "epoch %d: train L2 %.4f (win %.4f) acc %.3f | val L2 %.4f (win %.4f) acc %.3f%s",
            epoch,
            entry["train"]["l2_all"], entry["train"]["l2_window"] or float("nan"), entry["train"]["aux_acc"],
            entry["val"]["l2_all"], vw or float("nan"), entry["val"]["aux_acc"],
            " *" if bad == 0 else "",
        )
        if bad >= args.patience:
            stop_epoch = epoch + 1
            logger.info("early stop at epoch %d (best %d, val win L2 %.4f)", epoch, best["epoch"], best["val_window_l2"])
            break

    if best["state"] is not None:
        state.replace_by_pure_dict(best["state"])
        logger.info("restored best state from epoch %d", best["epoch"])

    # ------------------------------------------------------------------ #
    # Acceptance (val set)                                                 #
    # ------------------------------------------------------------------ #
    tac_va, avla_va, ademo_va = tac_hist[is_val], a_vla[is_val], a_demo[is_val]
    label_va, ep_va, fr_va, win_va = label[is_val], episode[is_val], frame[is_val], in_window[is_val]
    relx_va = release_x[is_val]
    a_new_va, _ = batched_predict(eval_fn, state, tac_va, avla_va)
    acceptance: dict = {}

    # 1. direction probe (+ window breakdown)
    dir_label = (relx_va >= FAR_THRESHOLD).astype(np.int64)
    pred_new, _ = episode_grouped_oof_predictions(a_new_va, dir_label, ep_va, seed=args.seed)
    pred_vla, _ = episode_grouped_oof_predictions(avla_va, dir_label, ep_va, seed=args.seed)
    acceptance["direction_probe"] = {
        "far_threshold": FAR_THRESHOLD,
        "val_far_frac": float(dir_label.mean()),
        "a_new": windowed_acc(pred_new, dir_label, win_va),
        "a_vla": windowed_acc(pred_vla, dir_label, win_va),
    }
    logger.info("acc1 direction a_new %s vs a_vla %s", acceptance["direction_probe"]["a_new"], acceptance["direction_probe"]["a_vla"])

    # 2. swap causality (+ window breakdown)
    swap_rng = np.random.default_rng(args.seed + 7)
    swap_src = np.empty(len(tac_va), dtype=np.int64)
    for c in (0, 1):
        idx_c = np.nonzero(label_va == c)[0]
        idx_other = np.nonzero(label_va != c)[0]
        swap_src[idx_c] = idx_other[swap_rng.integers(0, len(idx_other), len(idx_c))]
    a_new_swap, _ = batched_predict(eval_fn, state, tac_va[swap_src], avla_va)
    swap_rel = np.linalg.norm(a_new_swap - a_new_va, axis=-1) / np.maximum(np.linalg.norm(a_new_va, axis=-1), 1e-12)
    pred_swap, _ = episode_grouped_oof_predictions(a_new_swap, dir_label, ep_va, seed=args.seed)
    flip = pred_swap != pred_new
    acceptance["swap_causality"] = {
        "rel_l2_mean": float(swap_rel.mean()),
        "rel_l2_in_window": float(swap_rel[win_va].mean()),
        "rel_l2_out_window": float(swap_rel[~win_va].mean()),
        "direction_flip_rate": float(flip.mean()),
        "flip_in_window": float(flip[win_va].mean()),
        "flip_out_window": float(flip[~win_va].mean()),
    }
    logger.info("acc2 swap: rel %.5f flip %.4f (win %.4f)", swap_rel.mean(), flip.mean(), flip[win_va].mean())

    # 3. L2 gain/loss
    l2_new = np.sqrt(np.sum((a_new_va - ademo_va) ** 2, axis=-1))
    l2_vla = np.sqrt(np.sum((avla_va - ademo_va) ** 2, axis=-1))
    acceptance["l2_gain"] = {
        "all": {"a_new": float(l2_new.mean()), "a_vla": float(l2_vla.mean()), "ratio": float(l2_new.mean() / l2_vla.mean())},
        "in_window": {"a_new": float(l2_new[win_va].mean()), "a_vla": float(l2_vla[win_va].mean()),
                      "ratio": float(l2_new[win_va].mean() / l2_vla[win_va].mean())},
        "out_window": {"a_new": float(l2_new[~win_va].mean()), "a_vla": float(l2_vla[~win_va].mean()),
                       "ratio": float(l2_new[~win_va].mean() / l2_vla[~win_va].mean())},
    }
    logger.info("acc3 L2 all x%.3f win x%.3f out x%.3f",
                acceptance["l2_gain"]["all"]["ratio"], acceptance["l2_gain"]["in_window"]["ratio"], acceptance["l2_gain"]["out_window"]["ratio"])

    # 4. degradation: tac <- train mean (0 in standardised space)
    a_new_mean, _ = batched_predict(eval_fn, state, np.zeros_like(tac_va), avla_va)
    mean_rel = np.linalg.norm(a_new_mean - avla_va, axis=-1) / np.maximum(np.linalg.norm(avla_va, axis=-1), 1e-12)
    acceptance["degradation_mean_tac"] = {
        "rel_l2_mean": float(mean_rel.mean()),
        "rel_l2_max": float(mean_rel.max()),
        "expectation": "< 0.02",
    }
    logger.info("acc4 mean-tac: rel %.5f max %.5f", mean_rel.mean(), mean_rel.max())

    # 5. temporal stability
    agree_new = agree_vla = npairs = 0
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
    logger.info("acc5 temporal: %d pairs, a_new %.4f / a_vla %.4f", npairs, agree_new / max(npairs, 1), agree_vla / max(npairs, 1))

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
                    "history": args.history,
                    "action_dims": ACTION_DIMS,
                    "tac_dim": TAC_DIM,
                    "frame_dim": FRAME_DIM,
                    "gru_dim": GRU_DIM if args.history == "gru" else None,
                    "h_dim": H_DIM,
                    "delta_scale": DELTA_SCALE,
                    "tac_dropout": args.tac_dropout,
                    "a_vla_dropout": args.a_vla_dropout,
                },
            },
            f,
        )
    (out_dir / "train_history.json").write_text(json.dumps(history_log, indent=1))
    summary = {
        "args": vars(args),
        "n_frames": int(n),
        "n_train": int(is_tr.sum()),
        "n_val": int(is_val.sum()),
        "val_episodes": sorted(val_eps),
        "ce_class_weights": cw.tolist(),
        "n_params": n_params,
        "augmentation": {
            "n_counterfactual": int(len(aug_src)),
            "p_copy": float(p_copy),
            "cf_fraction_realised": float(cf_frac),
            "mixture_size": int(len(mix_label)),
        },
        "early_stop": {"stop_epoch": stop_epoch, "best_epoch": best["epoch"], "best_val_window_l2": best["val_window_l2"]},
        "acceptance": acceptance,
    }
    (out_dir / "acceptance.json").write_text(json.dumps(summary, indent=1))
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
