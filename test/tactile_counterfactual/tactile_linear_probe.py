"""Linear probe: are FastViT tactile features linearly separable for water vs no-water?

Encodes the 4 tactile images of every sample in ``grasp_start_frames.yaml``
(160 water + 160 no_water, labels derived from release position) with the
checkpoint's tactile encoder, then fits an L2-regularised logistic regression
on the features and reports held-out accuracy.

Two evaluation protocols:
  * episode-grouped 3-fold: all samples of an episode land in the same fold
    (guards against near-duplicate frames leaking across train/test);
  * random 3-fold, for reference.

Feature variants: FastViT output and the post-``tactile_proj`` TAC tokens,
each either mean-pooled over the 4 views (1024-d) or flattened (4096-d).

Example:
    python test/tactile_counterfactual/tactile_linear_probe.py \
        --checkpoint-dir checkpoints/ckpt59999_bf16 \
        --yaml grasp_start_frames.yaml --out ~/tactile_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# NOTE: do NOT set --xla_gpu_autotune_level=0 here. With autotuning disabled,
# XLA's fallback bf16 conv algorithms produce numerically wrong results on this
# machine (isolated 1x1 conv rel error ~1.0 vs fp32), which silently poisons the
# FastViT/SigLIP features. The default autotune level is correct.

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402

from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("tactile_linear_probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    p.add_argument("--yaml", default="grasp_start_frames.yaml")
    p.add_argument("--batch-size", type=int, default=32, help="tactile images per encoder call")
    p.add_argument("--out", required=True)
    return p.parse_args()


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 3000, lr: float = 0.5) -> tuple:
    """Full-batch L2 logistic regression on standardised features (numpy)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def probe_accuracy(feats: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """3-fold accuracy under episode-grouped and random splits."""
    rng = np.random.default_rng(seed)
    out = {}
    for protocol in ("episode_grouped", "random"):
        if protocol == "episode_grouped":
            uniq = np.unique(groups)
            perm = rng.permutation(len(uniq))
            fold_of_group = np.empty(len(uniq), dtype=int)
            fold_of_group[perm] = np.arange(len(uniq)) % 3
            folds = fold_of_group[np.searchsorted(uniq, groups)]
        else:
            folds = np.arange(len(labels)) % 3
            rng.shuffle(folds)
        accs = []
        for f in range(3):
            tr, te = folds != f, folds == f
            mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-8
            Xtr, Xte = (feats[tr] - mu) / sd, (feats[te] - mu) / sd
            w, b = logistic_fit(Xtr, labels[tr].astype(np.float64))
            pred = (Xte @ w + b > 0).astype(int)
            accs.append(float((pred == labels[te]).mean()))
        out[protocol] = {"fold_acc": accs, "mean_acc": float(np.mean(accs))}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(open(args.yaml))
    records = []  # (label, episode, frame)
    for cat, lab in (("water", 1), ("no_water", 0)):
        for s in spec["categories"][cat]["samples"]:
            # The yaml numbers episodes 1..160; the dataset index is 0..159.
            records.append((lab, int(s["episode"]) - 1, int(s["frame"])))

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    model, mc = setup.model, setup.model_config
    keys = tuple(mc.tactile_image_keys)

    available = setup.dataset.episodes
    before = len(records)
    records = [r for r in records if r[1] in available]
    if len(records) < before:
        logger.warning("skipped %d samples whose episode is not in the dataset index", before - len(records))

    labels = np.array([r[0] for r in records])
    episodes = np.array([r[1] for r in records])
    logger.info("samples: %d (water %d, no_water %d), unique episodes %d",
                len(records), labels.sum(), (labels == 0).sum(), len(np.unique(episodes)))

    # Collect preprocessed tactile images for every sample.
    imgs_all = np.empty((len(records), len(keys)), dtype=object)
    for i, (_, ep, fr) in enumerate(records):
        obs = setup.dataset.observation_from_sample(setup.dataset.get_sample(ep, fr))
        obs = model._preprocess_observation(None, obs, train=False)
        for j, k in enumerate(keys):
            imgs_all[i, j] = np.asarray(obs.images[k])[0]  # drop batch dim
        if (i + 1) % 40 == 0:
            logger.info("preprocessed %d/%d samples", i + 1, len(records))
    n, nv = imgs_all.shape
    h, w, c = imgs_all[0, 0].shape
    flat = jnp.asarray(np.stack([imgs_all[i, j] for i in range(n) for j in range(nv)]))  # [n*4, h, w, c]

    # Encode through the same nnx.split + jit(state, x) pattern the production
    # policy uses (a closure over the nnx module inside jax.jit is NOT safe).
    graphdef, state = nnx.split(model)

    def encode_both(state, x):
        m = nnx.merge(graphdef, state)
        feats = m.tactile_encoder(x)
        return feats, m.tactile_proj(feats)

    jit_encode = jax.jit(encode_both)
    feats, toks = [], []
    for s in range(0, n * nv, args.batch_size):
        f_chunk, t_chunk = jit_encode(state, flat[s : s + args.batch_size])
        if s == 0:
            ref = np.asarray(model.tactile_encoder(flat[: args.batch_size]), np.float32)
            rel = float(np.linalg.norm(np.asarray(f_chunk, np.float32) - ref) / np.linalg.norm(ref))
            logger.info("jit-vs-eager encoder sanity check: rel diff %.5f", rel)
            if rel > 0.05:
                raise RuntimeError(
                    f"jitted tactile encoder disagrees with eager (rel {rel:.3f}); check XLA flags / conv numerics"
                )
        feats.append(np.asarray(f_chunk, np.float32))
        toks.append(np.asarray(t_chunk, np.float32))
    feats = np.concatenate(feats).reshape(n, nv, -1)  # [n, 4, feat]
    toks = np.concatenate(toks).reshape(n, nv, -1)
    logger.info("encoded: fastvit %s, tokens %s", feats.shape, toks.shape)

    variants = {
        "fastvit_mean4": feats.mean(axis=1),
        "fastvit_flat4": feats.reshape(n, -1),
        "token_mean4": toks.mean(axis=1),
        "token_flat4": toks.reshape(n, -1),
    }
    np.savez(out_dir / "features.npz", labels=labels, episodes=episodes, **variants)

    report = {"checkpoint": args.checkpoint_dir, "n_samples": int(n), "labels_from": args.yaml, "variants": {}}
    for name, X in variants.items():
        report["variants"][name] = probe_accuracy(X.astype(np.float64), labels, episodes)
        logger.info(
            "%-14s episode-grouped acc %.3f (folds %s) | random acc %.3f",
            name,
            report["variants"][name]["episode_grouped"]["mean_acc"],
            " ".join(f"{a:.3f}" for a in report["variants"][name]["episode_grouped"]["fold_acc"]),
            report["variants"][name]["random"]["mean_acc"],
        )
    (out_dir / "probe_report.json").write_text(json.dumps(lp.to_jsonable(report), indent=1, ensure_ascii=False))
    logger.info("wrote %s", out_dir / "probe_report.json")


if __name__ == "__main__":
    main()
