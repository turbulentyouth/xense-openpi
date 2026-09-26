"""Linear probe: do the SCENE camera (SigLIP) features linearly separate water vs no-water?

Same protocol as ``tactile_linear_probe.py`` (grasp_start_frames.yaml, 320
samples, episode-grouped 3-fold L2 logistic regression), but the features are
the PaliGemma image-token embeddings that enter the prefix, mean-pooled over
the 256 tokens of each scene camera. If the scene features decode the label
as well as (or better than) the tactile features, the "model reads the weight
from vision instead of touch" shortcut hypothesis is supported.

Example:
    python test/tactile_counterfactual/scene_linear_probe.py \
        --checkpoint-dir checkpoints/ckpt59999_bf16 \
        --yaml grasp_start_frames.yaml --out ~/scene_probe
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
from test.tactile_counterfactual.tactile_linear_probe import probe_accuracy  # noqa: E402

logger = logging.getLogger("scene_linear_probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    p.add_argument("--yaml", default="grasp_start_frames.yaml")
    p.add_argument("--batch-size", type=int, default=16, help="images per image-tower call")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(open(args.yaml))
    records = []  # (label, episode, frame); yaml episodes are 1-based, dataset is 0-based
    for cat, lab in (("water", 1), ("no_water", 0)):
        for s in spec["categories"][cat]["samples"]:
            records.append((lab, int(s["episode"]) - 1, int(s["frame"])))
    labels = np.array([r[0] for r in records])
    episodes = np.array([r[1] for r in records])
    logger.info("samples: %d (water %d, no_water %d)", len(records), labels.sum(), (labels == 0).sum())

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    model, mc = setup.model, setup.model_config
    from openpi.models import model as _model

    cam_keys = list(_model.IMAGE_KEYS)  # 3 scene cameras; tactile keys are excluded by construction
    logger.info("scene cameras: %s", cam_keys)

    # Collect preprocessed scene images: [n, n_cam, h, w, c].
    imgs = []
    for i, (_, ep, fr) in enumerate(records):
        obs = setup.dataset.observation_from_sample(setup.dataset.get_sample(ep, fr))
        obs = model._preprocess_observation(None, obs, train=False)
        imgs.append(np.stack([np.asarray(obs.images[k])[0] for k in cam_keys]))
        if (i + 1) % 40 == 0:
            logger.info("preprocessed %d/%d samples", i + 1, len(records))
    imgs = np.stack(imgs)
    n, n_cam, h, w, c = imgs.shape

    # Same nnx.split + jit(state, x) pattern as the production policy.
    graphdef, state = nnx.split(model)

    def encode(state, x):
        m = nnx.merge(graphdef, state)
        tokens, _ = m.PaliGemma.img(x, train=False)  # [b, 256, width]
        return tokens

    jit_encode = jax.jit(encode)
    cam_feats = []  # per camera: [n, width]
    for j, cam in enumerate(cam_keys):
        outs = []
        flat = jnp.asarray(imgs[:, j])
        for s in range(0, n, args.batch_size):
            tok = jit_encode(state, flat[s : s + args.batch_size])
            if s == 0 and j == 0:
                ref = np.asarray(model.PaliGemma.img(flat[: args.batch_size], train=False)[0], np.float32)
                rel = float(np.linalg.norm(np.asarray(tok, np.float32) - ref) / np.linalg.norm(ref))
                logger.info("jit-vs-eager image tower sanity check: rel diff %.5f", rel)
                if rel > 0.05:
                    raise RuntimeError(f"jitted image tower disagrees with eager (rel {rel:.3f}); check XLA flags")
            outs.append(np.asarray(tok, np.float32))
        tok_all = np.concatenate(outs)  # [n, 256, width]
        cam_feats.append(tok_all.mean(axis=1))
        logger.info("encoded %s: tokens %s", cam, tok_all.shape)

    variants = {cam: f for cam, f in zip(cam_keys, cam_feats)}
    variants["all_cams_concat"] = np.concatenate(cam_feats, axis=-1)
    np.savez(out_dir / "features.npz", labels=labels, episodes=episodes, **variants)

    report = {"checkpoint": args.checkpoint_dir, "n_samples": int(n), "labels_from": args.yaml, "variants": {}}
    for name, X in variants.items():
        report["variants"][name] = probe_accuracy(X.astype(np.float64), labels, episodes)
        logger.info(
            "%-18s episode-grouped acc %.3f (folds %s) | random acc %.3f",
            name,
            report["variants"][name]["episode_grouped"]["mean_acc"],
            " ".join(f"{a:.3f}" for a in report["variants"][name]["episode_grouped"]["fold_acc"]),
            report["variants"][name]["random"]["mean_acc"],
        )
    (out_dir / "probe_report.json").write_text(json.dumps(lp.to_jsonable(report), indent=1, ensure_ascii=False))
    logger.info("wrote %s", out_dir / "probe_report.json")


if __name__ == "__main__":
    main()
