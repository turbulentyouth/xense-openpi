"""End-to-end verification for the attention dump tooling.

Checks (todo section 5), given one tactile-mode dump and one full-mode dump
produced with the SAME seed / num-frames / num-steps:

1. attention files have the shapes meta.json promises;
2. no NaN anywhere;
3. attention values are within [0, 1] (raw softmax probabilities);
4. the tactile-mode file equals the Action->Tactile slice of the full-mode file
   for the same frame;
5. each dumped final action matches an AttentionSampler single-sample (B=1) run
   on the same frame with the same noise;
6. optionally, a second tactile dump (--tactile-dir-2) is compared bitwise for
   same-seed reproducibility.

Requires the dataset + checkpoint, so this is a script, not a unit test.

Example:
    python test/tactile_counterfactual/verify_attention_dump.py \
        --tactile-dir /tmp/attn_smoke --full-dir /tmp/attn_full_smoke \
        --tactile-dir-2 /tmp/attn_smoke2 \
        --checkpoint-dir checkpoints/59999 --num-steps 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

# NOTE: do NOT set --xla_gpu_autotune_level=0 here. With autotuning disabled,
# XLA's fallback bf16 conv algorithms produce numerically wrong results on this
# machine (isolated 1x1 conv rel error ~1.0 vs fp32), which silently poisons the
# FastViT/SigLIP features. The default autotune level is correct.

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("verify_attention_dump")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tactile-dir", required=True)
    p.add_argument("--full-dir", default=None)
    p.add_argument("--tactile-dir-2", default=None, help="second tactile dump for the reproducibility diff")
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/59999")
    p.add_argument("--num-steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtol", type=float, default=lp.EQ_RTOL)
    p.add_argument("--atol", type=float, default=lp.EQ_ATOL)
    # The single-sample check re-runs frames at B=1 while the dump batched them
    # (B=2). Different batch shapes make XLA pick different GEMM kernels, which
    # shifts bf16 results by ~1e-2 after 10 Euler steps. This check guards against
    # pipeline errors (wrong frame / noise / preprocessing, which give O(1)
    # errors), so it uses a coarser tolerance than the same-shape comparisons.
    p.add_argument("--single-rtol", type=float, default=0.05)
    p.add_argument("--single-atol", type=float, default=0.02)
    return p.parse_args()


def check_files(d: pathlib.Path, name: str) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    attn_name = "attention_tactile" if meta["attention_mode"] == "tactile" else "attention_full"
    attn = np.load(d / f"{attn_name}.npy", mmap_mode="r")
    assert tuple(attn.shape) == tuple(meta["tensor_shapes"][attn_name]), f"{name}: shape mismatch {attn.shape}"
    assert attn.shape[1] == meta["num_steps"] == args.num_steps
    assert attn.shape[2] == meta["depth"] and attn.shape[3] == meta["num_heads"]
    a32 = np.asarray(attn, dtype=np.float32)
    assert not np.isnan(a32).any(), f"{name}: NaN in attention"
    assert a32.min() >= 0.0 and a32.max() <= 1.0, f"{name}: attention out of [0,1]: [{a32.min()}, {a32.max()}]"
    fa = np.asarray(np.load(d / "final_actions.npy", mmap_mode="r"), dtype=np.float32)
    assert not np.isnan(fa).any(), f"{name}: NaN in final_actions"
    frames = json.loads((d / "frames.json").read_text())
    logger.info(
        "%s OK: %s %s, attention in [%.4f, %.4f], no NaN", name, attn_name, attn.shape, a32.min(), a32.max()
    )
    return {"meta": meta, "attn": a32, "final_actions": fa, "frames": frames}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    tac = check_files(pathlib.Path(args.tactile_dir), "tactile dump")

    if args.tactile_dir_2:
        tac2 = check_files(pathlib.Path(args.tactile_dir_2), "tactile dump 2")
        assert tac["frames"] == tac2["frames"], "same seed gave different frames"
        assert np.array_equal(tac["attn"], tac2["attn"]), "same seed gave different attention"
        assert np.array_equal(tac["final_actions"], tac2["final_actions"])
        logger.info("reproducibility OK: two same-seed dumps are bit-identical")

    if args.full_dir:
        full = check_files(pathlib.Path(args.full_dir), "full dump")
        assert full["frames"] == tac["frames"], "tactile and full dumps sampled different frames"
        meta = tac["meta"]
        ah, n_tac, p_len = meta["action_horizon"], meta["num_tactile"], meta["prefix_len"]
        # Full layout: [N, steps, depth, heads, Q, K]; action queries are the last ah
        # rows, tactile keys are columns p_len : p_len + n_tac.
        tac_slice = full["attn"][..., -ah:, p_len : p_len + n_tac]
        assert tac_slice.shape == tac["attn"].shape
        if not np.array_equal(tac_slice, tac["attn"]):
            diff = np.abs(tac_slice - tac["attn"]).max()
            assert diff <= 2e-3, f"tactile vs full slice mismatch: max diff {diff}"
            logger.warning("tactile vs full slice not bit-identical, max diff %.2e (within fp16 eps)", diff)
        logger.info("tactile-mode == full-mode Action->Tactile slice OK")

    # Per-frame single-sample consistency with AttentionSampler.
    logger.info("loading model for single-sample consistency check ...")
    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    mc = setup.model_config
    frames = tac["frames"]
    n = len(frames["episode"])
    noise = lp.fixed_noise(args.seed, n, mc.action_horizon, mc.action_dim)
    sampler = lp.AttentionSampler(setup.model)
    worst = 0.0
    for i in range(n):
        sample = setup.dataset.get_sample(frames["episode"][i], frames["frame"][i])
        obs = setup.dataset.observation_from_sample(sample)  # B=1
        trace = sampler(obs, noise=noise[i : i + 1], num_steps=args.num_steps)
        got = np.asarray(trace.final_action[0], dtype=np.float32)
        want = tac["final_actions"][i]
        np.testing.assert_allclose(got, want, rtol=args.single_rtol, atol=args.single_atol)
        worst = max(worst, float(np.abs(got - want).max()))
        logger.info("frame %d (ep %d fr %d): max |diff| %.3e", i, frames["episode"][i], frames["frame"][i],
                    float(np.abs(got - want).max()))
    logger.info(
        "single-sample consistency OK (worst max diff %.3e, rtol=%g atol=%g)",
        worst, args.single_rtol, args.single_atol,
    )
    print("ALL CHECKS PASSED")
