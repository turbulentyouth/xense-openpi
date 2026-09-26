"""Dump per-denoise-step attention for sampled dataset frames (analysis only).

Runs real pi05 inference (``AttentionSampler``, the scan-based replica of
``Pi0.sample_actions``) on frames sampled from the LeRobot dataset, with one fixed
noise chunk per frame, and streams results to disk via memmap — attention is never
accumulated in RAM.

Two modes:

* ``tactile`` (default): only the Action-query -> Tactile-key block is saved,
  ``attention_tactile.npy`` float16 ``[N, num_steps, depth, heads, action_horizon,
  num_tactile]`` (typical ``[N, 10, 18, 8, 50, 4]``).
* ``full``: the whole per-step attention, ``attention_full.npy`` float16
  ``[N, num_steps, depth, heads, Q, K]`` plus ``labels.json`` (query/key token
  labels) for BertViz. This is ~1000x larger per frame and capped at 8 frames
  unless ``--allow-large-full`` is given.

Both modes also save ``final_actions.npy``, ``v_t.npy``, ``denoise_times.npy``,
``frames.json`` and ``meta.json``. Nothing is trained; the production sampler and
BertViz are untouched.

Example (smoke):
    python scripts/dump_tactile_attention.py \
        --config-name pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100 \
        --checkpoint-dir checkpoints/59999 \
        --num-frames 4 --batch-size 2 --num-steps 2 --out /tmp/attn_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# NOTE: do NOT set --xla_gpu_autotune_level=0 here. With autotuning disabled,
# XLA's fallback bf16 conv algorithms produce numerically wrong results on this
# machine (isolated 1x1 conv rel error ~1.0 vs fp32), which silently poisons the
# FastViT/SigLIP features. The default autotune level is correct.

from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("dump_tactile_attention")

# Full mode writes [N, steps, depth, heads, Q, K] float16 — at the production shape
# that is ~240 MB per frame, versus ~0.3 MB for the tactile slice.
FULL_MODE_FRAME_CAP = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/59999")
    p.add_argument("--repo-id", default=None, help="override the dataset repo_id of the train config")
    p.add_argument("--episode", type=int, nargs="*", default=None, help="restrict sampling to these episodes")
    p.add_argument(
        "--frame",
        action="append",
        default=None,
        metavar="EP:FRAME",
        help="explicit frame(s) to dump, e.g. --frame 1:316 --frame 21:244. "
        "Overrides random sampling (--num-frames/--episode).",
    )
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4, help="dataloader workers for frame decoding")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--attention-mode", choices=("tactile", "full"), default="tactile")
    p.add_argument(
        "--allow-large-full",
        action="store_true",
        help=f"lift the full-mode frame cap ({FULL_MODE_FRAME_CAP})",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.attention_mode == "full":
        num_frames = len(args.frame) if args.frame else args.num_frames
        est_mb = num_frames * args.num_steps * 18 * 8 * 54 * 1022 * 2 / 1e6
        logger.warning(
            "full attention mode: ~%.0f MB of attention data for %d frames "
            "(vs ~%.1f MB in tactile mode). Use it for a few BertViz samples only.",
            est_mb,
            num_frames,
            num_frames * args.num_steps * 18 * 8 * 50 * 4 * 2 / 1e6,
        )
        if num_frames > FULL_MODE_FRAME_CAP and not args.allow_large_full:
            raise ValueError(
                f"full mode is capped at {FULL_MODE_FRAME_CAP} frames (got {num_frames}); "
                "pass --allow-large-full to override"
            )

    t0 = time.time()
    setup = lp.load_setup(args.config_name, args.checkpoint_dir, repo_id=args.repo_id)
    mc = setup.model_config
    ah, ad = mc.action_horizon, mc.action_dim
    n_tac = len(mc.tactile_image_keys)
    logger.info("setup loaded in %.1fs", time.time() - t0)

    if args.frame:
        eps, frs = zip(*(f.split(":") for f in args.frame))
        refs = lp.FrameRefs(
            episode=np.asarray(eps, dtype=np.int64),
            frame=np.asarray(frs, dtype=np.int64),
            batch_size=args.batch_size,
        )
        for ep, fr in zip(refs.episode.tolist(), refs.frame.tolist()):
            if not setup.dataset.has_sample(ep, fr):
                raise ValueError(f"(episode {ep}, frame {fr}) is not in the dataset index")
        logger.info("explicit frames: %s", list(zip(refs.episode.tolist(), refs.frame.tolist())))
    else:
        rng = np.random.default_rng(args.seed)
        refs = lp.sample_frames(
            setup.dataset,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
            rng=rng,
            min_tail=ah,
            episodes=args.episode,
        )
    noise = lp.fixed_noise(args.seed, len(refs), ah, ad)
    sampler = lp.AttentionSampler(setup.model)
    n = len(refs)

    # Memmaps are created lazily on the first batch: depth/heads/prefix_len are
    # read off the trace instead of being hardcoded.
    mm: dict[str, np.memmap] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    times_saved = False

    for bi, batch in enumerate(lp.iter_batches(setup.dataset, refs, num_workers=args.num_workers)):
        tb = time.time()
        idx = batch.index
        trace = sampler(batch.observation, noise=noise[idx], num_steps=args.num_steps)
        depth, heads = trace.attention.shape[1], trace.attention.shape[3]

        if not mm:
            shapes["v_t"] = (n, args.num_steps, ah, ad)
            shapes["final_actions"] = (n, ah, ad)
            if args.attention_mode == "tactile":
                shapes["attention_tactile"] = (n, args.num_steps, depth, heads, ah, n_tac)
            else:
                shapes["attention_full"] = (
                    n,
                    args.num_steps,
                    depth,
                    heads,
                    trace.suffix_len,
                    trace.prefix_len + trace.suffix_len,
                )
            for name, shape in shapes.items():
                dtype = np.float32 if name == "final_actions" else np.float16
                mm[name] = np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            logger.info("memmaps: %s", {k: v for k, v in shapes.items()})

        # Trace layout is [steps, depth, B, ...]; files are frame-major [N, ...].
        mm["v_t"][idx] = np.asarray(trace.v_t.transpose(1, 0, 2, 3), dtype=np.float16)
        mm["final_actions"][idx] = np.asarray(trace.final_action, dtype=np.float32)
        if args.attention_mode == "tactile":
            tac = lp.action_to_tactile_attention(trace)  # [steps, depth, B, heads, ah, n_tac]
            mm["attention_tactile"][idx] = np.asarray(tac.transpose(2, 0, 1, 3, 4, 5), dtype=np.float16)
        else:
            # [steps, depth, B, heads, Q, K] -> [B, steps, depth, heads, Q, K]
            mm["attention_full"][idx] = np.asarray(trace.attention.transpose(2, 0, 1, 3, 4, 5), dtype=np.float16)
        for m in mm.values():
            m.flush()

        if not times_saved:
            np.save(out_dir / "denoise_times.npy", np.asarray(trace.time, dtype=np.float32))
            times_saved = True
        logger.info(
            "batch %d/%d done in %.1fs (frames %s)", bi + 1, refs.num_batches, time.time() - tb, idx.tolist()
        )

    (out_dir / "frames.json").write_text(json.dumps(lp.to_jsonable(refs.to_dict()), indent=1))

    query_labels, key_labels = lp.build_token_labels(
        trace.prefix_len, n_tac, ah, mc.max_token_len
    )
    if args.attention_mode == "full":
        (out_dir / "labels.json").write_text(
            json.dumps(
                {
                    "query_labels": query_labels,
                    "key_labels": key_labels,
                    "note": "attention_full [N, steps, depth, heads, Q, K]; Q indexes query_labels, "
                    "K indexes key_labels. Values are raw softmax probabilities (float16).",
                },
                indent=1,
            )
        )

    meta = {
        "config_name": args.config_name,
        "checkpoint_dir": str(setup.checkpoint_dir),
        "repo_id": setup.data_config.repo_id,
        "num_frames": n,
        "num_steps": args.num_steps,
        "depth": int(depth),
        "num_heads": int(heads),
        "action_horizon": int(ah),
        "action_dim": int(ad),
        "num_tactile": int(n_tac),
        "prefix_len": int(trace.prefix_len),
        "suffix_len": int(trace.suffix_len),
        "seed": args.seed,
        "attention_mode": args.attention_mode,
        "attention_dtype": "float16",
        "tensor_shapes": {k: list(v) for k, v in shapes.items()},
        "tactile_keys": list(setup.tactile_keys),
        "episodes": sorted(set(refs.episode.tolist())),
    }
    (out_dir / "meta.json").write_text(json.dumps(lp.to_jsonable(meta), indent=1))
    logger.info("wrote %s in %.1fs", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
