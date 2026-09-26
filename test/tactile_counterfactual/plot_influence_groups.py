"""Value-aware influence heatmaps of ACT queries over every key group.

For one frame, runs ``AttentionSampler`` (10 denoise steps, fixed noise) and
computes the value-aware influence share

    share[q, group] = sum_{k in group} prob[q,k] * ||W_o v[k]||  /  sum_all_keys,

per (step, layer, head, action query) -- dimensionless, in [0, 1], and the six
groups sum to 1, so a single shared colorbar makes the per-group heatmaps
directly comparable. Averaged over heads and the 50 ACT queries, each group
yields one [num_steps, depth] heatmap.

Example:
    python test/tactile_counterfactual/plot_influence_groups.py \
        --checkpoint-dir checkpoints/ckpt59999_bf16 --frame 1:316 \
        --out ~/tactile_influence_ep1f316
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

from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("plot_influence_groups")

# Key-group prefix -> display name. Order is the subplot order.
GROUPS = [
    ("IMG_BASE", "img_base"),
    ("IMG_LEFT_WRIST", "img_left_wrist"),
    ("IMG_RIGHT_WRIST", "img_right_wrist"),
    ("PROMPT", "prompt"),
    ("TAC_", "tactile"),
    ("ACT_", "act"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    p.add_argument("--frame", required=True, metavar="EP:FRAME")
    p.add_argument(
        "--swap-tactile-from",
        default=None,
        metavar="EP:FRAME",
        help="counterfactual: replace the frame's tactile images with this donor frame's, keep everything else",
    )
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vmax", type=float, default=None, help="force shared colorbar max (default: data max)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ep, fr = (int(x) for x in args.frame.split(":"))

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    mc = setup.model_config
    sample = setup.dataset.get_sample(ep, fr)
    if args.swap_tactile_from:
        from test.tactile_counterfactual.run_tactile_swap import swap_tactile

        d_ep, d_fr = (int(x) for x in args.swap_tactile_from.split(":"))
        donor = setup.dataset.get_sample(d_ep, d_fr)
        sample = swap_tactile(sample, donor, tuple(mc.tactile_image_keys))
        logger.info("tactile images swapped in from ep%d f%d", d_ep, d_fr)
    obs = setup.dataset.observation_from_sample(sample)
    noise = lp.fixed_noise(args.seed, 1, mc.action_horizon, mc.action_dim)
    trace = lp.AttentionSampler(setup.model)(obs, noise=noise, num_steps=args.num_steps)
    if trace.value_norm is None:
        raise RuntimeError("trace has no value_norm; gemma value collection is required")

    p_len, ah = trace.prefix_len, trace.action_horizon
    _, key_labels = lp.build_token_labels(p_len, len(mc.tactile_image_keys), ah, mc.max_token_len)

    contrib = np.asarray(trace.attention) * np.asarray(trace.value_norm)[..., None, :]  # [steps, depth, B, H, T, K]
    act = contrib[..., -ah:, :]  # ACT queries
    total = act.sum(axis=-1, keepdims=True)

    maps: dict[str, np.ndarray] = {}
    for prefix, name in GROUPS:
        idx = [i for i, label in enumerate(key_labels) if label.startswith(prefix)]
        share = act[..., idx].sum(axis=-1) / np.maximum(total[..., 0], 1e-20)
        maps[name] = share.mean(axis=(2, 3, 4))  # [steps, depth]
        np.save(out_dir / f"influence_{name}.npy", maps[name])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = args.vmax if args.vmax is not None else max(float(m.max()) for m in maps.values())
    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex=True, sharey=True)
    for ax, (_, name) in zip(axes.flat, GROUPS):
        im = ax.imshow(maps[name], aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"ACT -> {name}  (total {maps[name].sum():.1f})")
        ax.set_xlabel("layer")
        ax.set_ylabel("denoise step")
    # One shared colorbar for all subplots: shares are fractions of the same
    # per-query whole, so a common scale is what makes the groups comparable.
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    cax = fig.add_axes([0.945, 0.08, 0.012, 0.8])
    fig.colorbar(im, cax=cax, label="influence share (per-query normalised; groups sum to 1)")
    condition = f"ep{ep} f{fr}"
    if args.swap_tactile_from:
        condition += f", tactile <- ep{args.swap_tactile_from.replace(':', ' f')}"
    fig.suptitle(
        f"{condition}: value-aware ACT-query influence per key group [step x layer], shared colorbar [0, {vmax:.2f}]",
        y=0.99,
    )
    fig.savefig(out_dir / "influence_groups.png", dpi=110)

    meta = {
        "frame": {"episode": ep, "frame": fr},
        "swap_tactile_from": args.swap_tactile_from,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "shared_vmax": vmax,
        "group_totals": {name: float(m.sum()) for name, m in maps.items()},
        "definition": "share[q,group] = sum_k prob[q,k]*||W_o v[k]|| / sum_all; per-query normalised, groups sum to 1",
    }
    (out_dir / "influence_groups.json").write_text(json.dumps(lp.to_jsonable(meta), indent=1, ensure_ascii=False))
    logger.info("wrote %s (shared vmax %.3f)", out_dir / "influence_groups.png", vmax)
    for name, m in maps.items():
        logger.info("  %-14s total %6.2f  max cell %.3f at step %d layer %d", name, m.sum(), m.max(), *np.unravel_index(m.argmax(), m.shape))


if __name__ == "__main__":
    main()
