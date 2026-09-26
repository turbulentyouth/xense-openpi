"""Tactile counterfactual swap experiment (analysis only, nothing in the model changes).

Four conditions from two grasp-start frames (see grasp_start_frames.yaml):

- ``water_orig``        water episode frame, untouched
- ``water_tac_dry``     same water frame, but the 4 tactile images replaced by the
                        no-water frame's tactile images
- ``dry_orig``          no-water episode frame, untouched
- ``dry_tac_water``     same no-water frame, tactile replaced by the water frame's

All conditions share the same fixed noise and num_steps, and run through
``AttentionSampler`` (the scan-based replica of ``Pi0.sample_actions``), so any
difference in Action->Tactile attention or final actions is caused solely by the
tactile content.

Outputs under ``--out``: per-condition ``tac_attn_<name>.npy``
[steps, depth, heads, action_horizon, num_tactile] float16, ``final_<name>.npy``,
``v_t_<name>.npy``, plus ``summary.json`` and heatmap PNGs comparing orig vs swap
in both directions.

Example:
    python test/tactile_counterfactual/run_tactile_swap.py \
        --checkpoint-dir /tmp/ckpt59999_bf16 \
        --water 1:316 --no-water 21:244 --out ~/tactile_swap_ep1f316_ep21f244
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

logger = logging.getLogger("tactile_swap")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="/tmp/ckpt59999_bf16")
    p.add_argument("--water", default="1:316", metavar="EP:FRAME", help="water (grasp with water) frame")
    p.add_argument("--no-water", default="21:244", metavar="EP:FRAME", help="no-water (dry grasp) frame")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mask-act-to-prefix",
        action="store_true",
        help="hide all prefix keys from the action-token queries (AttentionSampler "
        "mask_act_to_prefix); tactile queries are untouched",
    )
    p.add_argument(
        "--drop-prefix",
        action="store_true",
        help="never compute the prefix at all (AttentionSampler drop_prefix): no VLM "
        "forward, no KV cache, suffix-only attention for every query row; mutually "
        "exclusive with --mask-act-to-prefix",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.mask_act_to_prefix and args.drop_prefix:
        p.error("--mask-act-to-prefix and --drop-prefix are mutually exclusive")
    return args


def swap_tactile(sample: dict, donor: dict, tactile_keys: tuple[str, ...]) -> dict:
    """Return a copy of ``sample`` whose tactile images come from ``donor``.

    Both samples are transformed dataset rows, so tactile values live in
    ``sample["image"][key]`` already normalized by the shared data pipeline —
    swapping them keeps every other input (scene images, state, prompt) intact.
    """
    out = dict(sample)
    out["image"] = dict(sample["image"])
    out["image_mask"] = dict(sample["image_mask"])
    for key in tactile_keys:
        out["image"][key] = donor["image"][key]
        out["image_mask"][key] = donor["image_mask"][key]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    mc = setup.model_config
    ah, ad = mc.action_horizon, mc.action_dim
    tactile_keys = tuple(mc.tactile_image_keys)
    logger.info("tactile keys: %s", tactile_keys)

    w_ep, w_fr = (int(x) for x in args.water.split(":"))
    d_ep, d_fr = (int(x) for x in args.no_water.split(":"))
    water_sample = setup.dataset.get_sample(w_ep, w_fr)
    dry_sample = setup.dataset.get_sample(d_ep, d_fr)

    conditions = {
        "water_orig": water_sample,
        "water_tac_dry": swap_tactile(water_sample, dry_sample, tactile_keys),
        "dry_orig": dry_sample,
        "dry_tac_water": swap_tactile(dry_sample, water_sample, tactile_keys),
    }

    noise = lp.fixed_noise(args.seed, 1, ah, ad)  # identical noise for all conditions
    sampler = lp.AttentionSampler(setup.model)

    results: dict[str, dict[str, np.ndarray]] = {}
    for name, sample in conditions.items():
        obs = setup.dataset.observation_from_sample(sample)
        trace = sampler(
            obs,
            noise=noise,
            num_steps=args.num_steps,
            mask_act_to_prefix=args.mask_act_to_prefix,
            drop_prefix=args.drop_prefix,
        )
        tac = np.asarray(lp.action_to_tactile_attention(trace))  # [steps, depth, B, heads, ah, n_tac]
        tac = tac[:, :, 0].astype(np.float16)  # B=1 -> [steps, depth, heads, ah, n_tac]
        final = np.asarray(trace.final_action[0], dtype=np.float32)
        v_t = np.asarray(trace.v_t[:, 0], dtype=np.float16)
        results[name] = {"tac": tac, "final": final, "v_t": v_t}
        np.save(out_dir / f"tac_attn_{name}.npy", tac)
        np.save(out_dir / f"final_{name}.npy", final)
        np.save(out_dir / f"v_t_{name}.npy", v_t)
        logger.info(
            "%s done: tac %s in [%.4f, %.4f], final action std %.4f",
            name, tac.shape, float(tac.min()), float(tac.max()), float(final.std()),
        )

    summary = analyze(results, out_dir, args, (w_ep, w_fr), (d_ep, d_fr), tactile_keys)
    (out_dir / "summary.json").write_text(json.dumps(lp.to_jsonable(summary), indent=1, ensure_ascii=False))
    logger.info("wrote %s", out_dir)


def analyze(results, out_dir, args, water_ref, dry_ref, tactile_keys) -> dict:
    """Quantitative orig-vs-swap comparison in both directions + heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mass(tac: np.ndarray) -> np.ndarray:
        # [steps, depth, heads, ah, n_tac] -> per (step, layer) mass: mean over
        # heads/queries of prob summed over the tactile keys.
        return tac.astype(np.float32).sum(axis=-1).mean(axis=(2, 3))

    def head_mass(tac: np.ndarray) -> np.ndarray:
        # -> [steps, depth, heads]: mass per head
        return tac.astype(np.float32).sum(axis=-1).mean(axis=-2)

    summary: dict = {
        "water_ref": {"episode": water_ref[0], "frame": water_ref[1]},
        "dry_ref": {"episode": dry_ref[0], "frame": dry_ref[1]},
        "num_steps": args.num_steps,
        "seed": args.seed,
        "mask_act_to_prefix": bool(getattr(args, "mask_act_to_prefix", False)),
        "drop_prefix": bool(getattr(args, "drop_prefix", False)),
        "tactile_keys": list(tactile_keys),
        "conditions": {},
        "comparisons": {},
    }

    for name, r in results.items():
        m = mass(r["tac"])  # [steps, depth]
        summary["conditions"][name] = {
            "tac_mass_total_per_step": m.sum(axis=1).tolist(),
            "tac_mass_per_step_layer": m.tolist(),
            "final_action": r["final"].tolist(),
        }

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for row, (base, swap) in enumerate((("water_orig", "water_tac_dry"), ("dry_orig", "dry_tac_water"))):
        m_base, m_swap = mass(results[base]["tac"]), mass(results[swap]["tac"])
        delta = m_swap - m_base
        a_l2 = float(np.linalg.norm(results[swap]["final"] - results[base]["final"]))
        a_ref = float(np.linalg.norm(results[base]["final"]))
        rel = a_l2 / max(a_ref, 1e-12)
        vt_l2 = [
            float(np.linalg.norm(results[swap]["v_t"][s] - results[base]["v_t"][s]))
            for s in range(results[base]["v_t"].shape[0])
        ]
        key = f"{swap}_vs_{base}"
        summary["comparisons"][key] = {
            "final_action_l2": a_l2,
            "final_action_rel_l2": rel,
            "v_t_l2_per_step": vt_l2,
            "tac_mass_total_base": m_base.sum(axis=1).tolist(),
            "tac_mass_total_swap": m_swap.sum(axis=1).tolist(),
            "tac_mass_delta_per_step_layer": delta.tolist(),
        }
        for col, (data, title, cmap) in enumerate(
            ((m_base, f"{base} mass", "viridis"), (m_swap, f"{swap} mass", "viridis"),
             (delta, f"delta (swap-orig)", "RdBu_r"))
        ):
            ax = axes[row, col]
            vlim = np.abs(delta).max() if cmap == "RdBu_r" else None
            im = ax.imshow(data, aspect="auto", cmap=cmap,
                           vmin=-vlim if vlim else None, vmax=vlim if vlim else None)
            ax.set_title(title)
            ax.set_xlabel("layer")
            ax.set_ylabel("denoise step")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Action->Tactile attention mass: water ep{water_ref[0]}f{water_ref[1]} <-> dry ep{dry_ref[0]}f{dry_ref[1]}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "swap_comparison.png", dpi=110)
    plt.close(fig)

    # Per-head view of where the swap-induced change concentrates.
    for base, swap in (("water_orig", "water_tac_dry"), ("dry_orig", "dry_tac_water")):
        hd = head_mass(results[swap]["tac"]) - head_mass(results[base]["tac"])  # [steps, depth, heads]
        flat = hd.reshape(hd.shape[0], -1)
        top = np.unravel_index(np.argsort(np.abs(flat), axis=1)[:, -3:], hd.shape[1:])
        summary["comparisons"][f"{swap}_vs_{base}"]["top_delta_heads_per_step"] = [
            [[int(l), int(h), float(hd[s, l, h])] for l, h in zip(top[0][s][::-1], top[1][s][::-1])]
            for s in range(hd.shape[0])
        ]

    # Cross-observation comparisons. With mask_act_to_prefix the ACT queries only
    # see suffix keys, and given identical noise/time the sole suffix difference
    # between conditions is the 4 tactile tokens -- so water_orig-vs-dry_orig
    # then isolates the tactile-driven action difference (metric A). Pairs that
    # share the same tactile images are controls: they must be ~0 when masked,
    # and quantify the scene/state/prompt contribution when unmasked.
    def traj_diff(a: str, b: str) -> dict:
        fa, fb = results[a]["final"], results[b]["final"]
        va, vb = results[a]["v_t"].astype(np.float32), results[b]["v_t"].astype(np.float32)
        l2 = float(np.linalg.norm(fa - fb))
        denom = 0.5 * (float(np.linalg.norm(fa)) + float(np.linalg.norm(fb)))
        vt_l2 = np.linalg.norm(va - vb, axis=(1, 2))
        vt_ref = 0.5 * (np.linalg.norm(va, axis=(1, 2)) + np.linalg.norm(vb, axis=(1, 2)))
        return {
            "final_action_l2": l2,
            "final_action_rel_l2": l2 / max(denom, 1e-12),
            "v_t_l2_per_step": vt_l2.tolist(),
            "v_t_rel_l2_per_step": (vt_l2 / np.maximum(vt_ref, 1e-12)).tolist(),
        }

    summary["cross_obs_comparisons"] = {
        # different tactile, different prefix (= metric A when masked)
        "water_orig_vs_dry_orig": traj_diff("water_orig", "dry_orig"),
        # control: same dry tactile, different prefix (~0 when masked)
        "water_tac_dry_vs_dry_orig": traj_diff("water_tac_dry", "dry_orig"),
        # control: same water tactile, different prefix (~0 when masked)
        "water_orig_vs_dry_tac_water": traj_diff("water_orig", "dry_tac_water"),
        # different tactile, swapped prefixes (~A when masked)
        "water_tac_dry_vs_dry_tac_water": traj_diff("water_tac_dry", "dry_tac_water"),
    }
    return summary


if __name__ == "__main__":
    main()
