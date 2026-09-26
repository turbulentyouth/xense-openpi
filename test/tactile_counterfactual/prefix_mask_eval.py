"""Prefix-masking counterfactual: how much does the ACT query depend on the VLM prefix?

Three measurements on one observation (default episode 0, frame 316, the water
frame used by the swap experiments) with fixed noise, all through
``layer_probe.AttentionSampler``:

1. Normalisation arithmetic (exact, single normal trajectory): for ACT queries,
   the softmax mass on prefix keys (m_prefix), tactile keys (m_tac) and action
   keys (m_act) per (step, layer, head). Masking ACT->prefix only changes the
   softmax denominator, so the exact single-forward estimate of the masked
   tactile mass is m_tac / (1 - m_prefix), and the normalisation amplification
   is 1 / (1 - m_prefix).

2. A real masked trajectory (``mask_act_to_prefix=True``): actual masked m_tac
   vs the estimate from (1) -- step 0 / layer 0 must match exactly (same x_t,
   same KV cache, same logits; only the softmax subset differs), later layers
   drift as hidden states cascade. Plus the masked-vs-normal final action
   rel-L2 (= total prefix dependence of the current policy) and where the freed
   attention mass goes (TAC vs ACT keys).

3. Prefix panorama: ACT-query mass on each prefix key segment (3 camera views x
   256 SigLIP tokens + prompt segment), aggregated per layer.

Outputs under ``--out``: ``act_attention_{normal,masked}.npz`` (ACT-query rows of
the full attention, float16), ``final_{normal,masked}.npy``, ``summary.json``
and two PNGs.

Example:
    python test/tactile_counterfactual/prefix_mask_eval.py \
        --checkpoint-dir checkpoints/ckpt59999_bf16 --out outputs/prefix_mask_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# NOTE: do NOT set XLA_FLAGS=--xla_gpu_autotune_level=0 (wrong bf16 conv results
# on this machine, see run_tactile_swap.py).

from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("prefix_mask_eval")

_IMAGE_TOKENS_PER_VIEW = 256
_IMAGE_VIEWS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    p.add_argument("--frame", default="0:316", metavar="EP:FRAME", help="observation frame (water episode)")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/prefix_mask_eval")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    mc = setup.model_config
    ah, ad = mc.action_horizon, mc.action_dim
    ep, fr = (int(x) for x in args.frame.split(":"))
    sample = setup.dataset.get_sample(ep, fr)
    obs = setup.dataset.observation_from_sample(sample)
    noise = lp.fixed_noise(args.seed, 1, ah, ad)
    sampler = lp.AttentionSampler(setup.model)

    logger.info("running normal trajectory (ep %d frame %d, seed %d, %d steps)", ep, fr, args.seed, args.num_steps)
    normal = sampler(obs, noise=noise, num_steps=args.num_steps)
    logger.info("running masked trajectory (ACT queries blocked from prefix keys)")
    masked = sampler(obs, noise=noise, num_steps=args.num_steps, mask_act_to_prefix=True)

    p_len, n_tac = normal.prefix_len, normal.num_tactile
    assert masked.prefix_len == p_len and masked.num_tactile == n_tac
    attn_n = np.asarray(normal.attention)
    attn_m = np.asarray(masked.attention)
    final_n = np.asarray(normal.final_action[0], dtype=np.float32)
    final_m = np.asarray(masked.final_action[0], dtype=np.float32)

    # Save the ACT-query rows of both attentions (float16) for later re-analysis,
    # plus float64 per-(step, layer, B, head, query) block masses (exact, no
    # float16 rounding near m_prefix == 1).
    np.savez_compressed(
        out_dir / "act_attention_normal.npz",
        act_attn=attn_n[..., -ah:, :].astype(np.float16),
        prefix_len=p_len, num_tactile=n_tac, action_horizon=ah,
    )
    np.savez_compressed(
        out_dir / "act_attention_masked.npz",
        act_attn=attn_m[..., -ah:, :].astype(np.float16),
        prefix_len=p_len, num_tactile=n_tac, action_horizon=ah,
    )

    def block_masses(attn: np.ndarray) -> dict[str, np.ndarray]:
        act = attn[..., -ah:, :].astype(np.float64)  # [steps, depth, B, heads, ah, K]
        return {
            "prefix": act[..., :p_len].sum(axis=-1),
            "tac": act[..., p_len : p_len + n_tac].sum(axis=-1),
            "act": act[..., p_len + n_tac :].sum(axis=-1),
        }

    cubes_n, cubes_m = block_masses(attn_n), block_masses(attn_m)
    np.savez(out_dir / "act_mass_cubes.npz", **{f"normal_{k}": v for k, v in cubes_n.items()},
             **{f"masked_{k}": v for k, v in cubes_m.items()})
    np.save(out_dir / "final_normal.npy", final_n)
    np.save(out_dir / "final_masked.npy", final_m)
    np.save(out_dir / "v_t_normal.npy", np.asarray(normal.v_t[:, 0], dtype=np.float16))
    np.save(out_dir / "v_t_masked.npy", np.asarray(masked.v_t[:, 0], dtype=np.float16))

    mn = {k: v.mean(axis=(2, 4)) for k, v in cubes_n.items()}  # [steps, depth, heads]
    mm = {k: v.mean(axis=(2, 4)) for k, v in cubes_m.items()}
    steps, depth = mn["prefix"].shape[:2]

    summary: dict = {
        "frame": {"episode": ep, "frame": fr},
        "num_steps": args.num_steps,
        "seed": args.seed,
        "prefix_len": p_len,
        "num_tactile": n_tac,
        "action_horizon": ah,
        "depth": depth,
    }

    # ------------------------------------------------------------------ #
    # Experiment 1: normalisation arithmetic                              #
    # ------------------------------------------------------------------ #
    m_prefix, m_tac, m_act = mn["prefix"], mn["tac"], mn["act"]
    row_sum = m_prefix + m_tac + m_act
    amp = 1.0 / (1.0 - m_prefix)  # per (step, layer, head), from head-mean masses
    # Exact masked estimate, per query: renormalising the softmax over the
    # remaining (suffix) keys scales every suffix probability of THAT query by
    # 1/(1-m_prefix(q)). The per-head estimate is the mean over queries of the
    # per-query ratio -- NOT the ratio of per-head means.
    est_cube = cubes_n["tac"] / (1.0 - cubes_n["prefix"])  # [steps, depth, B, heads, ah]
    amp_cube = 1.0 / (1.0 - cubes_n["prefix"])
    tac_est = est_cube.mean(axis=(2, 4))  # [steps, depth, heads]
    exp1 = {
        # per (step, layer), mean over heads
        "m_prefix_step_layer": m_prefix.mean(axis=-1).tolist(),
        "m_tac_step_layer": m_tac.mean(axis=-1).tolist(),
        "m_act_step_layer": m_act.mean(axis=-1).tolist(),
        "row_sum_min_max": [float(row_sum.min()), float(row_sum.max())],
        "m_tac_masked_est_step_layer": tac_est.mean(axis=-1).tolist(),
        "amplification_step_layer": amp.mean(axis=-1).tolist(),
        "amplification_over_steps_layers_heads": {
            "min": float(amp.min()),
            "median": float(np.median(amp)),
            "max": float(amp.max()),
            "mean": float(amp.mean()),
        },
        "m_prefix_over_0.999_fraction": float((m_prefix > 0.999).mean()),
        "amplification_per_query_over_all": {
            "min": float(amp_cube.min()),
            "median": float(np.median(amp_cube)),
            "max": float(amp_cube.max()),
            "mean": float(amp_cube.mean()),
        },
        # per-layer and per-step aggregates (mean over the other axis and heads)
        "m_prefix_per_layer": m_prefix.mean(axis=(0, 2)).tolist(),
        "m_tac_per_layer": m_tac.mean(axis=(0, 2)).tolist(),
        "m_act_per_layer": m_act.mean(axis=(0, 2)).tolist(),
        "m_prefix_per_step": m_prefix.mean(axis=(1, 2)).tolist(),
        "m_tac_per_step": m_tac.mean(axis=(1, 2)).tolist(),
        "m_act_per_step": m_act.mean(axis=(1, 2)).tolist(),
        "amplification_per_layer": amp.mean(axis=(0, 2)).tolist(),
        "m_tac_masked_est_per_layer": tac_est.mean(axis=(0, 2)).tolist(),
        "m_tac_masked_est_per_step": tac_est.mean(axis=(1, 2)).tolist(),
        "tac_mass_total_normal_per_step": m_tac.sum(axis=(1, 2)).tolist(),
        "tac_mass_total_masked_est_per_step": tac_est.sum(axis=(1, 2)).tolist(),
    }
    summary["experiment1_normalisation"] = exp1

    # ------------------------------------------------------------------ #
    # Experiment 2: real masked forward                                   #
    # ------------------------------------------------------------------ #
    tac_masked = mm["tac"]
    est_vs_masked_step_layer = {
        "max_abs_diff": float(np.abs(tac_masked.mean(-1) - tac_est.mean(-1)).max()),
        "layer0_per_step_est": tac_est[:, 0, :].mean(axis=-1).tolist(),
        "layer0_per_step_masked": tac_masked[:, 0, :].mean(axis=-1).tolist(),
        "step0_layer0_max_abs_diff_over_heads": float(np.abs(tac_masked[0, 0] - tac_est[0, 0]).max()),
        "masked_step_layer": tac_masked.mean(axis=-1).tolist(),
        "est_step_layer": tac_est.mean(axis=-1).tolist(),
        "ratio_masked_over_est_step_layer": (tac_masked.mean(-1) / np.maximum(tac_est.mean(-1), 1e-20)).tolist(),
        "masked_per_layer": tac_masked.mean(axis=(0, 2)).tolist(),
        "est_per_layer": tac_est.mean(axis=(0, 2)).tolist(),
    }
    # Full-block exactness at (step 0, layer 0): masked ACT rows must equal the
    # normal rows restricted to suffix keys and renormalised by 1/(1-m_prefix).
    act_n00 = attn_n[0, 0, :, :, -ah:, :].astype(np.float64)  # [B, heads, ah, K]
    act_m00 = attn_m[0, 0, :, :, -ah:, :].astype(np.float64)
    renorm = act_n00[..., p_len:] / (1.0 - act_n00[..., :p_len].sum(axis=-1, keepdims=True))
    # Cross-check: per-head TAC mass at (0,0) computed two independent ways must agree.
    diag = {
        "est_from_full_block_per_head": renorm[..., :n_tac].sum(axis=-1).mean(axis=(0, 2)).tolist(),
        "masked_from_full_block_per_head": act_m00[..., p_len : p_len + n_tac].sum(axis=-1).mean(axis=(0, 2)).tolist(),
        "est_from_cube_per_head": tac_est[0, 0].tolist(),
        "masked_from_cube_per_head": tac_masked[0, 0].tolist(),
        "m_prefix_per_head_s0l0": m_prefix[0, 0].tolist(),
    }
    logger.info("s0l0 diagnostic: %s", json.dumps(diag, indent=1))
    exp2 = {
        "masked_vs_est_tac_mass": est_vs_masked_step_layer,
        "step0_layer0_full_block_max_abs_diff": float(np.abs(act_m00[..., p_len:] - renorm).max()),
        "step0_layer0_prefix_mass_residual": float(act_m00[..., :p_len].sum(axis=-1).max()),
        "final_action_rel_l2": float(np.linalg.norm(final_m - final_n) / max(np.linalg.norm(final_n), 1e-12)),
        "final_action_l2": float(np.linalg.norm(final_m - final_n)),
        "final_action_ref_norm": float(np.linalg.norm(final_n)),
        "final_normal": final_n.tolist(),
        "final_masked": final_m.tolist(),
        # Redistribution of the freed prefix mass (head-mean, per layer):
        "redistribution": {
            "normal_prefix_per_layer": m_prefix.mean(axis=(0, 2)).tolist(),
            "masked_prefix_per_layer": mm["prefix"].mean(axis=(0, 2)).tolist(),
            "normal_tac_per_layer": m_tac.mean(axis=(0, 2)).tolist(),
            "masked_tac_per_layer": tac_masked.mean(axis=(0, 2)).tolist(),
            "normal_act_per_layer": m_act.mean(axis=(0, 2)).tolist(),
            "masked_act_per_layer": mm["act"].mean(axis=(0, 2)).tolist(),
            "delta_tac_per_layer": (tac_masked - m_tac).mean(axis=(0, 2)).tolist(),
            "delta_act_per_layer": (mm["act"] - m_act).mean(axis=(0, 2)).tolist(),
        },
    }
    summary["experiment2_masked_forward"] = exp2

    # ------------------------------------------------------------------ #
    # Experiment 3: prefix segment panorama                               #
    # ------------------------------------------------------------------ #
    img_end = len(_IMAGE_VIEWS) * _IMAGE_TOKENS_PER_VIEW
    max_token_len = mc.max_token_len
    boundary_ok = p_len == img_end + max_token_len
    if not boundary_ok:
        logger.warning(
            "prefix_len=%d != 3x%d + max_token_len=%d; segment boundaries are approximate",
            p_len, _IMAGE_TOKENS_PER_VIEW, max_token_len,
        )
    segments = {
        **{f"img_{v.removesuffix('_rgb')}": (i * _IMAGE_TOKENS_PER_VIEW, (i + 1) * _IMAGE_TOKENS_PER_VIEW)
           for i, v in enumerate(_IMAGE_VIEWS)},
        "prompt": (img_end, p_len),
    }
    act_rows = attn_n[..., -ah:, :].astype(np.float64)  # [steps, depth, B, heads, ah, K]
    seg_mass = {}
    for name, (lo, hi) in segments.items():
        mass = act_rows[..., lo:hi].sum(axis=-1).mean(axis=(2, 3, 4))  # [steps, depth]
        seg_mass[name] = {
            "per_layer": mass.mean(axis=0).tolist(),
            "per_step": mass.mean(axis=1).tolist(),
            "total_share_of_act_attention": float(mass.sum() / (steps * depth)),
        }
    summary["experiment3_prefix_segments"] = {
        "boundary_assumption": {
            "image_tokens_per_view": _IMAGE_TOKENS_PER_VIEW,
            "image_views": list(_IMAGE_VIEWS),
            "max_token_len": max_token_len,
            "prefix_len_matches_assumption": boundary_ok,
        },
        "segments": seg_mass,
    }

    (out_dir / "summary.json").write_text(json.dumps(lp.to_jsonable(summary), indent=1, ensure_ascii=False))
    logger.info("wrote %s", out_dir)

    # ------------------------------------------------------------------ #
    # Plots (optional, numbers are authoritative)                         #
    # ------------------------------------------------------------------ #
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    im = axes[0].imshow(m_prefix.mean(-1), aspect="auto", cmap="viridis")
    axes[0].set_title("normal m_prefix (ACT->prefix mass)")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("denoise step")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    diff = tac_masked.mean(-1) - tac_est.mean(-1)
    vlim = max(np.abs(diff).max(), 1e-12)
    im = axes[1].imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[1].set_title("masked actual - estimate (m_tac)")
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("denoise step")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    x = np.arange(depth)
    axes[2].plot(x, m_tac.mean(axis=(0, 2)), label="normal")
    axes[2].plot(x, tac_est.mean(axis=(0, 2)), "--", label="estimate")
    axes[2].plot(x, tac_masked.mean(axis=(0, 2)), ":", label="masked actual")
    axes[2].set_title("ACT->TAC mass per layer")
    axes[2].set_xlabel("layer"); axes[2].legend()
    fig.suptitle(f"prefix mask eval ep{ep}f{fr} seed{args.seed}")
    fig.tight_layout()
    fig.savefig(out_dir / "prefix_mask_eval.png", dpi=110)
    plt.close(fig)

    # Console digest.
    logger.info("m_prefix mean %.4f | amplification min/med/max %.3f/%.3f/%.3f",
                m_prefix.mean(), amp.min(), np.median(amp), amp.max())
    logger.info("step0/layer0 exactness: max|masked - renorm| = %.3e, prefix residual %.3e",
                exp2["step0_layer0_full_block_max_abs_diff"], exp2["step0_layer0_prefix_mass_residual"])
    logger.info("final action rel_l2 (masked vs normal) = %.4f", exp2["final_action_rel_l2"])


if __name__ == "__main__":
    main()
