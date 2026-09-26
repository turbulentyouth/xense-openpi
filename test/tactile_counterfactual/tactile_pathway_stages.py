"""Where does the tactile signal die? Stage-by-stage pathway comparison.

Compares condition A (a frame as-is) against condition B (the SAME frame with
its 4 tactile images counterfactually swapped from a donor frame -- scene,
state, prompt, noise all identical), measuring the relative representation
change at every stage of the tactile pathway:

  0  raw tactile images (after observation preprocessing)
  1  FastViT encoder features
  2  tactile_proj output = the TAC tokens that enter the suffix
  3  per-layer K / V at the TAC positions (first denoise step, t=1)
  4  per-layer ACT->TAC attention probabilities and per-key ||W_o v||
  5  suffix final-layer output, TAC part and ACT part
  6  v_t at t=1 and the final action of the full 10-step rollout

A third condition C (another frame of the same episode) gives the natural
variation scale for the tactile-only stages 0-2: if the A<->B encoder
difference is comparable to A<->C, the encoder preserves tactile content and
the attenuation must happen downstream.

Example:
    python test/tactile_counterfactual/tactile_pathway_stages.py \
        --checkpoint-dir checkpoints/ckpt59999_bf16 \
        --frame 1:316 --donor 21:244 --ref-frame 1:200 \
        --out ~/tactile_pathway_ep1f316
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

from openpi.models import pi0 as _pi0  # noqa: E402
from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("tactile_pathway_stages")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    p.add_argument("--frame", required=True, metavar="EP:FRAME", help="condition A")
    p.add_argument("--donor", required=True, metavar="EP:FRAME", help="tactile donor for condition B")
    p.add_argument("--ref-frame", default=None, metavar="EP:FRAME", help="condition C (natural variation scale)")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric relative L2: ||a-b|| / mean(||a||, ||b||). 0 = identical, ~1.41 = orthogonal."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    denom = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / max(denom, 1e-20))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20))


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {"rel_l2": rel_l2(a, b), "cosine": cos_sim(a, b)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from test.tactile_counterfactual.run_tactile_swap import swap_tactile

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    model, mc = setup.model, setup.model_config
    keys = tuple(mc.tactile_image_keys)
    ah, ad = mc.action_horizon, mc.action_dim

    ep, fr = (int(x) for x in args.frame.split(":"))
    d_ep, d_fr = (int(x) for x in args.donor.split(":"))
    sample_a = setup.dataset.get_sample(ep, fr)
    donor = setup.dataset.get_sample(d_ep, d_fr)
    samples = {"A": sample_a, "B": swap_tactile(sample_a, donor, keys)}
    if args.ref_frame:
        c_ep, c_fr = (int(x) for x in args.ref_frame.split(":"))
        samples["C"] = setup.dataset.get_sample(c_ep, c_fr)

    # Preprocess once per condition, exactly as sample_actions does (rng=None,
    # train=False is deterministic), for the eager stages 0-2.
    proc = {
        name: model._preprocess_observation(None, setup.dataset.observation_from_sample(s), train=False)
        for name, s in samples.items()
    }

    # ---- Stages 0-2: images -> FastViT -> projection (eager, tactile-only) ----
    stages: dict[str, dict[str, np.ndarray]] = {"images": {}, "fastvit": {}, "tokens": {}}
    for name, obs in proc.items():
        imgs = jnp.concatenate([obs.images[k] for k in keys], axis=0)  # [4, h, w, 3] (b=1 folded into the 4 views)
        feats = model.tactile_encoder(imgs)  # [4, feat]
        toks = model.tactile_proj(feats)  # [4, width]
        stages["images"][name] = np.asarray(imgs, np.float32)
        stages["fastvit"][name] = np.asarray(feats, np.float32)
        stages["tokens"][name] = np.asarray(toks, np.float32)
    logger.info(
        "stage shapes: images %s, fastvit %s, tokens %s",
        stages["images"]["A"].shape,
        stages["fastvit"]["A"].shape,
        stages["tokens"]["A"].shape,
    )

    # ---- Stages 3-6: inside Gemma (jitted, mirrors AttentionSampler at t=1) ----
    graphdef, state = nnx.split(model)
    noise = lp.fixed_noise(args.seed, 1, ah, ad)

    def internals(state, obs, x_t):
        module = nnx.merge(graphdef, state)
        obs = module._preprocess_observation(None, obs, train=False)
        batch = obs.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar = module.embed_prefix(obs)
        prefix_attn = _pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = module.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn, positions=positions)

        suffix_tokens, suffix_mask, suffix_ar, adarms = module.embed_suffix(
            obs, x_t, jnp.broadcast_to(jnp.asarray(1.0, jnp.float32), batch)
        )
        suffix_attn = _pi0.make_attn_mask(suffix_mask, suffix_ar)
        import einops

        prefix_attn_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_mask = jnp.concatenate([prefix_attn_, suffix_attn], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), kv_full, attn, vnorms = module.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms],
            return_attention=True,
        )
        v_t = module.action_out_proj(suffix_out[:, -module.action_horizon :])
        return suffix_out, kv_full, attn, vnorms[0], v_t

    jit_internals = jax.jit(internals)
    raw_obs = {n: setup.dataset.observation_from_sample(s) for n, s in samples.items() if n in ("A", "B")}
    inner: dict[str, dict] = {}
    for name, obs in raw_obs.items():
        suffix_out, kv_full, attn, vnorm, v_t = jit_internals(state, obs, jnp.asarray(noise))
        k_all, v_all = kv_full  # KVCache: layer-stacked [depth, B, S, kvH, hd]
        inner[name] = {
            "suffix_out": np.asarray(suffix_out, np.float32),
            "k": np.asarray(k_all, np.float32),
            "v": np.asarray(v_all, np.float32),
            "attn": np.asarray(attn, np.float32),  # [depth, B, H, T, S]
            "vnorm": np.asarray(vnorm, np.float32),  # [depth, B, H, S]
            "v_t": np.asarray(v_t, np.float32),
        }
    k_len = inner["A"]["attn"].shape[-1]
    suffix_len = inner["A"]["attn"].shape[-2]
    p_len = k_len - suffix_len
    n_tac = len(keys)
    logger.info("prefix_len=%d, TAC positions %d..%d, ACT positions %d..%d", p_len, p_len, p_len + n_tac - 1, p_len + n_tac, k_len - 1)

    # Final actions of the full rollout for A and B (same noise, same steps).
    sampler = lp.AttentionSampler(model)
    final = {n: np.asarray(sampler(obs, noise=noise, num_steps=args.num_steps).final_action, np.float32) for n, obs in raw_obs.items()}

    # ---- Metrics ----
    tac = slice(p_len, p_len + n_tac)
    act_q = slice(-ah, None)
    report: dict = {
        "frames": {"A": args.frame, "B": f"{args.frame} with tactile from {args.donor}", "C": args.ref_frame},
        "prefix_len": p_len,
        "num_tactile": n_tac,
        "action_horizon": ah,
        "metric_defs": {"rel_l2": "||a-b|| / mean(||a||,||b||); 0 identical, ~1.41 orthogonal", "cosine": "flattened cosine similarity"},
        "tactile_only_stages": {},
        "gemma_stages": {},
    }

    for stage in ("images", "fastvit", "tokens"):
        entry = {"A_vs_B": compare(stages[stage]["A"], stages[stage]["B"])}
        if "C" in stages[stage]:
            entry["A_vs_C"] = compare(stages[stage]["A"], stages[stage]["C"])
        report["tactile_only_stages"][stage] = entry

    depth = inner["A"]["k"].shape[0]
    k_rel = [rel_l2(inner["A"]["k"][l][:, tac], inner["B"]["k"][l][:, tac]) for l in range(depth)]
    v_rel = [rel_l2(inner["A"]["v"][l][:, tac], inner["B"]["v"][l][:, tac]) for l in range(depth)]
    vn_rel = [rel_l2(inner["A"]["vnorm"][l][..., tac], inner["B"]["vnorm"][l][..., tac]) for l in range(depth)]
    # ACT->TAC attention probability mass per (layer, head), then its relative change.
    mass = {
        n: inner[n]["attn"][:, 0, :, act_q, tac].sum(axis=-1).mean(axis=-1)  # [depth, H]
        for n in ("A", "B")
    }
    mass_rel_change = np.abs(mass["B"] - mass["A"]) / np.maximum(mass["A"], 1e-20)  # [depth, H]
    prob_max_abs_diff = [
        float(np.abs(inner["A"]["attn"][l][0, :, act_q, tac] - inner["B"]["attn"][l][0, :, act_q, tac]).max())
        for l in range(depth)
    ]
    report["gemma_stages"] = {
        "k_tac_rel_l2_per_layer": k_rel,
        "v_tac_rel_l2_per_layer": v_rel,
        "vnorm_tac_rel_l2_per_layer": vn_rel,
        "act_to_tac_prob_mass_A": mass["A"].mean(axis=-1).tolist(),
        "act_to_tac_prob_mass_rel_change_max_per_layer": mass_rel_change.max(axis=-1).tolist(),
        "act_to_tac_prob_max_abs_diff_per_layer": prob_max_abs_diff,
        "suffix_out_tac": compare(inner["A"]["suffix_out"][:, :n_tac], inner["B"]["suffix_out"][:, :n_tac]),
        "suffix_out_act": compare(inner["A"]["suffix_out"][:, -ah:], inner["B"]["suffix_out"][:, -ah:]),
        "v_t_t1": compare(inner["A"]["v_t"], inner["B"]["v_t"]),
        "final_action": compare(final["A"], final["B"]),
    }

    (out_dir / "pathway_report.json").write_text(json.dumps(lp.to_jsonable(report), indent=1, ensure_ascii=False))

    # ---- Figure ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))
    chain = [
        ("0 raw images", report["tactile_only_stages"]["images"]["A_vs_B"]["rel_l2"]),
        ("1 FastViT feat", report["tactile_only_stages"]["fastvit"]["A_vs_B"]["rel_l2"]),
        ("2 TAC tokens", report["tactile_only_stages"]["tokens"]["A_vs_B"]["rel_l2"]),
        ("3a K@TAC (L-mean)", float(np.mean(k_rel))),
        ("3b V@TAC (L-mean)", float(np.mean(v_rel))),
        ("5a suffix_out TAC", report["gemma_stages"]["suffix_out_tac"]["rel_l2"]),
        ("5b suffix_out ACT", report["gemma_stages"]["suffix_out_act"]["rel_l2"]),
        ("6a v_t (t=1)", report["gemma_stages"]["v_t_t1"]["rel_l2"]),
        ("6b final action", report["gemma_stages"]["final_action"]["rel_l2"]),
    ]
    names = [c[0] for c in chain]
    vals = [c[1] for c in chain]
    ax0.bar(range(len(chain)), vals, color=["#4C72B0"] * 3 + ["#DD8452"] * 2 + ["#55A868"] * 2 + ["#C44E52"] * 2)
    if "C" in stages["images"]:
        ref = [
            report["tactile_only_stages"]["images"]["A_vs_C"]["rel_l2"],
            report["tactile_only_stages"]["fastvit"]["A_vs_C"]["rel_l2"],
            report["tactile_only_stages"]["tokens"]["A_vs_C"]["rel_l2"],
        ]
        ax0.plot(range(3), ref, "ko--", label="A vs C (natural frame variation)")
        ax0.legend()
    ax0.set_xticks(range(len(chain)), names, rotation=30, ha="right")
    ax0.set_yscale("log")
    ax0.set_ylabel("relative L2 change (log scale)")
    ax0.set_title(f"Tactile swap (A=ep{ep}f{fr}, donor=ep{d_ep}f{d_fr}): where does the difference die?")
    ax0.grid(axis="y", alpha=0.3)

    layers = np.arange(depth)
    ax1.plot(layers, k_rel, "o-", label="K @ TAC rel_l2")
    ax1.plot(layers, v_rel, "s-", label="V @ TAC rel_l2")
    ax1.plot(layers, vn_rel, "^-", label="||W_o v|| @ TAC rel_l2")
    ax1.plot(layers, mass_rel_change.max(axis=-1), "d-", label="ACT->TAC prob mass rel change (max over heads)")
    ax1.set_yscale("log")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("relative change (log scale)")
    ax1.set_title("Per-layer change at TAC positions, A vs B")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pathway_stages.png", dpi=110)

    logger.info("wrote %s", out_dir / "pathway_report.json")
    logger.info("=== tactile-only stages (A vs B | A vs C) ===")
    for stage, e in report["tactile_only_stages"].items():
        msg = f"  {stage:8s} rel_l2 {e['A_vs_B']['rel_l2']:.4f} cos {e['A_vs_B']['cosine']:.4f}"
        if "A_vs_C" in e:
            msg += f"   | natural: rel_l2 {e['A_vs_C']['rel_l2']:.4f} cos {e['A_vs_C']['cosine']:.4f}"
        logger.info(msg)
    logger.info("=== gemma stages (A vs B) ===")
    logger.info("  K@TAC rel_l2 per layer: %s", " ".join(f"{x:.3f}" for x in k_rel))
    logger.info("  V@TAC rel_l2 per layer: %s", " ".join(f"{x:.3f}" for x in v_rel))
    logger.info("  suffix_out TAC: %s", report["gemma_stages"]["suffix_out_tac"])
    logger.info("  suffix_out ACT: %s", report["gemma_stages"]["suffix_out_act"])
    logger.info("  v_t(t=1): %s", report["gemma_stages"]["v_t_t1"])
    logger.info("  final action: %s", report["gemma_stages"]["final_action"])


if __name__ == "__main__":
    main()
