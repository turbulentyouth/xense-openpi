"""Profile the v1.1 GRU refiner's correction delta on the val set (no training).

Decomposes Delta a = a_new - a_vla on all 5,870 val frames:

Part 1 (in-segment, pure npz/pkl, no dataset):
1. |Delta| per chunk step (mean/p90 over val frames); buckets steps 0-4, 0-9
   (RTC execution window) vs all 50 steps; padding dims 20-31 sanity check.
2. |Delta| per action dim (20 valid dims): dim 9 (right_tcp.x) share of total
   |Delta|, top-3 non-direction dims.
3. Dangerous frames: fraction with any dim |Delta| > 0.5 (normalised units)
   within the first 5 steps, overall and by in_window.
4. aux water-probability distribution (for comparison with Part 2).

Part 2 (out-of-segment, label=-1; needs the dataset + frozen FastViT):
~200 random out-of-segment val frames (fixed seed). Tactile history [t-3..t]
encoded on the fly by the frozen FastViT (encoder only, no VLA inference);
a_vla is BORROWED from the temporally nearest in-segment val frame of the same
episode (approximation, stated in the report). Reports per-frame ||Delta|| L2
and aux prob1 vs the in-segment distributions.

Output: outputs/refiner_v11/delta_profile.json

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/delta_profile.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from test.tactile_counterfactual import train_refiner_v11 as tr11  # noqa: E402

logger = logging.getLogger("delta_profile")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"
TAC_HIST = REPO_ROOT / "outputs" / "refiner_data" / "tac_history.npz"
PARAMS = REPO_ROOT / "outputs" / "refiner_v11" / "refiner_params.pkl"
ACCEPT = REPO_ROOT / "outputs" / "refiner_v11" / "acceptance.json"
AUX_LABELS = REPO_ROOT / "outputs" / "stage_a" / "aux_labels.npz"
OUT = REPO_ROOT / "outputs" / "refiner_v11" / "delta_profile.json"

CONFIG_NAME = "pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100"
CKPT_DIR = REPO_ROOT / "checkpoints" / "ckpt59999_bf16"
FASTVIT = REPO_ROOT / "checkpoints" / "params.safetensors"

VALID_DIMS = 20
DANGER_THRESHOLD = 0.5
DANGER_STEPS = 5
OOD_N = 200
OOD_SEED = 0
_OBS_KEYS = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")


def load_refiner():
    with open(PARAMS, "rb") as f:
        pkl = pickle.load(f)
    assert pkl["config"]["history"] == "gru", pkl["config"]
    model = tr11.RefinerV11(rngs=nnx.Rngs(0), history="gru")
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pkl["state"])
    eval_fn = tr11.make_eval_fn(graphdef)
    return pkl, eval_fn, state


def prob1_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))


def summarize_probs(p: np.ndarray) -> dict:
    return {
        "mean": float(p.mean()),
        "p10": float(np.percentile(p, 10)),
        "p50": float(np.percentile(p, 50)),
        "p90": float(np.percentile(p, 90)),
        "frac_uncertain_0.1_0.9": float(((p > 0.1) & (p < 0.9)).mean()),
    }


def part1() -> dict:
    z = np.load(DATA)
    zh = np.load(TAC_HIST)
    pkl, eval_fn, state = load_refiner()
    val_eps = set(json.loads(ACCEPT.read_text())["val_episodes"])

    episode = z["episode"].astype(np.int64)
    label = z["label"].astype(np.int64)
    is_val = np.asarray([ep in val_eps for ep in episode])
    n_val = int(is_val.sum())
    tac_hist = (zh["tac_hist"].astype(np.float32) - pkl["tac_mean"]) / pkl["tac_std"]
    a_vla = z["a_vla"].astype(np.float32).reshape(len(label), -1)
    logger.info("part1: val frames %d; running refiner", n_val)

    a_new, logits = tr11.batched_predict(eval_fn, state, tac_hist[is_val], a_vla[is_val])
    delta = (a_new - a_vla[is_val]).reshape(n_val, 50, 32)
    abs_d = np.abs(delta)
    in_window = z["in_window"][is_val].astype(bool)
    prob1 = prob1_from_logits(logits)

    # Padding dims sanity: refiner should not touch dims 20-31.
    pad = abs_d[:, :, VALID_DIMS:]
    valid = abs_d[:, :, :VALID_DIMS]

    # 1. per-step curve (mean over valid dims first, then over frames)
    per_step_mean = valid.mean(axis=(0, 2))  # [50]
    per_step_p90 = np.percentile(valid.mean(axis=2), 90, axis=0)
    curve_points = {int(s): {"mean": float(per_step_mean[s]), "p90": float(per_step_p90[s])} for s in [0, 1, 2, 4, 9, 24, 49]}

    # 2. per-dim profile
    per_dim = valid.mean(axis=(0, 1))  # [20]
    total = float(per_dim.sum())
    non9 = [(int(d), float(per_dim[d])) for d in range(VALID_DIMS) if d != 9]
    non9.sort(key=lambda x: -x[1])

    # 3. dangerous frames
    early_max = valid[:, :DANGER_STEPS, :].max(axis=(1, 2))  # [n_val]
    danger = early_max > DANGER_THRESHOLD

    # per-frame L2 over (steps x valid dims), for the Part 2 comparison
    frame_l2 = np.sqrt((valid**2).sum(axis=(1, 2)))

    out = {
        "val_frames": n_val,
        "in_window_frames": int(in_window.sum()),
        "padding_dims_20_31": {
            "abs_delta_mean": float(pad.mean()),
            "abs_delta_max": float(pad.max()),
        },
        "per_step_abs_delta": {
            "mean_by_step": [float(x) for x in per_step_mean],
            "p90_by_step": [float(x) for x in per_step_p90],
            "selected_steps": curve_points,
            "bucket_steps_0_4_mean": float(per_step_mean[:5].mean()),
            "bucket_steps_0_9_mean": float(per_step_mean[:10].mean()),
            "bucket_all_50_mean": float(per_step_mean.mean()),
            "bucket_steps_0_4_p90_mean": float(per_step_p90[:5].mean()),
            "bucket_steps_0_9_p90_mean": float(per_step_p90[:10].mean()),
            "bucket_all_50_p90_mean": float(per_step_p90.mean()),
        },
        "per_dim_abs_delta": {
            "mean_abs_by_dim": [float(x) for x in per_dim],
            "dim9_share_of_total_valid": float(per_dim[9] / total),
            "dim9_mean_abs": float(per_dim[9]),
            "top3_non_dim9": [{"dim": d, "mean_abs": v, "share": v / total} for d, v in non9[:3]],
        },
        "dangerous_frames": {
            "definition": f"any valid dim |Delta| > {DANGER_THRESHOLD} within first {DANGER_STEPS} steps (normalised units)",
            "frac_all": float(danger.mean()),
            "frac_in_window": float(danger[in_window].mean()),
            "frac_out_window": float(danger[~in_window].mean()),
            "early_max_abs_p50": float(np.percentile(early_max, 50)),
            "early_max_abs_p90": float(np.percentile(early_max, 90)),
            "early_max_abs_p99": float(np.percentile(early_max, 99)),
        },
        "frame_delta_l2": {
            "mean": float(frame_l2.mean()),
            "p50": float(np.percentile(frame_l2, 50)),
            "p90": float(np.percentile(frame_l2, 90)),
        },
        "aux_prob1_in_segment": summarize_probs(prob1),
    }
    logger.info(
        "part1: steps0-4 mean %.4f / steps0-9 %.4f / all50 %.4f; dim9 share %.3f; danger all %.4f (in-win %.4f / out-win %.4f)",
        out["per_step_abs_delta"]["bucket_steps_0_4_mean"],
        out["per_step_abs_delta"]["bucket_steps_0_9_mean"],
        out["per_step_abs_delta"]["bucket_all_50_mean"],
        out["per_dim_abs_delta"]["dim9_share_of_total_valid"],
        out["dangerous_frames"]["frac_all"],
        out["dangerous_frames"]["frac_in_window"],
        out["dangerous_frames"]["frac_out_window"],
    )
    return out


def part2(in_seg_frame_l2: dict, in_seg_prob: dict) -> dict:
    from openpi.models.tactile_encoders import build_tactile_encoder
    from test.tactile_counterfactual import build_refiner_data as brd
    from test.tactile_counterfactual import layer_probe as lp

    z = np.load(DATA)
    pkl, eval_fn, state = load_refiner()
    val_eps = sorted(json.loads(ACCEPT.read_text())["val_episodes"])

    episode = z["episode"].astype(np.int64)
    frame = z["frame"].astype(np.int64)
    a_vla = z["a_vla"].astype(np.float32).reshape(len(frame), -1)
    is_val = np.asarray([ep in val_eps for ep in episode])

    # Out-of-segment candidate frames in val episodes.
    zl = np.load(AUX_LABELS)
    cand = []  # (episode, frame)
    for ep in val_eps:
        key = f"ep{ep}"
        if key not in zl:
            continue
        lab = zl[key]
        for fr in np.nonzero(lab == -1)[0].tolist():
            cand.append((ep, fr))
    rng = np.random.default_rng(OOD_SEED)
    picks = rng.choice(len(cand), size=min(OOD_N, len(cand)), replace=False)
    ood = [cand[int(i)] for i in picks]
    logger.info("part2: %d out-of-segment candidates in val episodes; sampled %d", len(cand), len(ood))

    # Borrow a_vla from the temporally nearest in-segment val frame of the same episode.
    val_index = {}
    for i in np.nonzero(is_val)[0].tolist():
        val_index.setdefault(int(episode[i]), []).append(i)
    borrow = []
    for ep, fr in ood:
        rows = val_index[ep]
        i = min(rows, key=lambda j: abs(int(frame[j]) - fr))
        borrow.append((i, abs(int(frame[i]) - fr)))
    a_vla_ood = np.stack([a_vla[i] for i, _ in borrow])
    borrow_dist = np.asarray([d for _, d in borrow])
    logger.info(
        "part2: borrowed a_vla frame distance mean %.1f / p90 %.1f / max %d",
        borrow_dist.mean(),
        np.percentile(borrow_dist, 90),
        borrow_dist.max(),
    )

    # Encode tactile history windows [t-3..t] with the frozen FastViT.
    setup = lp.load_setup(CONFIG_NAME, CKPT_DIR)
    encoder = build_tactile_encoder("fastvit_t12", rngs=nnx.Rngs(0), pretrained_path=str(FASTVIT))
    prep_fn = brd.make_prep_fn()
    encode_pack = brd.make_encode_fn(encoder)
    encode_fn, estate = encode_pack
    tac_keys = tuple(setup.model_config.tactile_image_keys)

    n = len(ood)
    tac_hist = np.empty((n, 4, 4, 1024), dtype=np.float32)  # [sample, time, view, feat]
    B = 32
    for i0 in range(0, n, B):
        chunk = ood[i0 : i0 + B]
        for t_back in range(3, -1, -1):  # t-3 .. t
            samples = [setup.dataset.get_sample(ep, max(fr - t_back, 0)) for ep, fr in chunk]
            batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)
            obs = lp.observation_from_batch({k: batch[k] for k in _OBS_KEYS if k in batch})
            po = prep_fn(obs)
            imgs = jnp.stack([po.images[k] for k in tac_keys], axis=1)  # [B, 4, 224, 224, 3]
            b_, nv, h, w, c = imgs.shape
            feats = np.asarray(encode_fn(estate, imgs.reshape(b_ * nv, h, w, c)), dtype=np.float32)
            tac_hist[i0 : i0 + b_, 3 - t_back] = feats.reshape(b_, nv, -1)
        logger.info("part2: encoded %d-%d / %d", i0, min(i0 + B, n) - 1, n)

    tac_std = (tac_hist - pkl["tac_mean"]) / pkl["tac_std"]
    a_new, logits = tr11.batched_predict(eval_fn, state, tac_std, a_vla_ood)
    delta = (a_new - a_vla_ood).reshape(n, 50, 32)[:, :, :VALID_DIMS]
    frame_l2 = np.sqrt((delta**2).sum(axis=(1, 2)))
    prob1 = prob1_from_logits(logits)

    out = {
        "approximation": (
            "out-of-segment frames are NOT in refiner_data; tac history is the REAL out-of-segment "
            "tactile signal encoded on the fly, but a_vla is BORROWED from the temporally nearest "
            "in-segment val frame of the same episode (frame-distance stats below)."
        ),
        "n_candidates": len(cand),
        "n_sampled": n,
        "seed": OOD_SEED,
        "borrowed_a_vla_frame_distance": {
            "mean": float(borrow_dist.mean()),
            "p50": float(np.percentile(borrow_dist, 50)),
            "p90": float(np.percentile(borrow_dist, 90)),
            "max": int(borrow_dist.max()),
        },
        "frame_delta_l2_out_of_segment": {
            "mean": float(frame_l2.mean()),
            "p50": float(np.percentile(frame_l2, 50)),
            "p90": float(np.percentile(frame_l2, 90)),
        },
        "frame_delta_l2_in_segment_ref": in_seg_frame_l2,
        "aux_prob1_out_of_segment": summarize_probs(prob1),
        "aux_prob1_in_segment_ref": in_seg_prob,
        "early5_danger_frac_out_of_segment": float((np.abs(delta)[:, :DANGER_STEPS, :].max(axis=(1, 2)) > DANGER_THRESHOLD).mean()),
    }
    logger.info(
        "part2: ood ||Delta|| mean %.3f vs in-seg %.3f; ood prob1 mean %.3f / uncertain %.3f vs in-seg %.3f / %.3f",
        out["frame_delta_l2_out_of_segment"]["mean"],
        in_seg_frame_l2["mean"],
        out["aux_prob1_out_of_segment"]["mean"],
        out["aux_prob1_out_of_segment"]["frac_uncertain_0.1_0.9"],
        in_seg_prob["mean"],
        in_seg_prob["frac_uncertain_0.1_0.9"],
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    p1 = part1()
    p2 = part2(p1["frame_delta_l2"], p1["aux_prob1_in_segment"])
    out = {
        "model": "refiner v1.1 GRU (outputs/refiner_v11/refiner_params.pkl), frozen, no training",
        "action_space": "normalised delta-cartesian; dim 9 = right_tcp.x; valid dims 0-19; 20-31 padding",
        "part1_in_segment": p1,
        "part2_out_of_segment": p2,
    }
    OUT.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
