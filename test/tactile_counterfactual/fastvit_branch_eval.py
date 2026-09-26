"""Did 60k training steps move the FastViT tactile branch in a useful direction?

Three analyses over the tactile branch (FastViT-T12 encoder -> tactile_proj):

1. Weight-level drift (pure numpy): every FastViT parameter in the trained
   checkpoint (``params_npy/*.bf16.npy``) vs the ImageNet-pretrained safetensors
   the run initialised from. BN running mean/var are never updated during
   training (``use_running_average=True``), so they must match the safetensors
   exactly after bf16 rounding -- this validates that the safetensors file IS
   the training-time init and that every drift comparison is legal.

2. Feature-level drift + probe matrix (GPU, small batches): the 320 samples of
   ``grasp_start_frames.yaml`` are encoded by both the ImageNet-init encoder
   (built straight from the safetensors) and the trained encoder. Combined with
   init/trained ``tactile_proj`` this gives a 2x2 matrix (encoder init/trained x
   pre/post proj) of water-vs-no-water linear probe accuracies, reusing
   ``tactile_linear_probe.probe_accuracy``. Also reports the per-sample feature
   cosine (init vs trained encoder) and whether the feature displacement
   ``f_trained - f_init`` aligns with the class-discriminant direction.

3. A JSON report tying (1) and (2) together.

Trained-encoder / trained-proj features are taken from the cached
``outputs/tactile_probe_fixed/features.npz``; the script re-encodes a batch with
the trained model and checks it against the cache so sample-order alignment is
verified, not assumed.

Example:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python test/tactile_counterfactual/fastvit_branch_eval.py \
        --out outputs/fastvit_branch_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# NOTE: do NOT set --xla_gpu_autotune_level=0 here. With autotuning disabled,
# XLA's fallback bf16 conv algorithms produce numerically wrong results on this
# machine (see tactile_linear_probe.py).

logger = logging.getLogger("fastvit_branch_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    p.add_argument("--init-safetensors", default="checkpoints/params.safetensors",
                   help="ImageNet-pretrained FastViT Flax weights (the training-time init).")
    p.add_argument("--yaml", default="grasp_start_frames.yaml")
    p.add_argument("--features-cache", default="outputs/tactile_probe_fixed/features.npz",
                   help="Cached trained-model features from tactile_linear_probe.py (variants B and D).")
    p.add_argument("--proj-drift", default="checkpoints/tactile_proj_drift.npz",
                   help="npz with k_init/b_init/k_trained/b_trained of tactile_proj (tactile_proj_drift.py).")
    p.add_argument("--batch-size", type=int, default=32, help="tactile images per encoder call")
    p.add_argument("--skip-features", action="store_true", help="only run the weight-level drift analysis")
    p.add_argument("--out", default="outputs/fastvit_branch_eval")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Part 1: weight-level drift (pure numpy)                                     #
# --------------------------------------------------------------------------- #


def load_leaf(path: pathlib.Path) -> np.ndarray:
    """Load a params_npy leaf; uint16 arrays are bf16 bit patterns."""
    raw = np.load(path)
    if raw.dtype == np.uint16:
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 (round to nearest even) -> fp32, matching jnp.astype(bf16)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return (rounded << np.uint32(16)).view(np.float32)


def is_bn_stat(key: str) -> bool:
    """BN running mean/var: frozen during training (use_running_average=True)."""
    return key.endswith("/mean") or key.endswith("/var")


def group_of(key: str) -> str:
    top = key.split("/")[0]
    return "stem" if top.startswith("stem_") else top


def weight_drift(args: argparse.Namespace) -> dict:
    from safetensors.flax import load_file

    flat = load_file(str(pathlib.Path(args.init_safetensors).expanduser().resolve()))
    npy_dir = pathlib.Path(args.checkpoint_dir) / "params_npy"

    def npy_path(key: str) -> pathlib.Path:
        return npy_dir / f"params.tactile_encoder.module.{key.replace('/', '.')}.value.bf16.npy"

    missing = [k for k in flat if not npy_path(k).exists()]
    extra = sorted(
        p.name.removeprefix("params.tactile_encoder.module.").removesuffix(".value.bf16.npy").replace(".", "/")
        for p in npy_dir.glob("params.tactile_encoder.module.*.value.bf16.npy")
        if p.name.removeprefix("params.tactile_encoder.module.").removesuffix(".value.bf16.npy").replace(".", "/")
        not in flat
    )
    if missing or extra:
        raise RuntimeError(f"key mapping broken: {len(missing)} missing npy, {len(extra)} unmatched npy; "
                           f"first missing={missing[:3]} first extra={extra[:3]}")
    logger.info("key mapping verified: %d safetensors keys <-> %d npy leaves", len(flat), len(flat))

    # BN frozen-stat validation: safetensors must equal the checkpoint after
    # bf16 rounding. If not, the safetensors file is NOT the training-time init.
    bn_keys = sorted(k for k in flat if is_bn_stat(k))
    bn_mismatch = []
    for k in bn_keys:
        trained = load_leaf(npy_path(k))
        expected = to_bf16_rne(np.asarray(flat[k], dtype=np.float32))
        if not np.array_equal(trained, expected):
            bn_mismatch.append((k, float(np.abs(trained - expected).max())))
    logger.info("BN frozen stats: %d/%d bitwise identical after bf16 rounding", len(bn_keys) - len(bn_mismatch), len(bn_keys))
    if bn_mismatch:
        raise RuntimeError(
            f"BN mean/var mismatch on {len(bn_mismatch)}/{len(bn_keys)} keys (e.g. {bn_mismatch[:3]}); "
            "the safetensors file is NOT the training-time init -- stop and investigate."
        )

    per_param = []
    sum_d2 = sum_i2 = sum_dot = sum_t2 = 0.0
    for k in sorted(flat):
        init = np.asarray(flat[k], dtype=np.float64)
        trained = load_leaf(npy_path(k)).astype(np.float64)
        if init.shape != trained.shape:
            raise RuntimeError(f"shape mismatch at {k}: {init.shape} vs {trained.shape}")
        d = trained - init
        ni, nd = np.linalg.norm(init), np.linalg.norm(d)
        nt = np.linalg.norm(trained)
        rel = float(nd / ni) if ni > 0 else float("nan")
        cos = float(np.sum(init * trained) / (ni * nt)) if ni > 0 and nt > 0 else float("nan")
        per_param.append({
            "key": k, "group": group_of(k), "shape": list(init.shape), "numel": int(init.size),
            "norm_init": float(ni), "norm_delta": float(nd), "rel_l2": rel, "cosine": cos,
            "bn_stat": is_bn_stat(k),
        })
        sum_d2 += nd * nd
        sum_i2 += ni * ni
        sum_dot += float(np.sum(init * trained))
        sum_t2 += nt * nt

    groups: dict[str, dict] = {}
    for g in sorted({p["group"] for p in per_param}):
        ps = [p for p in per_param if p["group"] == g]
        d2 = sum(p["norm_delta"] ** 2 for p in ps)
        i2 = sum(p["norm_init"] ** 2 for p in ps)
        rels = np.array([p["rel_l2"] for p in ps if not np.isnan(p["rel_l2"])])
        groups[g] = {
            "n_params": len(ps),
            "numel": int(sum(p["numel"] for p in ps)),
            "rel_l2_global": float(np.sqrt(d2) / np.sqrt(i2)) if i2 > 0 else float("nan"),
            "rel_l2_median": float(np.median(rels)) if len(rels) else float("nan"),
            "rel_l2_max": float(rels.max()) if len(rels) else float("nan"),
        }

    drifted = [p for p in per_param if not p["bn_stat"]]
    top10 = sorted(drifted, key=lambda p: -p["rel_l2"])[:10]

    report = {
        "n_params": len(per_param),
        "bn_stat_keys": len(bn_keys),
        "bn_exact_match": len(bn_keys) - len(bn_mismatch),
        "global": {
            "rel_l2": float(np.sqrt(sum_d2) / np.sqrt(sum_i2)),
            "cosine": float(sum_dot / (np.sqrt(sum_i2) * np.sqrt(sum_t2))),
            "norm_init": float(np.sqrt(sum_i2)),
            "norm_delta": float(np.sqrt(sum_d2)),
        },
        "groups": groups,
        "top10_drift": [{k: p[k] for k in ("key", "group", "shape", "rel_l2", "cosine")} for p in top10],
        "per_param": per_param,
    }
    logger.info("global drift: rel_l2=%.4f cosine=%.6f", report["global"]["rel_l2"], report["global"]["cosine"])
    for g, v in groups.items():
        logger.info("  %-10s n=%3d rel_l2(global)=%.4f median=%.4f max=%.4f",
                    g, v["n_params"], v["rel_l2_global"], v["rel_l2_median"], v["rel_l2_max"])
    return report


# --------------------------------------------------------------------------- #
# Part 2: feature-level drift + probe matrix (GPU)                            #
# --------------------------------------------------------------------------- #


def feature_eval(args: argparse.Namespace) -> dict:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import yaml

    from openpi.models.tactile_encoders import build_tactile_encoder
    from test.tactile_counterfactual import layer_probe as lp
    from test.tactile_counterfactual.tactile_linear_probe import probe_accuracy
    from test.tactile_counterfactual.tactile_linear_probe import logistic_fit

    cache = np.load(args.features_cache)
    labels, episodes = cache["labels"], cache["episodes"]
    n = len(labels)

    # --- collect the same 320 samples, same order as tactile_linear_probe --- #
    spec = yaml.safe_load(open(args.yaml))
    records = []
    for cat, lab in (("water", 1), ("no_water", 0)):
        for s in spec["categories"][cat]["samples"]:
            records.append((lab, int(s["episode"]) - 1, int(s["frame"])))

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    model, mc = setup.model, setup.model_config
    keys = tuple(mc.tactile_image_keys)
    available = setup.dataset.episodes
    records = [r for r in records if r[1] in available]
    assert len(records) == n, f"sample count {len(records)} != cache {n}"
    rec_labels = np.array([r[0] for r in records])
    rec_episodes = np.array([r[1] for r in records])
    if not (np.array_equal(rec_labels, labels) and np.array_equal(rec_episodes, episodes)):
        raise RuntimeError("records do not reproduce the cached labels/episodes order")

    imgs_all = np.empty((n, len(keys)), dtype=object)
    for i, (_, ep, fr) in enumerate(records):
        obs = setup.dataset.observation_from_sample(setup.dataset.get_sample(ep, fr))
        obs = model._preprocess_observation(None, obs, train=False)
        for j, k in enumerate(keys):
            imgs_all[i, j] = np.asarray(obs.images[k])[0]
        if (i + 1) % 80 == 0:
            logger.info("preprocessed %d/%d samples", i + 1, n)
    nv = len(keys)
    flat = jnp.asarray(np.stack([imgs_all[i, j] for i in range(n) for j in range(nv)]))

    def encode(module, x):
        graphdef, state = nnx.split(module)

        def fn(state, x):
            return nnx.merge(graphdef, state)(x)

        jit_fn = jax.jit(fn)
        outs = []
        for s in range(0, x.shape[0], args.batch_size):
            chunk = jit_fn(state, x[s : s + args.batch_size])
            if s == 0:
                ref = np.asarray(module(x[: args.batch_size]), np.float32)
                rel = float(np.linalg.norm(np.asarray(chunk, np.float32) - ref) / np.linalg.norm(ref))
                logger.info("jit-vs-eager sanity check: rel diff %.5f", rel)
                if rel > 0.05:
                    raise RuntimeError(f"jitted encoder disagrees with eager (rel {rel:.3f})")
            outs.append(np.asarray(chunk, np.float32))
        return np.concatenate(outs).reshape(n, nv, -1)

    # --- B sanity: re-encode with the trained encoder, compare to cache ----- #
    feats_trained = encode(model.tactile_encoder, flat)
    for agg, arr in (("mean4", feats_trained.mean(axis=1)), ("flat4", feats_trained.reshape(n, -1))):
        ref = cache[f"fastvit_{agg}"]
        rel = float(np.linalg.norm(arr - ref) / np.linalg.norm(ref))
        logger.info("trained-encoder vs cache fastvit_%s: rel diff %.6f", agg, rel)
        if rel > 1e-3:
            raise RuntimeError(f"re-encoded trained features disagree with cache (rel {rel}); sample order broken")

    # --- A: ImageNet-init encoder from the safetensors ----------------------- #
    compute_dtype = jnp.dtype(mc.tactile_compute_dtype)
    init_encoder = build_tactile_encoder(
        mc.tactile_encoder_name,
        rngs=nnx.Rngs(0),
        pretrained_path=args.init_safetensors,
        compute_dtype=compute_dtype,
    )
    # Verify the built encoder really holds the safetensors weights (all 448).
    from safetensors.flax import load_file

    flat_sd = load_file(str(pathlib.Path(args.init_safetensors).expanduser().resolve()))
    _, enc_state = nnx.split(init_encoder.module)
    pure = enc_state.to_pure_dict()

    def walk(d, prefix):
        for k, v in d.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, dict):
                yield from walk(v, p)
            else:
                yield p, np.asarray(v)

    enc_leaves = dict(walk(pure, ""))
    matched = sum(
        1 for k, v in flat_sd.items()
        if k in enc_leaves and enc_leaves[k].shape == v.shape and np.array_equal(enc_leaves[k], np.asarray(v))
    )
    logger.info("init encoder holds %d/%d safetensors tensors exactly", matched, len(flat_sd))
    if matched != len(flat_sd):
        raise RuntimeError(f"init encoder only picked up {matched}/{len(flat_sd)} pretrained tensors")

    feats_init = encode(init_encoder, flat)
    np.savez(pathlib.Path(args.out) / "features_init.npz",
             fastvit_mean4=feats_init.mean(axis=1), fastvit_flat4=feats_init.reshape(n, -1))

    # --- C: init features -> init tactile_proj (numpy Linear) ---------------- #
    proj = np.load(args.proj_drift)
    k_init, b_init = proj["k_init"], proj["b_init"]
    toks_init = feats_init.astype(np.float64) @ k_init + b_init  # [n, 4, width]
    toks_init = toks_init.astype(np.float32)

    variants = {
        "A_init_enc_mean4": feats_init.mean(axis=1),
        "A_init_enc_flat4": feats_init.reshape(n, -1),
        "B_trained_enc_mean4": cache["fastvit_mean4"],
        "B_trained_enc_flat4": cache["fastvit_flat4"],
        "C_init_enc_init_proj_mean4": toks_init.mean(axis=1),
        "C_init_enc_init_proj_flat4": toks_init.reshape(n, -1),
        "D_trained_enc_trained_proj_mean4": cache["token_mean4"],
        "D_trained_enc_trained_proj_flat4": cache["token_flat4"],
    }

    probes = {name: probe_accuracy(X.astype(np.float64), labels, episodes) for name, X in variants.items()}
    for name, r in probes.items():
        logger.info("%-34s episode-grouped %.3f | random %.3f", name,
                    r["episode_grouped"]["mean_acc"], r["random"]["mean_acc"])

    # --- feature drift: per-sample cosine, init vs trained encoder ---------- #
    def per_sample_cos(a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    drift = {}
    for agg in ("mean4", "flat4"):
        cos = per_sample_cos(variants[f"A_init_enc_{agg}"], variants[f"B_trained_enc_{agg}"])
        drift[agg] = {
            "cosine_mean": float(cos.mean()),
            "cosine_p05": float(np.percentile(cos, 5)),
            "cosine_p25": float(np.percentile(cos, 25)),
            "cosine_median": float(np.percentile(cos, 50)),
            "cosine_p75": float(np.percentile(cos, 75)),
            "cosine_p95": float(np.percentile(cos, 95)),
            "rel_l2_mean": float(np.mean(
                np.linalg.norm(variants[f"B_trained_enc_{agg}"].astype(np.float64)
                               - variants[f"A_init_enc_{agg}"].astype(np.float64), axis=1)
                / np.linalg.norm(variants[f"A_init_enc_{agg}"].astype(np.float64), axis=1))),
        }
        logger.info("feature cosine init-vs-trained (%s): mean %.4f median %.4f [p05 %.4f, p95 %.4f]",
                    agg, drift[agg]["cosine_mean"], drift[agg]["cosine_median"],
                    drift[agg]["cosine_p05"], drift[agg]["cosine_p95"])

    # --- feature geometry: where did the displacement go? -------------------- #
    A = variants["A_init_enc_mean4"].astype(np.float64)
    Bm = variants["B_trained_enc_mean4"].astype(np.float64)
    y = labels.astype(np.float64)

    d_all = Bm - A
    mean_d = d_all.mean(0)
    centered_cos = per_sample_cos(A - A.mean(0), Bm - Bm.mean(0))
    geometry = {
        "feat_norm_init_mean": float(np.linalg.norm(A, axis=1).mean()),
        "feat_norm_trained_mean": float(np.linalg.norm(Bm, axis=1).mean()),
        "across_sample_dim_std_init": float(A.std(0).mean()),
        "across_sample_dim_std_trained": float(Bm.std(0).mean()),
        "delta_norm_mean": float(np.linalg.norm(d_all, axis=1).mean()),
        "mean_delta_norm": float(np.linalg.norm(mean_d)),
        "delta_residual_norm_mean": float(np.linalg.norm(d_all - mean_d, axis=1).mean()),
        # share of displacement energy in the input-independent common-mode shift
        "common_mode_share": float(np.sum(mean_d**2) / np.mean(np.sum(d_all**2, axis=1))),
        # cosine between per-sample deviations from the grand mean (the part a
        # classifier can actually use)
        "cos_centered_mean": float(centered_cos.mean()),
        "cos_centered_median": float(np.percentile(centered_cos, 50)),
    }
    logger.info(
        "geometry: |f| %.2f->%.2f, across-sample dim std %.5f->%.5f, common-mode share %.3f, centered cos %.4f",
        geometry["feat_norm_init_mean"], geometry["feat_norm_trained_mean"],
        geometry["across_sample_dim_std_init"], geometry["across_sample_dim_std_trained"],
        geometry["common_mode_share"], geometry["cos_centered_mean"],
    )

    # --- direction alignment (mean4 space, raw features) --------------------- #
    mu, sd = Bm.mean(0), Bm.std(0) + 1e-8
    w_std, _ = logistic_fit((Bm - mu) / sd, y)
    w_raw = w_std / sd  # class-discriminant direction in raw feature space
    w_raw /= np.linalg.norm(w_raw)

    delta = Bm - A  # per-sample feature displacement from training
    dn = np.linalg.norm(delta, axis=1) + 1e-12
    cos_w = (delta @ w_raw) / dn
    d_cls = Bm[y == 1].mean(0) - Bm[y == 0].mean(0)
    d_cls /= np.linalg.norm(d_cls)
    mean_delta = delta.mean(0)

    # Class separation along the probe direction before/after encoder training.
    def separation(X):
        s = X @ w_raw
        s1, s0 = s[y == 1], s[y == 0]
        pooled = np.sqrt((s1.var() + s0.var()) / 2) + 1e-12
        return {"mean_water": float(s1.mean()), "mean_no_water": float(s0.mean()),
                "std": float(pooled), "d_prime": float((s1.mean() - s0.mean()) / pooled)}

    alignment = {
        # w_raw is fit IN-SAMPLE on the trained encoder's features, so the
        # trained-enc d' below is optimistic; use the cross-validated probe
        # accuracies for the honest separability comparison.
        "probe_direction_fit_on": "B_trained_enc_mean4 (in-sample)",
        "cos_delta_vs_probe_w": {
            "mean": float(cos_w.mean()), "median": float(np.percentile(cos_w, 50)),
            "p25": float(np.percentile(cos_w, 25)), "p75": float(np.percentile(cos_w, 75)),
            "frac_positive": float((cos_w > 0).mean()),
        },
        "cos_mean_delta_vs_probe_w": float(np.dot(mean_delta, w_raw) / (np.linalg.norm(mean_delta) + 1e-12)),
        "cos_mean_delta_vs_class_diff": float(np.dot(mean_delta, d_cls) / (np.linalg.norm(mean_delta) + 1e-12)),
        "cos_probe_w_vs_class_diff": float(np.dot(w_raw, d_cls)),
        "class_separation_init_enc": separation(A),
        "class_separation_trained_enc": separation(Bm),
    }
    logger.info("delta vs probe direction: mean cos %.4f (frac>0 %.2f); mean-delta vs w: %.4f; "
                "separation d' init %.3f -> trained %.3f",
                alignment["cos_delta_vs_probe_w"]["mean"], alignment["cos_delta_vs_probe_w"]["frac_positive"],
                alignment["cos_mean_delta_vs_probe_w"],
                alignment["class_separation_init_enc"]["d_prime"],
                alignment["class_separation_trained_enc"]["d_prime"])

    return {"probes": probes, "feature_drift": drift, "feature_geometry": geometry,
            "direction_alignment": alignment,
            "n_samples": int(n), "tactile_proj_shape": list(k_init.shape)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"checkpoint": args.checkpoint_dir, "init_safetensors": args.init_safetensors}
    report["weight_drift"] = weight_drift(args)
    if not args.skip_features:
        report["feature_eval"] = feature_eval(args)

    per_param = report["weight_drift"].pop("per_param")
    (out_dir / "eval_report.json").write_text(json.dumps(lp_to_jsonable(report), indent=1, ensure_ascii=False))
    np.savez(out_dir / "weight_drift_per_param.npz",
             keys=np.array([p["key"] for p in per_param]),
             groups=np.array([p["group"] for p in per_param]),
             rel_l2=np.array([p["rel_l2"] for p in per_param]),
             cosine=np.array([p["cosine"] for p in per_param]),
             bn_stat=np.array([p["bn_stat"] for p in per_param]))
    logger.info("wrote %s", out_dir / "eval_report.json")


def lp_to_jsonable(obj):
    from test.tactile_counterfactual import layer_probe as lp

    return lp.to_jsonable(obj)


if __name__ == "__main__":
    main()
