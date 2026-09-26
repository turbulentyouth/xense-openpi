"""Build the v1 tactile-refiner dataset.

For every labeled frame (aux label != -1, stride 2) of the bottle-sorting
dataset this records:

- ``a_vla``   [50, 32] open-loop action chunk from the 60k production model
  (checkpoints/ckpt59999_bf16) under full normal inference (prefix included,
  10 denoise steps, zeros noise -- see metadata).
- ``a_demo``  [50, 32] the same frame's ``actions`` from the transformed data
  pipeline (DeltaActions + norm stats already applied; nothing recomputed by
  hand, no time offset).
- ``tac``     [4, 1024] features of the 4 preprocessed tactile images from a
  standalone FROZEN ImageNet FastViT-T12 (checkpoints/params.safetensors) --
  deliberately NOT the collapsed 60k in-model encoder.
- ``label`` / ``episode`` / ``in_window`` (first 30 frames of the grasp
  segment) / ``release_tcp_x`` (right_tcp.x of the segment's release frame,
  the same raw signal compute_aux_labels.py used for labelling).

Two-phase usage -- ALWAYS run validation first and inspect validation.json:

    python test/tactile_counterfactual/build_refiner_data.py --validate-only
    python test/tactile_counterfactual/build_refiner_data.py

Validation reports a_vla-vs-a_demo L2 on 20 random frames (catch action
convention mistakes before the expensive full pass), the tac feature per-dim
std (catch a collapsed encoder), and an empirical action-layout check of
dim 9 = right_tcp.x (delta) against the raw future state trajectory.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# NOTE: do NOT set --xla_gpu_autotune_level=0 (wrong bf16 conv numerics on this
# machine; see run_tactile_swap.py).

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import openpi.models.model as _model  # noqa: E402
from openpi.models.tactile_encoders import build_tactile_encoder  # noqa: E402
from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("build_refiner_data")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_NAME = "pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100"
CKPT_DIR = REPO_ROOT / "checkpoints" / "ckpt59999_bf16"
FASTVIT_SAFETENSORS = REPO_ROOT / "checkpoints" / "params.safetensors"
AUX_LABELS = REPO_ROOT / "outputs" / "stage_a" / "aux_labels.npz"
SEGMENTS_REPORT = REPO_ROOT / "outputs" / "stage_a" / "aux_labels.report.json"

RIGHT_TCP_X = 9  # BiFlexiv state/action layout, src/openpi/training/config.py:438
IN_WINDOW_FRAMES = 30
_OBS_KEYS = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default=CONFIG_NAME)
    p.add_argument("--checkpoint-dir", default=str(CKPT_DIR))
    p.add_argument("--fastvit", default=str(FASTVIT_SAFETENSORS), help="frozen ImageNet FastViT weights")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "refiner_data"))
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--validate-only", action="store_true", help="20-frame validation pass, no full build")
    p.add_argument("--validate-frames", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2000)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Frame table                                                                 #
# --------------------------------------------------------------------------- #


def build_frame_table(setup: lp.ProbeSetup, stride: int) -> list[dict]:
    """All labeled frames at `stride` with label/in_window/release_tcp_x."""
    z = np.load(AUX_LABELS)
    segments = json.loads(SEGMENTS_REPORT.read_text())["segments"]  # ep -> [[start, end, label]]

    hf = setup.dataset._raw_dataset.hf_dataset  # noqa: SLF001 (own infra; avoids a second parquet load)
    states = np.asarray(hf["observation.state"], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    index = np.asarray(hf["index"], dtype=np.int64)

    rows: list[dict] = []
    for ep_str, segs in segments.items():
        ep = int(ep_str)
        key = f"ep{ep}"
        if key not in z:
            continue
        labels = z[key]
        ep_rows = np.nonzero(episodes == ep)[0]
        start0 = int(index[ep_rows].min())
        ep_states = states[ep_rows]
        labeled = np.nonzero(labels != -1)[0][::stride]
        if labeled.size == 0:
            continue
        starts = np.asarray([s[0] for s in segs])
        ends = np.asarray([s[1] for s in segs])
        for fr in labeled.tolist():
            si = int(np.searchsorted(ends, fr, side="right"))
            if si >= len(segs) or not (starts[si] <= fr < ends[si]):
                raise RuntimeError(f"labeled frame ep{ep}f{fr} not inside any reported segment")
            s, e, lab = segs[si]
            if int(labels[fr]) != int(lab):
                raise RuntimeError(f"label mismatch ep{ep}f{fr}: npz {labels[fr]} vs segment {lab}")
            rows.append(
                {
                    "episode": ep,
                    "frame": fr,
                    "label": int(lab),
                    "in_window": bool(fr - s < IN_WINDOW_FRAMES),
                    "release_tcp_x": float(ep_states[min(e, ep_states.shape[0] - 1), RIGHT_TCP_X]),
                    "current_tcp_x": float(ep_states[fr, RIGHT_TCP_X]),
                    "ep_len": int(ep_states.shape[0]),
                    "_start0": start0,  # unused, kept for debugging
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Model-side pieces (all jits take state explicitly; graphdef closed over)     #
# --------------------------------------------------------------------------- #


def make_infer_fn(model: _model.BaseModel, num_steps: int):
    graphdef, state = nnx.split(model)

    def infer(state, obs, noise):
        m = nnx.merge(graphdef, state)
        return m.sample_actions(None, obs, noise=noise, num_steps=num_steps)

    return jax.jit(infer), state


def make_prep_fn():
    def prep(obs):
        return _model.preprocess_observation_tactile(None, obs, train=False, image_keys=_model.IMAGE_KEYS_TACTILE_4)

    return jax.jit(prep)


def make_encode_fn(encoder):
    graphdef, state = nnx.split(encoder)

    def encode(state, imgs):
        m = nnx.merge(graphdef, state)
        return m(imgs)

    return jax.jit(encode), state


def run_rows(setup, infer_pack, prep_fn, encode_pack, rows: list[dict], batch_size: int, num_steps: int) -> dict:
    """Run a list of frame rows through VLA inference + frozen encoder (no DataLoader)."""
    infer_fn, mstate = infer_pack
    encode_fn, estate = encode_pack
    tac_keys = tuple(setup.model_config.tactile_image_keys)
    ah, ad = setup.model_config.action_horizon, setup.model_config.action_dim
    n = len(rows)
    a_vla = np.empty((n, ah, ad), dtype=np.float16)
    a_demo = np.empty((n, ah, ad), dtype=np.float16)
    tac = np.empty((n, len(tac_keys), 1024), dtype=np.float16)
    for i0 in range(0, n, batch_size):
        chunk = rows[i0 : i0 + batch_size]
        samples = [setup.dataset.get_sample(r["episode"], r["frame"]) for r in chunk]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)
        obs = lp.observation_from_batch({k: batch[k] for k in _OBS_KEYS if k in batch})
        b = obs.state.shape[0]
        noise = jnp.zeros((b, ah, ad), dtype=jnp.float32)  # fixed zeros-noise convention
        av = np.asarray(infer_fn(mstate, obs, noise), dtype=np.float32)
        po = prep_fn(obs)
        imgs = jnp.stack([po.images[k] for k in tac_keys], axis=1)  # [B, 4, 224, 224, 3]
        b_, nv, h, w, c = imgs.shape
        feats = np.asarray(encode_fn(estate, imgs.reshape(b_ * nv, h, w, c)), dtype=np.float32)
        sl = slice(i0, i0 + b)
        a_vla[sl] = av.astype(np.float16)
        a_demo[sl] = np.asarray(batch["actions"], dtype=np.float32).astype(np.float16)
        tac[sl] = feats.reshape(b_, nv, -1).astype(np.float16)
        logger.info("rows %d-%d / %d done", sl.start, sl.stop - 1, n)
    return {"a_vla": a_vla, "a_demo": a_demo, "tac": tac}


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def validate(setup, infer_pack, prep_fn, encode_pack, rows: list[dict], args) -> dict:
    rng = np.random.default_rng(0)
    picks = rng.choice(len(rows), size=min(args.validate_frames, len(rows)), replace=False)
    vrows = [rows[int(i)] for i in picks]
    out = run_rows(setup, infer_pack, prep_fn, encode_pack, vrows, batch_size=args.batch_size, num_steps=args.num_steps)
    av = out["a_vla"].astype(np.float32)
    ad_ = out["a_demo"].astype(np.float32)
    tac = out["tac"].astype(np.float32)

    per_frame_rel = np.linalg.norm(av - ad_, axis=(1, 2)) / np.maximum(np.linalg.norm(ad_, axis=(1, 2)), 1e-12)
    per_dim_abs = np.abs(av - ad_).mean(axis=(0, 1))  # [32]
    top_dims = np.argsort(per_dim_abs)[::-1][:8].tolist()

    # Action layout check: dim 9 of the delta-action chunk should track the raw
    # future right_tcp.x displacement x(fr+t) - x(fr) (DeltaActions: a[t] =
    # absolute(fr+t) - state(fr), then normalised; monotonic in raw space).
    hf = setup.dataset._raw_dataset.hf_dataset  # noqa: SLF001
    states = np.asarray(hf["observation.state"], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    index = np.asarray(hf["index"], dtype=np.int64)
    ep_start = {int(ep): int(index[episodes == ep].min()) for ep in np.unique(episodes)}
    demo_dim9, raw_disp = [], []
    for r, demo in zip(vrows, ad_):
        ep, fr = r["episode"], r["frame"]
        ep_states = states[episodes == ep]
        cur = ep_states[fr, RIGHT_TCP_X]
        fut = ep_states[fr + 1 : min(fr + 51, ep_states.shape[0]), RIGHT_TCP_X] - cur
        demo_dim9.append(demo[: len(fut), RIGHT_TCP_X])
        raw_disp.append(fut)
    demo_cat = np.concatenate(demo_dim9)
    raw_cat = np.concatenate(raw_disp)
    layout_corr = float(np.corrcoef(demo_cat, raw_cat)[0, 1])

    tac_std = tac.reshape(-1, tac.shape[-1]).std(axis=0)  # per-dim std over frames x views
    report = {
        "frames": [{"episode": r["episode"], "frame": r["frame"], "label": r["label"]} for r in vrows],
        "a_vla_shape": list(av.shape),
        "a_demo_shape": list(ad_.shape),
        "rel_l2_per_frame": per_frame_rel.tolist(),
        "rel_l2_mean": float(per_frame_rel.mean()),
        "rel_l2_median": float(np.median(per_frame_rel)),
        "norm_a_vla_mean": float(np.linalg.norm(av, axis=(1, 2)).mean()),
        "norm_a_demo_mean": float(np.linalg.norm(ad_, axis=(1, 2)).mean()),
        "per_dim_abs_err_top8": [{"dim": int(d), "mean_abs_err": float(per_dim_abs[d])} for d in top_dims],
        "action_layout_dim9": {
            "corr_demo_dim9_vs_raw_future_x_displacement": layout_corr,
            "note": "a_demo[t,9] vs raw x(fr+t)-x(fr) pooled over validation frames; "
            "DeltaActions makes dim 9 the normalised right_tcp.x displacement",
        },
        "tac_feat_std": {
            "mean": float(tac_std.mean()),
            "min": float(tac_std.min()),
            "max": float(tac_std.max()),
            "frac_dead_dims_std_lt_1e-6": float((tac_std < 1e-6).mean()),
        },
        "release_tcp_x": [r["release_tcp_x"] for r in vrows],
        "current_tcp_x": [r["current_tcp_x"] for r in vrows],
    }
    return report


# --------------------------------------------------------------------------- #
# Full build (torch DataLoader, spawn workers -- same pattern as layer_probe)  #
# --------------------------------------------------------------------------- #


class _Frames:  # module-level: the spawn multiprocessing context must pickle it
    def __init__(self, dataset, episodes: np.ndarray, frames: np.ndarray, obs_keys) -> None:
        self._dataset = dataset
        self._episodes = episodes
        self._frames = frames
        self._obs_keys = obs_keys

    def __len__(self):
        return self._episodes.shape[0]

    def __getitem__(self, i):
        s = self._dataset.get_sample(int(self._episodes[i]), int(self._frames[i]))
        out = {k: s[k] for k in self._obs_keys if k in s}
        out["actions"] = s["actions"]
        out["_i"] = np.asarray(i, dtype=np.int64)
        return out


def _collate(samples):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)


def full_build(setup, infer_pack, prep_fn, encode_pack, rows: list[dict], args, out_dir: pathlib.Path) -> None:
    import multiprocessing

    import torch.utils.data as torch_data

    infer_fn, mstate = infer_pack
    encode_fn, estate = encode_pack
    tac_keys = tuple(setup.model_config.tactile_image_keys)
    ah, ad = setup.model_config.action_horizon, setup.model_config.action_dim
    nv = len(tac_keys)
    n = len(rows)

    episodes = np.asarray([r["episode"] for r in rows], dtype=np.int64)
    frames = np.asarray([r["frame"] for r in rows], dtype=np.int64)

    loader = torch_data.DataLoader(
        _Frames(setup.dataset, episodes, frames, _OBS_KEYS),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        multiprocessing_context=multiprocessing.get_context("spawn") if args.num_workers > 0 else None,
        collate_fn=_collate,
        drop_last=False,
    )

    a_vla = np.empty((n, ah, ad), dtype=np.float16)
    a_demo = np.empty((n, ah, ad), dtype=np.float16)
    tac = np.empty((n, nv, 1024), dtype=np.float16)
    done = 0
    part = 0
    t0 = time.time()

    def save_part(upto: int) -> None:
        nonlocal part
        p = out_dir / f"refiner_data.part{part:03d}.npz"
        np.savez(p, a_vla=a_vla[:upto], a_demo=a_demo[:upto], tac=tac[:upto])
        part += 1
        logger.info("incremental save %s (%d frames, %.0f s elapsed)", p.name, upto, time.time() - t0)

    for raw in loader:
        idx = np.asarray(raw.pop("_i"))
        actions = np.asarray(raw.pop("actions"), dtype=np.float32)
        obs = lp.observation_from_batch({k: raw[k] for k in _OBS_KEYS if k in raw})
        b = obs.state.shape[0]
        noise = jnp.zeros((b, ah, ad), dtype=jnp.float32)
        av = np.asarray(infer_fn(mstate, obs, noise), dtype=np.float32)
        po = prep_fn(obs)
        imgs = jnp.stack([po.images[k] for k in tac_keys], axis=1)
        _, _, h, w, c = imgs.shape
        feats = np.asarray(encode_fn(estate, imgs.reshape(b * nv, h, w, c)), dtype=np.float32)
        a_vla[idx] = av.astype(np.float16)
        a_demo[idx] = actions.astype(np.float16)
        tac[idx] = feats.reshape(b, nv, -1).astype(np.float16)
        done += b
        if done % args.save_every < b:
            save_part(done)
        if done % (args.batch_size * 20) < b:
            rate = done / max(time.time() - t0, 1e-9)
            logger.info("%d/%d frames (%.1f fps, eta %.0f s)", done, n, rate, (n - done) / max(rate, 1e-9))

    # Close the loader before the final save (host-memory OOM lesson).
    del loader
    gc.collect()

    labels = np.asarray([r["label"] for r in rows], dtype=np.int8)
    ep_out = episodes.astype(np.int16)
    fr_out = frames.astype(np.int32)
    in_window = np.asarray([r["in_window"] for r in rows], dtype=bool)
    release_x = np.asarray([r["release_tcp_x"] for r in rows], dtype=np.float32)
    np.savez(
        out_dir / "refiner_data.npz",
        a_vla=a_vla,
        a_demo=a_demo,
        tac=tac,
        label=labels,
        episode=ep_out,
        frame=fr_out,
        in_window=in_window,
        release_tcp_x=release_x,
    )
    logger.info("wrote %s", out_dir / "refiner_data.npz")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(args.fastvit).expanduser().exists():
        raise FileNotFoundError(f"frozen FastViT weights missing: {args.fastvit}")

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    rows = build_frame_table(setup, args.stride)
    n_water = sum(r["label"] == 1 for r in rows)
    logger.info(
        "frame table: %d rows (water %d / no_water %d), episodes %d, in_window %d",
        len(rows),
        n_water,
        len(rows) - n_water,
        len({r["episode"] for r in rows}),
        sum(r["in_window"] for r in rows),
    )

    encoder = build_tactile_encoder("fastvit_t12", rngs=nnx.Rngs(0), pretrained_path=args.fastvit)
    logger.info("frozen encoder feature_dim=%d", encoder.feature_dim)
    infer_pack = make_infer_fn(setup.model, args.num_steps)
    prep_fn = make_prep_fn()
    encode_pack = make_encode_fn(encoder)

    if args.validate_only:
        report = validate(setup, infer_pack, prep_fn, encode_pack, rows, args)
        report["frame_table_size"] = len(rows)
        (out_dir / "validation.json").write_text(json.dumps(report, indent=1))
        logger.info("validation: rel_l2 mean %.4f median %.4f", report["rel_l2_mean"], report["rel_l2_median"])
        logger.info("validation: dim9 layout corr %.4f", report["action_layout_dim9"]["corr_demo_dim9_vs_raw_future_x_displacement"])
        logger.info("validation: tac std mean %.5f, dead dims %.4f", report["tac_feat_std"]["mean"], report["tac_feat_std"]["frac_dead_dims_std_lt_1e-6"])
        logger.info("wrote %s", out_dir / "validation.json")
        return

    t0 = time.time()
    full_build(setup, infer_pack, prep_fn, encode_pack, rows, args, out_dir)
    meta = {
        "config_name": args.config_name,
        "checkpoint_dir": str(pathlib.Path(args.checkpoint_dir).resolve()),
        "noise": "zeros [B,50,32] (fixed, deterministic)",
        "num_steps": args.num_steps,
        "stride": args.stride,
        "n_frames": len(rows),
        "n_water": sum(r["label"] == 1 for r in rows),
        "n_no_water": sum(r["label"] == 0 for r in rows),
        "n_in_window": sum(r["in_window"] for r in rows),
        "n_episodes": len({r["episode"] for r in rows}),
        "tactile_encoder": "fastvit_t12 frozen ImageNet (standalone build_tactile_encoder), weights "
        + str(pathlib.Path(args.fastvit).resolve()),
        "action_space": "normalised delta-cartesian (pipeline actions, dims per LeRobotBiFlexivDataConfig: "
        "right_tcp.x = dim 9, right_gripper.pos = dim 19)",
        "build_seconds": time.time() - t0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1))
    logger.info("wrote %s", out_dir / "metadata.json")


if __name__ == "__main__":
    main()
