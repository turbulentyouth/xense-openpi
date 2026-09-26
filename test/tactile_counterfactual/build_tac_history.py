"""Build tactile history windows for refiner v1.1: tac features at [t-3..t].

For every row of outputs/refiner_data/refiner_data.npz this encodes the 4
tactile images at frames fr-3, fr-2, fr-1 with the same frozen ImageNet
FastViT-T12 + preprocessing as the main build (frame fr itself is reused from
the main npz). Out-of-episode frames are clamped to frame 0 (boundary
replication). No VLA inference anywhere -- encoder only.

Output: outputs/refiner_data/tac_history.npz
    tac_hist [N, 4, 4, 1024] fp16  -- [sample, time(t-3..t), view, feat]
aligned row-for-row with refiner_data.npz.
"""

from __future__ import annotations

import argparse
import gc
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# NOTE: do NOT set --xla_gpu_autotune_level=0 (see run_tactile_swap.py).

import jax  # noqa: E402

from test.tactile_counterfactual import build_refiner_data as brd  # noqa: E402
from test.tactile_counterfactual import layer_probe as lp  # noqa: E402

logger = logging.getLogger("build_tac_history")

HISTORY = 4  # frames t-3 .. t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(brd.REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"))
    p.add_argument("--out", default=str(brd.REPO_ROOT / "outputs" / "refiner_data" / "tac_history.npz"))
    p.add_argument("--config-name", default=brd.CONFIG_NAME)
    p.add_argument("--checkpoint-dir", default=str(brd.CKPT_DIR), help="only for norm stats / data config")
    p.add_argument("--fastvit", default=str(brd.FASTVIT_SAFETENSORS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    return p.parse_args()


class _HistFrames:  # module-level: spawn pickling
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
        out["_i"] = np.asarray(i, dtype=np.int64)
        return out


def _collate(samples):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    z = np.load(args.data)
    ep = z["episode"].astype(np.int64)
    fr = z["frame"].astype(np.int64)
    tac_main = z["tac"]  # [N, 4, 1024] fp16, features at frame fr
    n = len(fr)

    # History frame rows (clamped at episode start), deduplicated; rows that are
    # themselves sampled frames reuse the main npz features instead of encoding.
    main_key = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, fr))}
    hist_ep = np.empty((n, HISTORY), dtype=np.int64)
    hist_fr = np.empty((n, HISTORY), dtype=np.int64)
    need: dict[tuple[int, int], int] = {}  # (ep, fr) -> slot in unique list
    uniq_ep, uniq_fr = [], []
    for i in range(n):
        for k in range(HISTORY):
            f = max(0, int(fr[i]) - (HISTORY - 1 - k))
            hist_ep[i, k], hist_fr[i, k] = ep[i], f
            key = (int(ep[i]), f)
            if key not in main_key and key not in need:
                need[key] = len(uniq_ep)
                uniq_ep.append(key[0])
                uniq_fr.append(key[1])
    logger.info("samples %d; unique history frames to encode: %d (rest reused from main npz)", n, len(uniq_ep))

    setup = lp.load_setup(args.config_name, args.checkpoint_dir)
    from openpi.models.tactile_encoders import build_tactile_encoder
    import flax.nnx as nnx

    encoder = build_tactile_encoder("fastvit_t12", rngs=nnx.Rngs(0), pretrained_path=args.fastvit)
    prep_fn = brd.make_prep_fn()
    encode_fn, estate = brd.make_encode_fn(encoder)
    tac_keys = tuple(setup.model_config.tactile_image_keys)
    nv = len(tac_keys)

    feats = np.empty((len(uniq_ep), nv, 1024), dtype=np.float16)
    if len(uniq_ep):
        import multiprocessing

        import torch.utils.data as torch_data

        ue = np.asarray(uniq_ep, dtype=np.int64)
        uf = np.asarray(uniq_fr, dtype=np.int64)
        loader = torch_data.DataLoader(
            _HistFrames(setup.dataset, ue, uf, brd._OBS_KEYS),  # noqa: SLF001
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            multiprocessing_context=multiprocessing.get_context("spawn") if args.num_workers > 0 else None,
            collate_fn=_collate,
            drop_last=False,
        )
        t0 = time.time()
        done = 0
        import jax.numpy as jnp

        for raw in loader:
            idx = np.asarray(raw.pop("_i"))
            obs = lp.observation_from_batch({k: raw[k] for k in brd._OBS_KEYS if k in raw})  # noqa: SLF001
            po = prep_fn(obs)
            imgs = jnp.stack([po.images[k] for k in tac_keys], axis=1)
            b, _, h, w, c = imgs.shape
            f = np.asarray(encode_fn(estate, imgs.reshape(b * nv, h, w, c)), dtype=np.float32)
            feats[idx] = f.reshape(b, nv, -1).astype(np.float16)
            done += b
            if done % (args.batch_size * 50) < b:
                rate = done / max(time.time() - t0, 1e-9)
                logger.info("%d/%d (%.1f fps, eta %.0f s)", done, len(uniq_ep), rate, (len(uniq_ep) - done) / max(rate, 1e-9))
        del loader
        gc.collect()

    # Assemble [N, 4, 4, 1024]: time-major [t-3..t], main frame's features from
    # the main npz (exact copy), the rest from the freshly encoded map.
    tac_hist = np.empty((n, HISTORY, nv, 1024), dtype=np.float16)
    for i in range(n):
        for k in range(HISTORY):
            key = (int(hist_ep[i, k]), int(hist_fr[i, k]))
            mi = main_key.get(key)
            if mi is not None:
                tac_hist[i, k] = tac_main[mi]
            else:
                tac_hist[i, k] = feats[need[key]]
    # Sanity: the t slot must equal the main npz row exactly.
    assert np.array_equal(tac_hist[:, HISTORY - 1], tac_main), "t-slot misaligned with main npz"

    out = pathlib.Path(args.out)
    np.savez(out, tac_hist=tac_hist)
    logger.info("wrote %s %s", out, tac_hist.shape)


if __name__ == "__main__":
    main()
