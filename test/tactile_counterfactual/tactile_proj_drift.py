"""How much did 60k training steps move ``tactile_proj``?

Rebuilds the exact initial value of ``tactile_proj`` by replicating the
training-time RNG chain (config.seed=42 -> init_rng -> model_rng -> model
creation, same module construction order, so the same rng stream consumption),
and compares it against the trained kernel/bias stored in the checkpoint.

tactile_proj is NOT covered by the pi05_base weights (missing_regex backfills
it with the __init__ value), so "init" here is what step 0 actually started
from. Caveat: assumes the training run used the default seed=42.

Example:
    python test/tactile_counterfactual/tactile_proj_drift.py \
        --checkpoint-dir checkpoints/ckpt59999_bf16
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import jax  # noqa: E402

from openpi.training import config as _config  # noqa: E402

logger = logging.getLogger("tactile_proj_drift")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100")
    p.add_argument("--checkpoint-dir", default="checkpoints/ckpt59999_bf16")
    args = p.parse_args()

    train_config = _config.get_config(args.config_name)
    seed = train_config.seed
    model_config = train_config.model
    if (
        getattr(model_config, "tactile_pretrained_path", None) is not None
        and not pathlib.Path(model_config.tactile_pretrained_path).expanduser().exists()
    ):
        import dataclasses

        model_config = dataclasses.replace(model_config, tactile_pretrained_path=None)
    logger.info("rebuilding init model with seed=%d (rng chain: split(split(key(seed))[1])[1])", seed)

    # Replicate scripts/train.py: rng=key(seed); train_rng, init_rng = split(rng);
    # init(init_rng): rng, model_rng = split(init_rng); model = create(model_rng).
    rng = jax.random.key(seed)
    _, init_rng = jax.random.split(rng)
    _, model_rng = jax.random.split(init_rng)
    model = model_config.create(model_rng)

    k0 = np.asarray(model.tactile_proj.kernel.value, np.float64)
    b0 = np.asarray(model.tactile_proj.bias.value, np.float64)

    def load_leaf(path: pathlib.Path) -> np.ndarray:
        raw = np.load(path)
        if raw.dtype == np.uint16:  # bf16 bit pattern storage
            return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return raw.astype(np.float64)

    ckpt = pathlib.Path(args.checkpoint_dir) / "params_npy"
    k1 = load_leaf(ckpt / "params.tactile_proj.kernel.value.bf16.npy")
    b1 = load_leaf(ckpt / "params.tactile_proj.bias.value.bf16.npy")
    logger.info("kernel %s, bias %s", k1.shape, b1.shape)

    dk = k1 - k0
    rel = np.linalg.norm(dk) / np.linalg.norm(k0)
    cos = float(np.sum(k0 * k1) / (np.linalg.norm(k0) * np.linalg.norm(k1)))
    print(f"kernel: |init|={np.linalg.norm(k0):.3f} |trained|={np.linalg.norm(k1):.3f} |delta|={np.linalg.norm(dk):.3f}")
    print(f"kernel: rel_l2(delta/init) = {rel:.4f}, cosine(init, trained) = {cos:.6f}")
    print(f"kernel init stats: std={k0.std():.5f} (lecun_normal 1/sqrt(1024)={1/np.sqrt(1024):.5f})")
    print(f"bias: |init|={np.linalg.norm(b0):.5f} |trained|={np.linalg.norm(b1):.5f}")

    # Concentration: per output-unit (column) and per input-feature (row) rel change.
    col_rel = np.linalg.norm(dk, axis=0) / np.linalg.norm(k0, axis=0)
    row_rel = np.linalg.norm(dk, axis=1) / np.linalg.norm(k0, axis=1)
    for name, v in (("per output unit", col_rel), ("per input feature", row_rel)):
        q = np.percentile(v, [0, 25, 50, 75, 95, 100])
        print(f"{name} rel change: min={q[0]:.3f} p25={q[1]:.3f} median={q[2]:.3f} p75={q[3]:.3f} p95={q[4]:.3f} max={q[5]:.3f}")

    # Reference scale: expected rel change if the trained kernel were an
    # independent lecun_normal draw (~sqrt(2)); values << that mean training
    # kept most of the init structure.
    rng_np = np.random.default_rng(0)
    k_rand = rng_np.standard_normal(k0.shape) * (1 / np.sqrt(1024))
    print(f"reference: rel_l2 of an independent redraw = {np.linalg.norm(k_rand-k0)/np.linalg.norm(k0):.4f}")

    np.savez(
        pathlib.Path(args.checkpoint_dir).parent / "tactile_proj_drift.npz",
        k_init=k0, k_trained=k1, b_init=b0, b_trained=b1,
    )


if __name__ == "__main__":
    main()
