"""Convert a training checkpoint's params to a bfloat16 npy directory.

Motivation: the fp32 checkpoint is 12.9 GB and orbax's full-tree restore peaks at
~12 GB host RAM (all leaves are read concurrently), which gets OOM-killed on
machines with ~14 GB RAM. This converter streams leaf by leaf directly from the
ocdbt store via tensorstore (peak ~= largest single leaf, ~4.8 GiB) and writes one
``.bf16.npy`` per leaf (uint16 bit pattern of bfloat16, RNE-cast via jnp).
``test/tactile_counterfactual/runner.py: load_model`` detects the resulting
``params_npy/`` directory automatically.

Usage:
    python scripts/convert_checkpoint_bf16.py --src checkpoints/59999 --dst /tmp/ckpt59999_bf16

The output dir gets ``params_npy/`` (the weights) and ``assets/`` (copied norm
stats), i.e. it can be passed anywhere a checkpoint step dir is expected.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import shutil

import jax.numpy as jnp
import numpy as np
import tensorstore as ts

logger = logging.getLogger("convert_checkpoint_bf16")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="checkpoint step dir (containing params/) or the params dir itself")
    p.add_argument("--dst", required=True, help="output checkpoint dir; gets params_npy/ and assets/")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    src = pathlib.Path(args.src).expanduser().resolve()
    params_dir = src / "params" if (src / "params").is_dir() else src
    dst = pathlib.Path(args.dst).expanduser().resolve()
    npy_dir = dst / "params_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    base = f"file://{params_dir}"
    kv = ts.KvStore.open({"driver": "ocdbt", "base": base}).result()
    leaf_paths = sorted(
        k.decode()[: -len("/.zarray")] for k in kv.list().result() if k.decode().endswith("/.zarray")
    )
    logger.info("%d leaves in %s", len(leaf_paths), params_dir)

    total_bytes = 0
    for i, dotted in enumerate(leaf_paths):
        arr = ts.open({"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": base}, "path": dotted}, read=True).result()
        x = np.asarray(arr.read().result())
        del arr
        # Cast via jnp (round-to-nearest-even, same as the training/inference path),
        # store the raw bf16 bit pattern as uint16 (numpy has no native bf16).
        bits = np.asarray(jnp.asarray(x, dtype=jnp.bfloat16).view(jnp.uint16))
        np.save(npy_dir / f"{dotted}.bf16.npy", bits)
        total_bytes += bits.nbytes
        del x, bits
        if (i + 1) % 50 == 0:
            logger.info("%d/%d leaves converted", i + 1, len(leaf_paths))
    logger.info("converted %d leaves, %.2f GiB bf16", len(leaf_paths), total_bytes / 2**30)

    assets = src / "assets" if (src / "assets").is_dir() else src.parent / "assets"
    if assets.is_dir():
        if (dst / "assets").exists():
            shutil.rmtree(dst / "assets")
        shutil.copytree(assets, dst / "assets")
        logger.info("copied assets -> %s", dst / "assets")
    logger.info("done: %s", dst)


if __name__ == "__main__":
    main()
