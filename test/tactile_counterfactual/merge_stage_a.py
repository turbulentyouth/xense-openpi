"""Merge stage-A slim params back into a full checkpoint tree.

Reads a slim params pickle (train_stage_a.py --save-slim) and the 60k bf16 npy
dir, writes checkpoints/ckpt_stage_a_bf16/ in exactly the ckpt59999_bf16 format
(params_npy/*.bf16.npy + assets/), so every existing analysis/eval script and
``layer_probe.load_setup`` accepts it unchanged:

  - expert leaves: slim ``PaliGemma.llm.*`` (no suffix) -> ``*_1`` full-tree names
  - action_in_proj / time_mlp_in / time_mlp_out / action_out_proj / tactile_proj:
    overwritten with the trained slim values
  - tactile_encoder.*: overwritten with the slim (frozen ImageNet-init) values --
    NOT the collapsed 60k encoder, matching what stage A actually ran with
  - tactile_aux_head.*: written as extra new leaves (eval scripts ignore them;
    model_config.load(remove_extra_params=True) intersects them away)
  - every other leaf (backbone, SigLIP, embedder): copied byte-for-byte
  - assets/: copied from ckpt59999_bf16

Validation: after writing, ``layer_probe.load_setup`` must load the merged dir
(the pytree equality check inside model_config.load is the real assertion).

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/merge_stage_a.py \
        --slim outputs/stage_a/slim_params_final.pkl \
        --dst checkpoints/ckpt_stage_a_bf16
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

logger = logging.getLogger("merge_stage_a")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "checkpoints" / "ckpt59999_bf16"
CONFIG_NAME = "pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100"

# Slim path prefix -> full-tree path prefix rewrites (order matters: longest first).
_REWRITES = (
    (re.compile(r"^PaliGemma\.llm\.layers\.attn\.(q_einsum|kv_einsum|attn_vec_einsum)\."), r"PaliGemma.llm.layers.attn.\1_1."),
    (re.compile(r"^PaliGemma\.llm\.layers\.(mlp|pre_attention_norm|pre_ffw_norm)\."), r"PaliGemma.llm.layers.\1_1."),
    (re.compile(r"^PaliGemma\.llm\.final_norm\."), "PaliGemma.llm.final_norm_1."),
)
_PASSTHROUGH_TOP = ("action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj", "tactile_proj", "tactile_encoder", "tactile_aux_head")


def slim_to_full_path(dotted: str) -> str:
    for pattern, repl in _REWRITES:
        if pattern.match(dotted):
            return pattern.sub(repl, dotted)
    top = dotted.split(".")[0]
    if top in _PASSTHROUGH_TOP:
        return dotted
    raise ValueError(f"unmapped slim path: {dotted}")


def _flatten(tree: dict, prefix: tuple = ()) -> dict[tuple, object]:
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flatten(v, (*prefix, k)))
        else:
            out[(*prefix, k)] = v
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slim", required=True, help="slim params pickle from train_stage_a.py")
    p.add_argument("--dst", default=str(REPO_ROOT / "checkpoints" / "ckpt_stage_a_bf16"))
    p.add_argument("--src", default=str(SRC_DIR))
    args = p.parse_args()

    from test.tactile_counterfactual.train_stage_a import encode_bf16

    src = pathlib.Path(args.src).expanduser().resolve()
    dst = pathlib.Path(args.dst).expanduser().resolve()
    npy_src = src / "params_npy"
    npy_dst = dst / "params_npy"
    npy_dst.mkdir(parents=True, exist_ok=True)

    with open(args.slim, "rb") as f:
        slim_pure = pickle.load(f)
    slim_flat = {".".join(str(x) for x in path): v for path, v in _flatten(slim_pure).items()}
    full_from_slim = {slim_to_full_path(k): np.asarray(v) for k, v in slim_flat.items()}
    logger.info("slim pickle: %d leaves -> %d full-tree leaves", len(slim_flat), len(full_from_slim))

    src_files = {f.name: f for f in npy_src.glob("*.bf16.npy")}
    replaced, copied, added = 0, 0, 0
    consumed: set[str] = set()

    for name, f in sorted(src_files.items()):
        dotted = name[: -len(".bf16.npy")]  # params.<path>.value
        assert dotted.startswith("params.") and dotted.endswith(".value")
        key = dotted[len("params.") : -len(".value")]
        out = npy_dst / name
        if key in full_from_slim:
            arr = full_from_slim[key]
            ref_shape = np.load(f).shape
            if tuple(arr.shape) != tuple(ref_shape):
                raise ValueError(f"shape mismatch for {key}: slim {arr.shape} vs ckpt {ref_shape}")
            np.save(out, encode_bf16(arr.astype(np.float32)))
            consumed.add(key)
            replaced += 1
        else:
            shutil.copyfile(f, out)
            copied += 1

    # New leaves that only exist in the slim tree (aux head).
    for key, arr in sorted(full_from_slim.items()):
        if key in consumed:
            continue
        name = f"params.{key}.value.bf16.npy"
        if (npy_dst / name).exists():
            continue  # already written via the replacement path
        np.save(npy_dst / name, encode_bf16(arr.astype(np.float32)))
        consumed.add(key)
        added += 1
    leftover = set(full_from_slim) - consumed
    if leftover:
        raise RuntimeError(f"slim leaves not written: {sorted(leftover)[:10]}")

    # Sanity: every expert `_1` leaf in the source must have been replaced (not copied).
    expert_keys = {
        name[: -len(".bf16.npy")][len("params.") : -len(".value")]
        for name in src_files
        if (".llm.layers." in name and "_1." in name) or ".llm.final_norm_1." in name
    }
    not_replaced = expert_keys - consumed
    if not_replaced:
        raise RuntimeError(f"expert _1 leaves were copied instead of replaced: {sorted(not_replaced)[:10]}")
    logger.info(
        "replaced %d leaves (incl. all %d expert _1 leaves), copied %d, added %d new",
        replaced,
        len(expert_keys),
        copied,
        added,
    )

    assets_src = src / "assets"
    if assets_src.is_dir():
        if (dst / "assets").exists():
            shutil.rmtree(dst / "assets")
        shutil.copytree(assets_src, dst / "assets")
        logger.info("copied assets -> %s", dst / "assets")

    # ---- validation: the merged dir must load through the standard probe path ----
    from test.tactile_counterfactual import layer_probe

    setup = layer_probe.load_setup(CONFIG_NAME, dst)
    logger.info(
        "load_setup OK: model=%s, dataset=%d episodes",
        type(setup.model).__name__,
        setup.dataset.num_episodes,
    )
    logger.info("merge complete: %s", dst)


if __name__ == "__main__":
    main()
