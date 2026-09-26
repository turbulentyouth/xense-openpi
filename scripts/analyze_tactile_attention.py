"""Analyze tactile attention dumps produced by ``scripts/dump_tactile_attention.py``.

Reads ``attention_tactile.npy`` (float16 memmap, ``[N, num_steps, depth, heads,
action_horizon, num_tactile]``) plus ``meta.json`` / ``frames.json`` from a dump
directory and reduces the raw softmax probability mass (never renormalized over
the tactile axis) along different dimension groups:

* ``step_layer``      ``[num_steps, depth]``            mean over (frame, head, action, tactile)
* ``step_layer_head`` ``[num_steps, depth, heads]``     mean over (frame, action, tactile)
* ``step_action``     ``[num_steps, action_horizon]``   mean over (frame, layer, head, tactile)
* ``sensor_overall``  ``[num_tactile]``                 mean over (frame, step, layer, head, action)
* ``step_sensor``     ``[num_steps, num_tactile]``      mean over (frame, layer, head, action)
* top-K ``(denoise_step, layer, head)`` triples ranked by ``step_layer_head`` mass

Outputs (written to ``--out``, default ``<input>/analysis``):

* ``summary.npz``             all arrays above
* ``summary.json``            sensor ranking, top-K list, meta copy, array shapes
* ``heatmap_step_layer.png``  step x layer heatmap (requires matplotlib)
* ``heatmap_step_sensor.png`` step x sensor heatmap (requires matplotlib)
* ``head_ranking.csv``        rank,denoise_step,layer,head,mass

Example:
    python scripts/analyze_tactile_attention.py --input /tmp/attn_smoke --top-k 20

Self-test (builds a tiny synthetic dump under /tmp and verifies the outputs):
    python scripts/analyze_tactile_attention.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys

import numpy as np

logger = logging.getLogger("analyze_tactile_attention")

# Max number of frames reduced per chunk; keeps the float32 working set bounded
# (16 frames at the production shape [10, 18, 8, 50, 4] is ~74 MB float32).
CHUNK_FRAMES = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="dump directory containing attention_tactile.npy, meta.json, frames.json")
    p.add_argument("--out", default=None, help="output directory (default: <input>/analysis)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run a synthetic end-to-end self test under /tmp and exit",
    )
    return p.parse_args()


def _reduce(attn: np.ndarray, num_steps: int, depth: int, heads: int, ah: int, n_tac: int) -> dict[str, np.ndarray]:
    """Chunked mean reductions over the frame axis of the float16 memmap."""
    n = attn.shape[0]
    acc = {
        "step_layer": np.zeros((num_steps, depth), dtype=np.float64),
        "step_layer_head": np.zeros((num_steps, depth, heads), dtype=np.float64),
        "step_action": np.zeros((num_steps, ah), dtype=np.float64),
        "sensor_overall": np.zeros((n_tac,), dtype=np.float64),
        "step_sensor": np.zeros((num_steps, n_tac), dtype=np.float64),
    }
    num_chunks = (n + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    for ci, c0 in enumerate(range(0, n, CHUNK_FRAMES)):
        block = attn[c0 : c0 + CHUNK_FRAMES].astype(np.float32)  # [B, S, L, H, A, T]
        # attn axes: (frame=0, step=1, layer=2, head=3, action=4, tactile=5)
        acc["step_layer"] += block.sum(axis=(0, 3, 4, 5))
        acc["step_layer_head"] += block.sum(axis=(0, 4, 5))
        acc["step_action"] += block.sum(axis=(0, 2, 3, 5))
        acc["sensor_overall"] += block.sum(axis=(0, 1, 2, 3, 4))
        acc["step_sensor"] += block.sum(axis=(0, 2, 3, 4))
        logger.info("reduced chunk %d/%d (frames %d..%d)", ci + 1, num_chunks, c0, min(c0 + CHUNK_FRAMES, n) - 1)

    out = {
        "step_layer": acc["step_layer"] / (n * heads * ah * n_tac),
        "step_layer_head": acc["step_layer_head"] / (n * ah * n_tac),
        "step_action": acc["step_action"] / (n * depth * heads * n_tac),
        "sensor_overall": acc["sensor_overall"] / (n * num_steps * depth * heads * ah),
        "step_sensor": acc["step_sensor"] / (n * depth * heads * ah),
    }
    return {k: v.astype(np.float32) for k, v in out.items()}


def _save_heatmap(matrix: np.ndarray, xlabel: str, ylabel: str, title: str, path: pathlib.Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping heatmap %s (data is in summary.npz)", path.name)
        return

    fig, ax = plt.subplots(figsize=(max(4, 0.5 * matrix.shape[1] + 2), max(3, 0.5 * matrix.shape[0] + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label="mean softmax mass")
    if matrix.size <= 400:  # annotate values only when the grid stays readable
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6, color="w")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def analyze(input_dir: pathlib.Path, out_dir: pathlib.Path, top_k: int) -> dict:
    attn_path = input_dir / "attention_tactile.npy"
    if not attn_path.exists():
        raise FileNotFoundError(f"{attn_path} not found; is --input a dump directory from dump_tactile_attention.py?")

    meta = json.loads((input_dir / "meta.json").read_text()) if (input_dir / "meta.json").exists() else {}
    frames = json.loads((input_dir / "frames.json").read_text()) if (input_dir / "frames.json").exists() else {}

    attn = np.load(attn_path, mmap_mode="r")
    if attn.ndim != 6:
        raise ValueError(f"expected 6D attention_tactile [N, steps, depth, heads, ah, n_tac], got shape {attn.shape}")
    n, num_steps, depth, heads, ah, n_tac = (int(v) for v in attn.shape)
    logger.info(
        "attention_tactile: shape=%s dtype=%s (%.1f MB)", attn.shape, attn.dtype, attn.size * attn.itemsize / 1e6
    )

    arrays = _reduce(attn, num_steps, depth, heads, ah, n_tac)

    # Top-K (denoise_step, layer, head) by step_layer_head mass, descending.
    slh = arrays["step_layer_head"]
    flat_idx = np.argsort(slh, axis=None)[::-1][:top_k]
    top_entries = []
    for rank, fi in enumerate(flat_idx, start=1):
        s, l, h = np.unravel_index(fi, slh.shape)
        top_entries.append(
            {"rank": rank, "denoise_step": int(s), "layer": int(l), "head": int(h), "mass": float(slh[s, l, h])}
        )

    tactile_keys = meta.get("tactile_keys") or [f"TAC_{i}" for i in range(n_tac)]
    sensor_ranking = sorted(
        ({"sensor": tactile_keys[i], "index": i, "mass": float(arrays["sensor_overall"][i])} for i in range(n_tac)),
        key=lambda d: d["mass"],
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "summary.npz", **arrays)

    meta_keys = (
        "config_name", "checkpoint_dir", "repo_id", "num_frames", "num_steps", "depth",
        "num_heads", "action_horizon", "action_dim", "num_tactile", "seed", "tactile_keys", "episodes",
    )
    summary = {
        "input_dir": str(input_dir),
        "num_frames_analyzed": n,
        "meta": {k: meta[k] for k in meta_keys if k in meta},
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "sensor_overall": [float(v) for v in arrays["sensor_overall"]],
        "sensor_ranking": sensor_ranking,
        "top_k": top_entries,
        "frames_episode_range": (
            [int(min(frames["episode"])), int(max(frames["episode"]))] if frames.get("episode") else None
        ),
        "note": "all values are means of raw softmax probability mass; the tactile axis is NOT renormalized",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    with open(out_dir / "head_ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "denoise_step", "layer", "head", "mass"])
        for e in top_entries:
            w.writerow([e["rank"], e["denoise_step"], e["layer"], e["head"], f"{e['mass']:.6e}"])

    _save_heatmap(
        arrays["step_layer"], "layer", "denoise step", "tactile attention mass (mean over heads/actions/sensors)",
        out_dir / "heatmap_step_layer.png",
    )
    _save_heatmap(
        arrays["step_sensor"], "tactile sensor", "denoise step", "tactile attention mass per sensor",
        out_dir / "heatmap_step_sensor.png",
    )

    logger.info("top sensor: %s (mass %.6f)", sensor_ranking[0]["sensor"], sensor_ranking[0]["mass"])
    t1 = top_entries[0]
    logger.info(
        "top-1 head: step=%d layer=%d head=%d (mass %.6f)", t1["denoise_step"], t1["layer"], t1["head"], t1["mass"]
    )
    logger.info("wrote analysis to %s", out_dir)
    return summary


def self_test() -> None:
    """Build a tiny synthetic dump under /tmp, run the analysis, and verify outputs."""
    import subprocess
    import tempfile

    rng = np.random.default_rng(0)
    n, steps, depth, heads, ah, n_tac = 3, 2, 4, 2, 5, 2
    top_k = 3

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="attn_selftest_") as tmp:
        dump = pathlib.Path(tmp) / "dump"
        dump.mkdir()
        # Values in [0, 0.25) so per-(query) tactile mass sums stay below 1.
        data = (rng.random((n, steps, depth, heads, ah, n_tac)) * 0.24).astype(np.float16)
        np.save(dump / "attention_tactile.npy", data)
        (dump / "meta.json").write_text(
            json.dumps(
                {
                    "config_name": "selftest",
                    "num_frames": n,
                    "num_steps": steps,
                    "depth": depth,
                    "num_heads": heads,
                    "action_horizon": ah,
                    "num_tactile": n_tac,
                    "seed": 0,
                    "tactile_keys": ["TAC_0", "TAC_1"],
                    "tensor_shapes": {"attention_tactile": list(data.shape)},
                },
                indent=1,
            )
        )
        (dump / "frames.json").write_text(json.dumps({"episode": [0, 0, 1], "frame": [10, 20, 5]}))

        out = dump / "analysis"
        cmd = [sys.executable, __file__, "--input", str(dump), "--out", str(out), "--top-k", str(top_k)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"analysis run failed:\n{proc.stdout}\n{proc.stderr}"

        expected = {
            "step_layer": (steps, depth),
            "step_layer_head": (steps, depth, heads),
            "step_action": (steps, ah),
            "sensor_overall": (n_tac,),
            "step_sensor": (steps, n_tac),
        }
        z = np.load(out / "summary.npz")
        for name, shape in expected.items():
            assert name in z, f"missing {name} in summary.npz"
            assert z[name].shape == shape, f"{name}: shape {z[name].shape} != {shape}"
            assert np.isfinite(z[name]).all(), f"{name} contains NaN/inf"

        # Cross-check reductions against a direct computation.
        d32 = data.astype(np.float32)
        np.testing.assert_allclose(z["step_layer"], d32.mean(axis=(0, 3, 4, 5)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(z["step_layer_head"], d32.mean(axis=(0, 4, 5)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(z["step_action"], d32.mean(axis=(0, 2, 3, 5)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(z["sensor_overall"], d32.mean(axis=(0, 1, 2, 3, 4)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(z["step_sensor"], d32.mean(axis=(0, 2, 3, 4)), rtol=1e-4, atol=1e-6)

        # Per-query softmax mass over the tactile subset never exceeds 1, so the mean cannot either.
        assert float(z["sensor_overall"].sum()) <= 1.0 + 1e-3, z["sensor_overall"]

        summary = json.loads((out / "summary.json").read_text())
        assert len(summary["top_k"]) == top_k
        assert summary["top_k"][0]["rank"] == 1
        masses = [e["mass"] for e in summary["top_k"]]
        assert masses == sorted(masses, reverse=True), "top_k not sorted by mass"
        assert len(summary["sensor_ranking"]) == n_tac
        assert summary["shapes"]["step_layer"] == [steps, depth]

        with open(out / "head_ranking.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["rank", "denoise_step", "layer", "head", "mass"]
        assert len(rows) == top_k + 1

        # Top-1 entry must match the argmax of step_layer_head.
        slh = z["step_layer_head"]
        s, l, h = np.unravel_index(int(np.argmax(slh)), slh.shape)
        t1 = summary["top_k"][0]
        assert (t1["denoise_step"], t1["layer"], t1["head"]) == (int(s), int(l), int(h))

        for png in ("heatmap_step_layer.png", "heatmap_step_sensor.png"):
            p = out / png
            assert p.exists() and p.stat().st_size > 0, f"missing or empty {png}"

    print("self-test PASSED")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input:
        raise SystemExit("--input is required (or pass --self-test)")
    input_dir = pathlib.Path(args.input).expanduser().resolve()
    out_dir = pathlib.Path(args.out).expanduser().resolve() if args.out else input_dir / "analysis"
    analyze(input_dir, out_dir, args.top_k)


if __name__ == "__main__":
    main()
