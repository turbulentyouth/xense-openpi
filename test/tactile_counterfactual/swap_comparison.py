"""Swap-tactile comparison figure + numbers for the v1.1 GRU refiner.

For a few representative in-window val frames (large correction, >=1 water and
>=1 no_water), plots the 50-step action trajectory of dim 9 (right_tcp.x, the
direction dim) and dim 0 (left_tcp.x, control) for:
  - a_vla               (raw VLA output; dashed -- the VLA itself moves only
                         ~0.2% under the tactile swap, so one line stands for both)
  - refiner(orig tac)   (solid)
  - refiner(swapped tac)(solid, other color; opposite-class donor, SAME pairing
                         as the acceptance swap test: seed = args.seed + 7 = 7)
Plus a bar chart of swap-induced relative action change: VLA ~0.2% / v1 3.5% /
v1.1 GRU 14.1% / v1.1 mean-pool 17.2%.

Outputs:
  outputs/refiner_v11/swap_comparison.png
  outputs/refiner_v11/swap_comparison.json

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/swap_comparison.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx  # noqa: E402

from test.tactile_counterfactual import train_refiner_v11 as tr11  # noqa: E402

logger = logging.getLogger("swap_comparison")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"
TAC_HIST = REPO_ROOT / "outputs" / "refiner_data" / "tac_history.npz"
PARAMS = REPO_ROOT / "outputs" / "refiner_v11" / "refiner_params.pkl"
ACCEPT = REPO_ROOT / "outputs" / "refiner_v11" / "acceptance.json"
OUT_PNG = REPO_ROOT / "outputs" / "refiner_v11" / "swap_comparison.png"
OUT_JSON = REPO_ROOT / "outputs" / "refiner_v11" / "swap_comparison.json"

DIM_DIR = 9  # right_tcp.x (direction dim)
DIM_CTRL = 0  # left_tcp.x (control)
N_PICK_PER_CLASS = 2
FIRST_STEPS = 10

# Acceptance-table numbers (measured earlier; VLA-internal swap from the
# counterfactual VLA rollouts, not recomputed here).
BAR_ENTRIES = [
    ("VLA 内部 (60k 模型)\n≈0.2% — 肉眼不可分", 0.0018),
    ("refiner v1\n(单帧 tac)", 0.035),
    ("refiner v1.1 GRU\n(4 帧历史)", 0.141),
    ("refiner v1.1 mean-pool\n(4 帧历史)", 0.172),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    z = np.load(DATA)
    zh = np.load(TAC_HIST)
    with open(PARAMS, "rb") as f:
        pkl = pickle.load(f)
    val_eps = set(json.loads(ACCEPT.read_text())["val_episodes"])

    episode = z["episode"].astype(np.int64)
    frame = z["frame"].astype(np.int64)
    label = z["label"].astype(np.int64)
    is_val = np.asarray([ep in val_eps for ep in episode])
    tac_hist = (zh["tac_hist"].astype(np.float32) - pkl["tac_mean"]) / pkl["tac_std"]
    a_vla_all = z["a_vla"].astype(np.float32).reshape(len(label), -1)

    model = tr11.RefinerV11(rngs=nnx.Rngs(0), history="gru")
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pkl["state"])
    eval_fn = tr11.make_eval_fn(graphdef)

    tac_va = tac_hist[is_val]
    avla_va = a_vla_all[is_val]
    label_va, ep_va, fr_va = label[is_val], episode[is_val], frame[is_val]
    win_va = z["in_window"][is_val].astype(bool)

    a_new, _ = tr11.batched_predict(eval_fn, state, tac_va, avla_va)

    # Acceptance swap pairing, verbatim (train_refiner_v11.py, seed = args.seed + 7).
    swap_rng = np.random.default_rng(7)
    swap_src = np.empty(len(tac_va), dtype=np.int64)
    for c in (0, 1):
        idx_c = np.nonzero(label_va == c)[0]
        idx_other = np.nonzero(label_va != c)[0]
        swap_src[idx_c] = idx_other[swap_rng.integers(0, len(idx_other), len(idx_c))]
    a_new_swap, _ = tr11.batched_predict(eval_fn, state, tac_va[swap_src], avla_va)

    swap_rel = np.linalg.norm(a_new_swap - a_new, axis=-1) / np.maximum(np.linalg.norm(a_new, axis=-1), 1e-12)
    logger.info(
        "sanity: recomputed swap rel_l2 mean %.4f (acceptance: %.4f), in-window %.4f",
        swap_rel.mean(),
        0.141,
        swap_rel[win_va].mean(),
    )

    # Representative frames: in-window, largest swap effect, per class.
    picks = []
    for c in (1, 0):  # water first
        idx = np.nonzero(win_va & (label_va == c))[0]
        order = idx[np.argsort(swap_rel[idx])[::-1][:N_PICK_PER_CLASS]]
        picks.extend(order.tolist())

    avla_3d = avla_va.reshape(-1, 50, 32)
    anew_3d = a_new.reshape(-1, 50, 32)
    aswap_3d = a_new_swap.reshape(-1, 50, 32)

    frame_reports = []
    for r in picks:
        d_vla = avla_3d[r]
        d_new = anew_3d[r]
        d_swap = aswap_3d[r]
        diff = d_swap - d_new  # swap effect, all dims
        corr = d_new - d_vla  # original correction
        d9 = diff[:, DIM_DIR]
        sgn_new = int(np.sign(corr[:, DIM_DIR].sum()))
        sgn_swap = int(np.sign((d_swap - d_vla)[:, DIM_DIR].sum()))
        abs_corr20 = np.abs(corr[:, :20]).sum(axis=1)  # per-step total correction
        abs_diff20 = np.abs(diff[:, :20]).sum(axis=1)  # per-step swap effect
        rep = {
            "episode": int(ep_va[r]),
            "frame": int(fr_va[r]),
            "label": int(label_va[r]),
            "donor_episode": int(ep_va[swap_src[r]]),
            "donor_frame": int(fr_va[swap_src[r]]),
            "swap_rel_l2_full_chunk": float(swap_rel[r]),
            "dim9_orig_vs_swap": {
                "max_abs_diff": float(np.abs(d9).max()),
                "mean_abs_diff": float(np.abs(d9).mean()),
                "direction_sign_orig": sgn_new,
                "direction_sign_swapped": sgn_swap,
                "direction_flipped": bool(sgn_new != 0 and sgn_swap != 0 and sgn_new != sgn_swap),
            },
            "correction_first10_vs_last40": {
                "orig_correction_sum_first10": float(abs_corr20[:FIRST_STEPS].sum()),
                "orig_correction_sum_last40": float(abs_corr20[FIRST_STEPS:].sum()),
                "orig_correction_per_step_first10": float(abs_corr20[:FIRST_STEPS].mean()),
                "orig_correction_per_step_last40": float(abs_corr20[FIRST_STEPS:].mean()),
                "swap_effect_sum_first10": float(abs_diff20[:FIRST_STEPS].sum()),
                "swap_effect_sum_last40": float(abs_diff20[FIRST_STEPS:].sum()),
                "swap_effect_per_step_first10": float(abs_diff20[:FIRST_STEPS].mean()),
                "swap_effect_per_step_last40": float(abs_diff20[FIRST_STEPS:].mean()),
            },
        }
        frame_reports.append(rep)
        logger.info(
            "pick ep%d f%d label=%d donor=ep%d f%d: swap_rel %.3f, dim9 max|Δ| %.3f mean|Δ| %.3f, dir %+d->%+d (flip=%s); "
            "corr 10/40 %.2f/%.2f, swap-eff 10/40 %.2f/%.2f",
            rep["episode"],
            rep["frame"],
            rep["label"],
            rep["donor_episode"],
            rep["donor_frame"],
            rep["swap_rel_l2_full_chunk"],
            rep["dim9_orig_vs_swap"]["max_abs_diff"],
            rep["dim9_orig_vs_swap"]["mean_abs_diff"],
            sgn_new,
            sgn_swap,
            rep["dim9_orig_vs_swap"]["direction_flipped"],
            rep["correction_first10_vs_last40"]["orig_correction_sum_first10"],
            rep["correction_first10_vs_last40"]["orig_correction_sum_last40"],
            rep["correction_first10_vs_last40"]["swap_effect_sum_first10"],
            rep["correction_first10_vs_last40"]["swap_effect_sum_last40"],
        )

    # ------------------------------------------------------------------ #
    # Figure                                                              #
    # ------------------------------------------------------------------ #
    n = len(picks)
    fig = plt.figure(figsize=(15, 3.4 * (n + 1) + 1.5))
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1.0] * n + [0.85], hspace=0.42, wspace=0.22)
    steps = np.arange(50)
    cls_name = {1: "water", 0: "no_water"}

    for row, (r, rep) in enumerate(zip(picks, frame_reports)):
        for col, (dim, dim_name) in enumerate([(DIM_DIR, "dim 9 = right_tcp.x（方向维）"), (DIM_CTRL, "dim 0 = left_tcp.x（对照）")]):
            ax = fig.add_subplot(gs[row, col])
            ax.axvspan(-0.5, FIRST_STEPS - 0.5, color="orange", alpha=0.08, lw=0)
            ax.plot(steps, avla_3d[r, :, dim], "k--", lw=1.6, label="a_vla（交换前后≈重合, ~0.2%）")
            ax.plot(steps, anew_3d[r, :, dim], "b-", lw=1.8, label="refiner（原触觉）")
            ax.plot(steps, aswap_3d[r, :, dim], "r-", lw=1.8, label=f"refiner（交换→{cls_name[1 - label_va[r]]} 触觉）")
            if col == 0:
                d = rep["dim9_orig_vs_swap"]
                title = (
                    f"ep{rep['episode']} f{rep['frame']} [{cls_name[rep['label']]}] {dim_name}\n"
                    f"swap rel_l2={rep['swap_rel_l2_full_chunk']:.3f} | dim9 max|Δ|={d['max_abs_diff']:.3f} "
                    f"mean|Δ|={d['mean_abs_diff']:.3f} | 方向 {d['direction_sign_orig']:+d}→{d['direction_sign_swapped']:+d}"
                    + ("（翻转）" if d["direction_flipped"] else "")
                )
            else:
                dd = np.abs(aswap_3d[r, :, dim] - anew_3d[r, :, dim])
                title = f"{dim_name} | swap max|Δ|={dd.max():.3f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("chunk step（阴影 = RTC 执行窗前 10 步）" if row == n - 1 else None)
            ax.set_ylabel("归一化动作值")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=9, loc="best")

    ax_bar = fig.add_subplot(gs[n, :])
    names = [e[0] for e in BAR_ENTRIES]
    vals = [100 * e[1] for e in BAR_ENTRIES]
    colors = ["#888888", "#1f77b4", "#d62728", "#ff7f0e"]
    bars = ax_bar.barh(np.arange(len(vals)), vals, color=colors, alpha=0.85, height=0.6)
    ax_bar.set_yticks(np.arange(len(vals)), names, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("交换触觉导致的动作相对变化 rel_l2（%）")
    ax_bar.set_title("量化对比：交换 water↔no_water 触觉后，动作改变了多少", fontsize=11)
    ax_bar.grid(axis="x", alpha=0.3)
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10)

    fig.suptitle("触觉交换（water↔no_water）对动作 chunk 的影响 — refiner v1.1 GRU, val 代表帧", fontsize=13)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", OUT_PNG)

    out = {
        "sanity_recomputed_swap_rel": {
            "mean_all_val": float(swap_rel.mean()),
            "mean_in_window": float(swap_rel[win_va].mean()),
            "acceptance_reference": 0.141,
        },
        "bar_entries_rel_l2": {e[0].split("\n")[0]: e[1] for e in BAR_ENTRIES},
        "note_vla": "VLA 内部交换实测 0.16-0.2%（此前 VLA 内部 swap rollout 测量），图中未重画第二条 VLA 线——肉眼不可分",
        "frames": frame_reports,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    logger.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
