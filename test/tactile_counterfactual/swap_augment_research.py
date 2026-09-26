"""Pre-study for tactile-swap counterfactual augmentation (research only, no training).

Answers, with numbers, on top of outputs/refiner_data/refiner_data.npz:

1. Class-conditional difference structure of a_demo: Delta(step, dim) =
   mean_water - mean_no_water and its SNR against the pooled within-class std.
   Where do the differences concentrate (chunk steps, action dims)? ||Delta||
   per chunk step.
2. Direction commitment point: per grasp-segment phase (frame - segment start),
   how separable are the two classes' chunks (SNR curve) -- from which segment
   frame on do the trajectories diverge?
3. Phase alignment: phase = frame - segment start (segments from
   compute_aux_labels.py's report); sample-count distribution per phase.
4. Target synthesis feasibility:
   (a) target = a_demo +/- Delta(phase), SNR>2-masked -- per-phase sample counts;
   (b) same-episode phase-matched opposite-class chunk as target -- how many
       frames actually have such a match.
5. Augmentation ratio / restriction recommendation.

Outputs: outputs/refiner_v11/research.json and (for the trainer)
outputs/refiner_v11/delta_phase.npz with the per-phase Delta table + SNR mask.

Pure numpy: no GPU, no dataset/video access (segments come from the JSON
report, everything else from the npz).
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

logger = logging.getLogger("swap_augment_research")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"
SEGMENTS = REPO_ROOT / "outputs" / "stage_a" / "aux_labels.report.json"
OUT_DIR = REPO_ROOT / "outputs" / "refiner_v11"

RIGHT_TCP_X = 9
SNR_KEEP = 2.0
MIN_PER_PHASE_CLASS = 20
PHASE_BIN = 10
MIN_DELTA_NORM = 0.5
AUG_MAX_PHASE = 100


def load() -> tuple[dict, np.ndarray, np.ndarray]:
    z = np.load(DATA)
    segs_raw = json.loads(SEGMENTS.read_text())["segments"]
    ep = z["episode"].astype(np.int64)
    fr = z["frame"].astype(np.int64)
    seg_start = np.empty(len(fr), dtype=np.int64)
    seg_end = np.empty(len(fr), dtype=np.int64)
    seg_label = np.empty(len(fr), dtype=np.int64)
    segs = {int(k): v for k, v in segs_raw.items()}
    for e, ss in segs.items():
        m = ep == e
        if not m.any():
            continue
        starts = np.asarray([s[0] for s in ss])
        ends = np.asarray([s[1] for s in ss])
        idx = np.searchsorted(ends, fr[m], side="right")
        for row, i in zip(np.nonzero(m)[0], idx):
            assert i < len(ss) and starts[i] <= fr[row] < ends[i], (e, int(fr[row]))
            seg_start[row], seg_end[row], seg_label[row] = starts[i], ends[i], ss[i][2]
    assert (seg_label == z["label"].astype(np.int64)).all()
    phase = fr - seg_start
    info = {
        "episode": ep,
        "frame": fr,
        "label": z["label"].astype(np.int64),
        "seg_start": seg_start,
        "seg_end": seg_end,
        "seg_len": seg_end - seg_start,
    }
    return info, phase, z["a_demo"].astype(np.float32)


def snr_map(x: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delta = mean_water - mean_no_water, pooled std, SNR over trailing dims."""
    x1, x0 = x[label == 1], x[label == 0]
    delta = x1.mean(axis=0) - x0.mean(axis=0)
    pooled = np.sqrt((x1.var(axis=0) + x0.var(axis=0)) / 2) + 1e-12
    return delta, pooled, np.abs(delta) / pooled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info, phase, a_demo = load()
    label = info["label"]
    n = len(label)
    logger.info("loaded %d frames; phase range [%d, %d]", n, phase.min(), phase.max())
    rep: dict = {"n_frames": n, "n_water": int((label == 1).sum()), "n_no_water": int((label == 0).sum())}

    # ------------------------------------------------------------------ #
    # Q1: class-conditional difference structure (pooled over all frames)  #
    # ------------------------------------------------------------------ #
    delta, pooled, snr = snr_map(a_demo, label)  # [50, 32] each
    step_norm = np.linalg.norm(delta, axis=1)  # [50]
    dim_norm = np.linalg.norm(delta, axis=0)  # [32]
    top_flat = np.dstack(np.unravel_index(np.argsort(snr, axis=None)[::-1][:15], snr.shape))[0]
    rep["q1_structure"] = {
        "delta_step_norm": step_norm.tolist(),
        "delta_dim_norm_top8": [
            {"dim": int(d), "norm": float(dim_norm[d])} for d in np.argsort(dim_norm)[::-1][:8]
        ],
        "snr_max": float(snr.max()),
        "snr_frac_gt_2": float((snr > SNR_KEEP).mean()),
        "snr_top15_step_dim": [
            {"step": int(s), "dim": int(d), "snr": float(snr[s, d]), "delta": float(delta[s, d])} for s, d in top_flat
        ],
        "dim9_snr_by_step": snr[:, RIGHT_TCP_X].tolist(),
    }
    logger.info(
        "Q1: SNR max %.2f, frac(SNR>2) %.4f; top dims %s",
        snr.max(),
        (snr > SNR_KEEP).mean(),
        [d["dim"] for d in rep["q1_structure"]["delta_dim_norm_top8"][:5]],
    )

    # ------------------------------------------------------------------ #
    # Q2: commitment point -- per-phase-bin separability of a_demo chunks   #
    # ------------------------------------------------------------------ #
    bins = []
    max_phase = int(phase.max())
    for lo in range(0, max_phase + 1, PHASE_BIN):
        m = (phase >= lo) & (phase < lo + PHASE_BIN)
        if m.sum() < 2 * MIN_PER_PHASE_CLASS or (label[m] == 1).sum() < MIN_PER_PHASE_CLASS or (label[m] == 0).sum() < MIN_PER_PHASE_CLASS:
            continue
        d, p, s = snr_map(a_demo[m], label[m])
        flat_snr = float(np.linalg.norm(d) / np.linalg.norm(p))
        d9 = float(abs(d[-1, RIGHT_TCP_X]) / p[-1, RIGHT_TCP_X])  # chunk-end tcp.x
        d9_all = float(np.linalg.norm(d[:, RIGHT_TCP_X]) / np.linalg.norm(p[:, RIGHT_TCP_X]))
        bins.append(
            {
                "phase_lo": lo,
                "n": int(m.sum()),
                "n_water": int((label[m] == 1).sum()),
                "chunk_snr": flat_snr,
                "dim9_end_snr": d9,
                "dim9_all_snr": d9_all,
            }
        )
    commit_05 = next((b["phase_lo"] for b in bins if b["chunk_snr"] >= 0.5), None)
    commit_10 = next((b["phase_lo"] for b in bins if b["chunk_snr"] >= 1.0), None)
    rep["q2_commitment"] = {
        "bin_width": PHASE_BIN,
        "bins": bins,
        "first_bin_chunk_snr_ge_0.5": commit_05,
        "first_bin_chunk_snr_ge_1.0": commit_10,
    }
    logger.info("Q2: %d phase bins; chunk SNR>=0.5 from phase %s, >=1.0 from %s", len(bins), commit_05, commit_10)

    # ------------------------------------------------------------------ #
    # Q3: phase distribution                                               #
    # ------------------------------------------------------------------ #
    uniq, counts = np.unique(phase, return_counts=True)
    per_class = {
        "water": np.bincount(phase[label == 1], minlength=uniq.max() + 1),
        "no_water": np.bincount(phase[label == 0], minlength=uniq.max() + 1),
    }
    rep["q3_phase"] = {
        "max_phase": max_phase,
        "n_distinct_phases": int(len(uniq)),
        "count_percentiles": {
            "p10": float(np.percentile(counts, 10)),
            "p50": float(np.percentile(counts, 50)),
            "p90": float(np.percentile(counts, 90)),
            "min": int(counts.min()),
            "max": int(counts.max()),
        },
        "phases_with_ge20_per_class": int(((per_class["water"] >= MIN_PER_PHASE_CLASS) & (per_class["no_water"] >= MIN_PER_PHASE_CLASS)).sum()),
        "frames_in_those_phases": int(
            np.isin(phase, np.nonzero((per_class["water"] >= MIN_PER_PHASE_CLASS) & (per_class["no_water"] >= MIN_PER_PHASE_CLASS))[0]).sum()
        ),
        "seg_len_percentiles": {
            "p10": float(np.percentile(info["seg_len"], 10)),
            "p50": float(np.percentile(info["seg_len"], 50)),
            "p90": float(np.percentile(info["seg_len"], 90)),
        },
    }
    logger.info(
        "Q3: %d distinct phases, count p50 %.0f; %d phases with >=%d/class covering %d frames",
        len(uniq),
        np.percentile(counts, 50),
        rep["q3_phase"]["phases_with_ge20_per_class"],
        MIN_PER_PHASE_CLASS,
        rep["q3_phase"]["frames_in_those_phases"],
    )

    # ------------------------------------------------------------------ #
    # Q4a: per-phase-bin Delta table (synthesis option A)                   #
    #                                                                       #
    # Exact phases alternate 241/79 samples (stride-2 parity), so an        #
    # exact-phase Delta is noisy for odd phases; the alignment unit is a    #
    # 10-frame bin (~1600 samples/bin), which makes the SNR>2 mask keep     #
    # the cells that matter (pooled per-frame SNR never reaches 2).         #
    # ------------------------------------------------------------------ #
    phase_bin = phase // PHASE_BIN
    n_bins = int(phase_bin.max()) + 1
    bin_ok_list = []
    for b in range(n_bins):
        m = phase_bin == b
        ok = (label[m] == 1).sum() >= MIN_PER_PHASE_CLASS and (label[m] == 0).sum() >= MIN_PER_PHASE_CLASS
        bin_ok_list.append(bool(ok))
    delta_bin = np.zeros((n_bins, a_demo.shape[1], a_demo.shape[2]), dtype=np.float32)
    mask_bin = np.zeros_like(delta_bin, dtype=bool)
    bin_ok = np.asarray(bin_ok_list, dtype=bool)
    per_bin_stats = []
    for b in range(n_bins):
        if not bin_ok[b]:
            continue
        m = phase_bin == b
        d, pl, s = snr_map(a_demo[m], label[m])
        keep = s > SNR_KEEP
        delta_bin[b] = np.where(keep, d, 0.0)
        mask_bin[b] = keep
        per_bin_stats.append(
            {"bin": int(b), "phase_lo": b * PHASE_BIN, "n": int(m.sum()), "frac_kept": float(keep.mean()),
             "delta_norm": float(np.linalg.norm(delta_bin[b])), "snr_max": float(s.max())}
        )
    delta_norms = np.asarray([s["delta_norm"] for s in per_bin_stats])
    # Eligibility for augmentation (see Q5 for the reasoning):
    #  - ||Delta|| >= MIN_DELTA_NORM: a zero/near-zero masked Delta would pair
    #    swapped tactile with an unchanged target, actively teaching the model
    #    to IGNORE the tactile swap (bin 0 and the mid-transport bins 10-19).
    #  - phase_lo < AUG_MAX_PHASE: past ~phase 190 (median segment length 184)
    #    the 50-frame chunk reaches into the release; a class flip there demands
    #    a motion that is physically inconsistent with the current state.
    stats_by_bin = {s["bin"]: s for s in per_bin_stats}
    eligible = np.asarray(
        [
            bool(
                bin_ok[b]
                and stats_by_bin.get(b, {}).get("delta_norm", 0.0) >= MIN_DELTA_NORM
                and b * PHASE_BIN < AUG_MAX_PHASE
            )
            for b in range(n_bins)
        ]
    )
    frames_augmentable = int(np.isin(phase_bin, np.nonzero(bin_ok)[0]).sum())
    frames_eligible = int(np.isin(phase_bin, np.nonzero(eligible)[0]).sum())
    rep["q4a_delta_table"] = {
        "bin_width": PHASE_BIN,
        "min_per_bin_class": MIN_PER_PHASE_CLASS,
        "snr_keep": SNR_KEEP,
        "min_delta_norm": MIN_DELTA_NORM,
        "aug_max_phase": AUG_MAX_PHASE,
        "n_valid_bins": int(bin_ok.sum()),
        "frames_in_valid_bins": frames_augmentable,
        "eligible_bins": [int(b) for b in np.nonzero(eligible)[0]],
        "frames_eligible": frames_eligible,
        "per_bin": per_bin_stats,
    }

    # ------------------------------------------------------------------ #
    # Q4b: same-episode phase-matched opposite-class chunks (option B)      #
    # ------------------------------------------------------------------ #
    ep = info["episode"]
    both_class_eps = [e for e in np.unique(ep) if len(np.unique(label[ep == e])) == 2]
    matched = 0
    for e in both_class_eps:
        m = ep == e
        ph, lb = phase[m], label[m]
        for c in (0, 1):
            pc, po = ph[lb == c], ph[lb != c]
            if len(pc) == 0 or len(po) == 0:
                continue
            d = np.abs(pc[:, None] - po[None, :]).min(axis=1)
            matched += int((d <= 1).sum())
    rep["q4b_episode_matched"] = {
        "episodes_with_both_classes": len(both_class_eps),
        "n_episodes_total": int(len(np.unique(ep))),
        "frames_with_match_pm1": matched,
        "frac_of_all_frames": matched / n,
    }
    logger.info(
        "Q4: option A valid bins %d/%d (covering %d frames, %d eligible); option B: %d/%d episodes both classes, %d frames (%.3f) phase-matched",
        int(bin_ok.sum()),
        n_bins,
        frames_augmentable,
        frames_eligible,
        len(both_class_eps),
        len(np.unique(ep)),
        matched,
        matched / n,
    )

    # ------------------------------------------------------------------ #
    # Q5: recommendation                                                   #
    # ------------------------------------------------------------------ #
    rep["q5_recommendation"] = {
        "chosen": "A (a_demo +/- Delta(10-frame phase bin), SNR>2-masked)",
        "reason": (
            "option A covers every frame in a valid phase bin with a state-consistent target "
            "(the correction is applied to the frame's own chunk); option B reaches only "
            f"{matched / n:.1%} of frames and reuses a chunk recorded at a different "
            "state (position/prefix mismatch baked into the target)."
        ),
        "counterfactual_fraction": 0.4,
        "restrict_to_pre_commitment": {
            "apply": False,
            "reason": (
                "the data shows NO pre-commitment window: the two classes' chunks are "
                "already separable at phase 0-9 (chunk SNR 0.53, dim9-end SNR 1.6) and "
                "divergence peaks at phase 40-49 -- the grasp itself is executed "
                "differently (weight). Restricting to 'pre-commitment' frames would "
                "exclude ~everything. Instead augmentation is limited to eligible bins "
                "(phase 10-99 with ||Delta||>=0.5): the SNR>2 mask zeroes the correction "
                "where classes do not diverge, near-zero-Delta bins are excluded because "
                "they would teach tactile-insensitivity (swapped tac, unchanged target), "
                "and phase>=100 is excluded because mid-transport there is nothing to "
                "teach (Delta~0) and release-adjacent frames (>=~190) would require "
                "physically state-inconsistent flip targets."
            ),
            "commitment_bin_snr_ge_0.5": commit_05,
            "commitment_bin_snr_ge_1.0": commit_10,
        },
        "eligible_bins": rep["q4a_delta_table"]["eligible_bins"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "delta_phase.npz",
        delta=delta_bin.astype(np.float16),
        mask=mask_bin,
        bin_ok=bin_ok,
        eligible=eligible,
        bin_width=np.asarray(PHASE_BIN, dtype=np.int64),
    )
    (OUT_DIR / "research.json").write_text(json.dumps(rep, indent=1))
    logger.info("wrote %s and delta_phase.npz", OUT_DIR / "research.json")


if __name__ == "__main__":
    main()
