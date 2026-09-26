"""Offline per-frame water/no-water auxiliary labels for stage-A tactile training.

Reads the raw ``observation.state`` trajectories of the bottle-sorting dataset
(no video decoding) and applies the label rules from ``grasp_start_frames.yaml``:

    grasp:    right_gripper.pos < 0.48      (state dim 19, BiFlexiv layout)
    water:    release right_tcp.x >= 0.8    (state dim  9)
    no_water: release right_tcp.x <= 0.6

A grasp segment is a contiguous run of frames with the gripper closed
(< 0.48); its release is the first frame after the run ends. Frames inside a
segment inherit the segment label; frames outside every segment, or in a
segment whose release x falls in the ambiguous (0.6, 0.8) band, get -1.

Output: ``outputs/stage_a/aux_labels.npz`` with key ``ep<episode_index>`` ->
int8 per-frame labels, plus a validation report against the 320 annotated
samples in ``grasp_start_frames.yaml`` (yaml episode numbers are 1-based,
dataset episode_index is 0-based).

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/compute_aux_labels.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_aux_labels")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "grasp_start_frames.yaml"
OUT_DIR = REPO_ROOT / "outputs" / "stage_a"

# BiFlexiv state layout (src/openpi/training/config.py LeRobotBiFlexivDataConfig):
#   left_tcp 0-8, right_tcp 9-17, left_gripper 18, right_gripper 19
RIGHT_TCP_X = 9
RIGHT_GRIPPER_POS = 19

GRIPPER_CLOSED = 0.48
WATER_X = 0.8
NO_WATER_X = 0.6

# A closed-gripper run shorter than this is a gripper twitch, not a grasp.
MIN_SEGMENT_FRAMES = 30
# Closed runs separated by fewer open frames than this are one grasp (threshold jitter).
MAX_GAP_FRAMES = 15


def find_grasp_segments(gripper: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive)] closed-gripper segments, gap-merged and length-filtered."""
    closed = gripper < GRIPPER_CLOSED
    segments: list[tuple[int, int]] = []
    n = closed.shape[0]
    i = 0
    while i < n:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j < n and closed[j]:
            j += 1
        segments.append((i, j))
        i = j
    # Merge segments separated by tiny open gaps.
    merged: list[list[int]] = []
    for s, e in segments:
        if merged and s - merged[-1][1] <= MAX_GAP_FRAMES:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= MIN_SEGMENT_FRAMES]


def label_episode(states: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Per-frame int8 labels for one episode + segment list [(start, end, label)]."""
    n = states.shape[0]
    labels = np.full(n, -1, dtype=np.int8)
    segments = find_grasp_segments(states[:, RIGHT_GRIPPER_POS].astype(np.float64))
    out: list[tuple[int, int, int]] = []
    for start, end in segments:
        release = min(end, n - 1)  # first open frame; end-exclusive == first frame after the run
        x = float(states[release, RIGHT_TCP_X])
        if x >= WATER_X:
            seg_label = 1
        elif x <= NO_WATER_X:
            seg_label = 0
        else:
            seg_label = -1  # ambiguous release position
        labels[start:end] = seg_label
        out.append((start, end, seg_label))
    return labels, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Xense/bottle-sorting-0810")
    parser.add_argument("--out", default=str(OUT_DIR / "aux_labels.npz"))
    args = parser.parse_args()

    from lerobot.datasets import lerobot_dataset

    logger.info("Loading dataset %s (metadata + parquet only, no video decode)", args.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(args.repo_id)
    hf = dataset.hf_dataset
    states = np.asarray(hf["observation.state"], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    index = np.asarray(hf["index"], dtype=np.int64)
    logger.info("states %s, %d frames", states.shape, states.shape[0])

    ep_start: dict[int, int] = {}
    for ep in np.unique(episodes):
        ep_start[int(ep)] = int(index[episodes == ep].min())

    per_ep: dict[str, np.ndarray] = {}
    seg_report: dict[int, list[tuple[int, int, int]]] = {}
    for ep in sorted(ep_start):
        rows = episodes == ep
        ep_states = states[rows]
        labels, segs = label_episode(ep_states)
        per_ep[f"ep{ep}"] = labels
        seg_report[ep] = segs

    # ---- validation against grasp_start_frames.yaml -------------------------
    spec = yaml.safe_load(YAML_PATH.read_text())
    total = 0
    agree = 0
    mismatches: list[dict] = []
    for category, expected in (("water", 1), ("no_water", 0)):
        for sample in spec["categories"][category]["samples"]:
            ep = int(sample["episode"]) - 1  # yaml is 1-based
            frame = int(sample["frame"])
            got = int(per_ep[f"ep{ep}"][frame])
            total += 1
            if got == expected:
                agree += 1
            else:
                mismatches.append({"episode": ep, "frame": frame, "expected": expected, "got": got})
    logger.info("yaml validation: %d/%d agree (%.2f%%)", agree, total, 100.0 * agree / total)
    if mismatches:
        logger.warning("mismatches (first 20): %s", mismatches[:20])

    # ---- dataset-wide stats --------------------------------------------------
    all_labels = np.concatenate(list(per_ep.values()))
    n = all_labels.shape[0]
    n_water = int((all_labels == 1).sum())
    n_nowater = int((all_labels == 0).sum())
    n_unlabeled = int((all_labels == -1).sum())
    labeled = n_water + n_nowater
    logger.info(
        "frames: %d total | water %d (%.2f%%) | no_water %d (%.2f%%) | unlabeled/ambiguous %d (%.2f%%)",
        n,
        n_water,
        100.0 * n_water / n,
        n_nowater,
        100.0 * n_nowater / n,
        n_unlabeled,
        100.0 * n_unlabeled / n,
    )
    if labeled:
        logger.info(
            "within labeled frames: water %.3f / no_water %.3f (CE class weights ~ %.3f / %.3f)",
            n_water / labeled,
            n_nowater / labeled,
            labeled / (2.0 * n_water) if n_water else float("nan"),
            labeled / (2.0 * n_nowater) if n_nowater else float("nan"),
        )
    seg_counts = [len(s) for s in seg_report.values()]
    logger.info(
        "grasp segments: %d total over %d episodes (min %d, max %d per episode)",
        sum(seg_counts),
        len(seg_counts),
        min(seg_counts),
        max(seg_counts),
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **per_ep)
    logger.info("wrote %s (%d episodes)", out_path, len(per_ep))

    report = {
        "yaml_agreement": {"agree": agree, "total": total, "rate": agree / total, "mismatches": mismatches},
        "frame_stats": {
            "total": n,
            "water": n_water,
            "no_water": n_nowater,
            "unlabeled": n_unlabeled,
        },
        "segments": {str(ep): [(int(s), int(e), int(l)) for s, e, l in segs] for ep, segs in seg_report.items()},
    }
    report_path = out_path.with_suffix(".report.json")
    import json

    report_path.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", report_path)

    if agree / total < 0.95:
        raise SystemExit(f"label validation rate {agree / total:.3f} below 0.95; fix segment detection before training")


if __name__ == "__main__":
    main()
