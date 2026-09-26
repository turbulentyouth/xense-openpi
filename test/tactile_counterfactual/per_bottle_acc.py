"""Per-grasp-segment (per-bottle) tactile classification accuracy on the val set.

Loads the v1.1 GRU refiner (outputs/refiner_v11/refiner_params.pkl) and scores
its aux water/no-water head on every val frame (tac_history input, same
standardisation as training), then aggregates per grasp segment:

- majority vote and mean-softmax per segment vs the segment's true label;
- vote-share distribution (close calls vs landslides);
- misclassified segments listed (episode, position, length, share);
- segment-averaged single-frame accuracy (the no-voting baseline);
- first-50%-of-segment voting (early-decision scenario).

Output: outputs/refiner_v11/per_bottle_acc.json
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx  # noqa: E402

from test.tactile_counterfactual import train_refiner_v11 as tr11  # noqa: E402

logger = logging.getLogger("per_bottle_acc")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs" / "refiner_data" / "refiner_data.npz"
TAC_HIST = REPO_ROOT / "outputs" / "refiner_data" / "tac_history.npz"
PARAMS = REPO_ROOT / "outputs" / "refiner_v11" / "refiner_params.pkl"
ACCEPT = REPO_ROOT / "outputs" / "refiner_v11" / "acceptance.json"
OUT = REPO_ROOT / "outputs" / "refiner_v11" / "per_bottle_acc.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    z = np.load(DATA)
    zh = np.load(TAC_HIST)
    with open(PARAMS, "rb") as f:
        pkl = pickle.load(f)
    val_eps = set(json.loads(ACCEPT.read_text())["val_episodes"])
    assert pkl["config"]["history"] == "gru", pkl["config"]

    episode = z["episode"].astype(np.int64)
    frame = z["frame"].astype(np.int64)
    label = z["label"].astype(np.int64)
    is_val = np.asarray([ep in val_eps for ep in episode])
    tac_hist = (zh["tac_hist"].astype(np.float32) - pkl["tac_mean"]) / pkl["tac_std"]
    a_vla = z["a_vla"].astype(np.float32).reshape(len(label), -1)

    model = tr11.RefinerV11(rngs=nnx.Rngs(0), history="gru")
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pkl["state"])
    eval_fn = tr11.make_eval_fn(graphdef)
    _, logits = tr11.batched_predict(eval_fn, state, tac_hist[is_val], a_vla[is_val])
    ep_v, fr_v, lb_v = episode[is_val], frame[is_val], label[is_val]
    prob1 = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))  # softmax water prob
    pred = (prob1 > 0.5).astype(np.int64)
    logger.info("val frames %d, single-frame acc %.4f", len(lb_v), (pred == lb_v).mean())

    # Rebuild grasp segments from the aux-labels report (contiguous same-label runs).
    segs = {int(k): v for k, v in json.loads(tr11.SEGMENTS.read_text())["segments"].items()}
    seg_id = np.empty(len(lb_v), dtype=np.int64)  # index into segment list
    seg_list = []
    seg_key = {}
    for e in sorted(set(ep_v.tolist())):
        m = np.nonzero(ep_v == e)[0]
        starts = np.asarray([s[0] for s in segs[e]])
        ends = np.asarray([s[1] for s in segs[e]])
        idx = np.searchsorted(ends, fr_v[m], side="right")
        for row, si in zip(m, idx):
            s, en, sl = segs[e][si]
            assert sl == lb_v[row] and starts[si] <= fr_v[row] < en
            key = (e, s)
            if key not in seg_key:
                seg_key[key] = len(seg_list)
                seg_list.append({"episode": e, "start": int(s), "end": int(en), "label": int(sl)})
            seg_id[row] = seg_key[key]

    rows = []
    for sid, seg in enumerate(seg_list):
        m = seg_id == sid
        n = int(m.sum())
        p1 = prob1[m]
        pr = pred[m]
        true = seg["label"]
        vote_share = float(max(pr.sum(), n - pr.sum()) / n)
        vote_pred = int(pr.sum() * 2 > n)  # ties -> no_water
        mean_p1 = float(p1.mean())
        soft_pred = int(mean_p1 > 0.5)
        k = max(1, n // 2 + (n % 2))  # temporally first 50% of sampled frames
        pr_early = pr[:k]
        early_pred = int(pr_early.sum() * 2 > k)
        early_share = float(max(pr_early.sum(), k - pr_early.sum()) / k)
        rows.append(
            {
                **seg,
                "seg_len_frames": seg["end"] - seg["start"],
                "n_sampled": n,
                "frame_acc": float((pr == true).mean()),
                "vote_pred": vote_pred,
                "vote_share": vote_share,
                "vote_correct": bool(vote_pred == true),
                "soft_pred": soft_pred,
                "soft_conf": float(max(mean_p1, 1 - mean_p1)),
                "soft_correct": bool(soft_pred == true),
                "early_pred": early_pred,
                "early_share": early_share,
                "early_correct": bool(early_pred == true),
            }
        )

    def acc(key):
        return float(np.mean([r[key] for r in rows]))

    shares = np.asarray([r["vote_share"] for r in rows])
    wrong = [r for r in rows if not r["vote_correct"]]
    by_len = {}
    for r in rows:
        bucket = "short(<150)" if r["seg_len_frames"] < 150 else ("mid(150-250)" if r["seg_len_frames"] < 250 else "long(>=250)")
        by_len.setdefault(bucket, []).append(r["vote_correct"])
    out = {
        "model": "refiner v1.1 GRU aux head (outputs/refiner_v11/refiner_params.pkl)",
        "val_frames": int(len(lb_v)),
        "single_frame_acc_all_val": float((pred == lb_v).mean()),
        "n_segments": len(rows),
        "n_water_segments": int(sum(r["label"] == 1 for r in rows)),
        "vote_acc": acc("vote_correct"),
        "softmax_mean_acc": acc("soft_correct"),
        "early_first50pct_vote_acc": acc("early_correct"),
        "frame_acc_segment_mean": float(np.mean([r["frame_acc"] for r in rows])),
        "vote_share_percentiles": {
            "p10": float(np.percentile(shares, 10)),
            "p50": float(np.percentile(shares, 50)),
            "p90": float(np.percentile(shares, 90)),
        },
        "n_close_segments_share_lt_0.6": int((shares < 0.6).sum()),
        "n_landslide_segments_share_gt_0.9": int((shares > 0.9).sum()),
        "vote_acc_by_seg_len": {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in by_len.items()},
        "misclassified_segments": [
            {
                "episode": r["episode"],
                "start": r["start"],
                "end": r["end"],
                "seg_len_frames": r["seg_len_frames"],
                "n_sampled": r["n_sampled"],
                "true": r["label"],
                "vote_pred": r["vote_pred"],
                "vote_share": r["vote_share"],
                "soft_conf": r["soft_conf"],
                "frame_acc": r["frame_acc"],
                "early_correct": r["early_correct"],
            }
            for r in wrong
        ],
        "note": "no contact-frame marker in refiner_data; per-segment voting uses ALL sampled "
        "segment frames (stride 2); 'early' = temporally first 50% of the segment's sampled frames",
        "segments": rows,
    }
    OUT.write_text(json.dumps(out, indent=1))
    logger.info(
        "segments %d (water %d): vote acc %.4f, softmax acc %.4f, early acc %.4f, frame acc %.4f",
        len(rows),
        out["n_water_segments"],
        out["vote_acc"],
        out["softmax_mean_acc"],
        out["early_first50pct_vote_acc"],
        out["frame_acc_segment_mean"],
    )
    logger.info("vote share p10/p50/p90: %.3f/%.3f/%.3f; misclassified: %d", *out["vote_share_percentiles"].values(), len(wrong))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
