"""Verify the TactileRefinerPolicy serving wrapper without a real robot.

Simulates the serving loop with frames from the Xense/bottle-sorting-0810
dataset, building observation dicts in EXACTLY the format the
examples/bi_flexiv_rizon4_rt client sends (state[20] + CHW uint8 images with
client camera keys, tactile views fit_square'd to 224 like env.py does, no
prompt -- the server injects the default).

Checks:
(a) action-space round trip: absolute -> (DeltaActions+Normalize) -> refiner
    space -> (Unnormalize+AbsoluteActions) -> absolute is identity without the
    refiner delta; and with the gripper OPEN the wrapper output is bit-identical
    to the inner policy (same obs, same fixed noise).
(b) gating: with the gripper closed the wrapper output differs from the inner
    policy, stays finite and in range, and carries the `refiner` info dict; a
    dim_mask=[9] wrapper only moves dim 9.
(c) reset() clears the tactile history and the direction lock.
(d) a run of consecutive grasp frames infers without NaN; the direction lock
    engages once and never flips.

Run:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        test/tactile_counterfactual/tactile_refiner_policy_test.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from xense_client import image_tools  # noqa: E402

from openpi import transforms as _transforms  # noqa: E402
from openpi.policies import tactile_refiner_policy as _trp  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from scripts import serve_tactile_refined_policy as srv  # noqa: E402
from test.tactile_counterfactual import runner as _runner  # noqa: E402
from test.tactile_counterfactual.dataset_index import ProbeDataset  # noqa: E402

logger = logging.getLogger("tactile_refiner_policy_test")

EPISODE = 0
N_OPEN = 2
N_CLOSED = 8

# client camera key -> lerobot column candidates (old _0/_1 and new _left/_right spellings)
_CAMERA_COLUMNS = {
    "head": ("observation.images.head",),
    "left_wrist": ("observation.images.left_wrist",),
    "right_wrist": ("observation.images.right_wrist",),
    "left_tactile_top": ("observation.images.left_tactile_0", "observation.images.left_tactile_left"),
    "left_tactile_bottom": ("observation.images.left_tactile_1", "observation.images.left_tactile_right"),
    "right_tactile_top": ("observation.images.right_tactile_0", "observation.images.right_tactile_left"),
    "right_tactile_bottom": ("observation.images.right_tactile_1", "observation.images.right_tactile_right"),
}


def build_client_obs(dataset: ProbeDataset, ep: int, fr: int) -> dict:
    """Raw lerobot item -> obs dict in the exact format the rt client sends."""
    row = dataset._ep_frame_to_row[(ep, fr)]  # noqa: SLF001 (own infra)
    item = dataset._raw_dataset[row]  # noqa: SLF001
    images = {}
    for client_key, candidates in _CAMERA_COLUMNS.items():
        col = next((c for c in candidates if c in item), None)
        if col is None:
            raise KeyError(f"none of {candidates} in raw item keys {sorted(item)}")
        img = np.asarray(item[col])  # CHW (uint8 or float [0,1])
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).round().astype(np.uint8)
        hwc = np.moveaxis(img, 0, -1)
        if "tactile" in client_key:
            # examples/bi_flexiv_rizon4_rt/env.py: tactile goes through fit_square
            # with the run config's resize mode (center_crop is the client default).
            hwc = _transforms.fit_square(hwc, 224, "center_crop")
        else:
            hwc = np.asarray(image_tools.resize_with_pad(hwc[None], 224, 224)[0])
        images[client_key] = np.moveaxis(hwc, -1, 0)  # back to CHW, as the client sends
    return {
        "state": np.asarray(item["observation.state"], dtype=np.float32),
        "images": images,
    }


def pick_frames(dataset: ProbeDataset, ep: int) -> tuple[list[int], list[int]]:
    """Open-gripper frames at the episode start + a consecutive closed-gripper run."""
    hf = dataset._raw_dataset.hf_dataset  # noqa: SLF001
    episodes = np.asarray(hf["episode_index"])
    rows = np.nonzero(episodes == ep)[0]
    grip = np.asarray(hf["observation.state"], dtype=np.float32)[rows, 19]
    open_frames = [int(i) for i in np.nonzero(grip >= 0.6)[0][:N_OPEN]]
    closed = np.nonzero(grip < 0.48)[0]
    if len(closed) < N_CLOSED:
        raise RuntimeError(f"episode {ep} has only {len(closed)} closed-gripper frames")
    start = int(closed[0])
    closed_frames = [start + k for k in range(N_CLOSED)]
    return open_frames, closed_frames


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    t0 = time.time()

    args = srv.Args()
    logger.info("building wrapped policy (inner VLA + refiner + FastViT)...")
    policy = srv.create_policy(args)
    inner = policy._inner  # noqa: SLF001

    train_config = _config.get_config(args.config_name)
    data_config = _runner.resolve_data_config(train_config, pathlib.Path(args.checkpoint_dir))
    dataset = ProbeDataset(
        data_config.repo_id,
        data_config,
        train_config.model.action_horizon,
        transform=False,
    )
    open_frames, closed_frames = pick_frames(dataset, EPISODE)
    logger.info("episode %d: open frames %s, closed frames %s", EPISODE, open_frames, closed_frames)

    noise = np.random.default_rng(0).normal(size=(50, 32)).astype(np.float32)
    report: dict = {}

    # ------------------------------------------------------------------ #
    # (a1) action-space round trip is identity without the refiner delta  #
    # ------------------------------------------------------------------ #
    obs0 = build_client_obs(dataset, EPISODE, open_frames[0])
    inner0 = inner.infer(obs0, noise=noise)
    a_abs = np.asarray(inner0["actions"], dtype=np.float32)
    state = np.asarray(obs0["state"], dtype=np.float32)
    rt = policy._from_refiner_space(  # noqa: SLF001
        policy._to_refiner_space({"state": state.copy(), "actions": a_abs.copy()})  # noqa: SLF001
    )
    roundtrip_err = float(np.max(np.abs(rt["actions"] - a_abs)))
    report["roundtrip_max_abs_err"] = roundtrip_err
    assert roundtrip_err < 1e-4, f"round trip not identity: {roundtrip_err}"
    logger.info("(a1) action-space round trip max|err| = %.3g", roundtrip_err)

    # ------------------------------------------------------------------ #
    # (a2) gripper open -> wrapper output bit-identical to inner          #
    # ------------------------------------------------------------------ #
    for fr in open_frames:
        obs = build_client_obs(dataset, EPISODE, fr)
        ref = inner.infer(obs, noise=noise)["actions"]
        out = policy.infer(obs, noise=noise)
        assert np.array_equal(out["actions"], ref), f"open-gripper frame {fr}: wrapper changed the output"
        assert "refiner" not in out, f"open-gripper frame {fr}: unexpected refiner info dict"
    report["passthrough_bit_identical_open_frames"] = len(open_frames)
    assert len(policy._tac_frames) == 0, "tactile history must stay empty while gripper is open"  # noqa: SLF001
    logger.info("(a2) %d open-gripper frames: wrapper output bit-identical to inner", len(open_frames))

    # ------------------------------------------------------------------ #
    # (d) consecutive closed-gripper frames: no NaN, direction lock holds #
    # ------------------------------------------------------------------ #
    deltas, signs, lat_ms = [], [], []
    for fr in closed_frames:
        obs = build_client_obs(dataset, EPISODE, fr)
        t = time.perf_counter()
        out = policy.infer(obs, noise=noise)
        lat_ms.append((time.perf_counter() - t) * 1000)
        a = np.asarray(out["actions"], dtype=np.float32)
        assert np.isfinite(a).all(), f"frame {fr}: non-finite actions"
        info = out["refiner"]
        deltas.append(info["delta_l2"])
        signs.append(info["direction_sign"])
        assert info["active"] and info["gripper"] < 0.48
    locked = [s for s in signs if s != 0]
    assert locked and all(s == locked[0] for s in locked), f"direction lock flipped: {signs}"
    report["closed_frames"] = {
        "n": len(closed_frames),
        "delta_l2_mean": float(np.mean(deltas)),
        "delta_l2_max": float(np.max(deltas)),
        "direction_signs": signs,
        "direction_suppressed_total": policy._n_direction_suppressed,  # noqa: SLF001
        "wrapped_infer_ms_median": float(np.median(lat_ms)),
    }
    logger.info(
        "(d) %d closed frames: delta_l2 mean %.4f max %.4f, signs %s, suppressed %d, infer %.0f ms",
        len(closed_frames),
        np.mean(deltas),
        np.max(deltas),
        signs,
        policy._n_direction_suppressed,  # noqa: SLF001
        np.median(lat_ms),
    )

    # ------------------------------------------------------------------ #
    # (b) gating: closed-gripper output differs from inner; dim_mask=[9]  #
    # ------------------------------------------------------------------ #
    fr = closed_frames[0]
    obs = build_client_obs(dataset, EPISODE, fr)
    ref = inner.infer(obs, noise=noise)["actions"]
    out = policy.infer(obs, noise=noise)["actions"]
    max_change = float(np.max(np.abs(out - ref)))
    assert max_change > 1e-4, "closed-gripper frame: wrapper did NOT change the output"
    report["gating_max_abs_change_vs_inner"] = max_change
    logger.info("(b) closed-gripper frame %d: max|wrapped-inner| = %.4f (> 0 as required)", fr, max_change)

    data_config_full = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.checkpoint_dir) / "assets", data_config_full.asset_id)
    masked = _trp.TactileRefinerPolicy(
        inner,
        refiner_params=args.refiner_params,
        fastvit_params=args.fastvit_params,
        norm_stats=norm_stats,
        use_quantile_norm=data_config_full.use_quantile_norm,
        delta_action_mask=_transforms.make_bool_mask(18, -1, -1),
        dim_mask=[9],
    )
    out_m = masked.infer(obs, noise=noise)["actions"]
    change = np.abs(out_m - ref).reshape(50, 20)
    moved = [int(d) for d in range(20) if change[:, d].max() > 1e-6]
    assert moved == [9], f"dim_mask=[9] moved dims {moved}"
    report["dim_mask_9_moved_dims"] = moved
    logger.info("(b) dim_mask=[9]: only dims %s moved", moved)

    # ------------------------------------------------------------------ #
    # (c) reset clears history + direction lock                           #
    # ------------------------------------------------------------------ #
    assert len(policy._tac_frames) > 0  # noqa: SLF001
    policy.reset()
    assert len(policy._tac_frames) == 0 and policy._locked_sign == 0  # noqa: SLF001
    report["reset_clears_state"] = True
    logger.info("(c) reset() cleared tactile history and direction lock")

    report["total_seconds"] = time.time() - t0
    logger.info("ALL CHECKS PASSED in %.1f s", report["total_seconds"])
    for k, v in report.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
