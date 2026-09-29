"""Serve a trained VLA policy wrapped with the tactile refiner (websocket server).

Same wire protocol as scripts/serve_policy.py -- the bi_flexiv_rizon4_rt client
connects unchanged. The wrapper (openpi.policies.tactile_refiner_policy) applies
the v1.1 tactile refiner on top of the inner policy during grasp segments
(right gripper closed) and passes the inner output through otherwise.

Example (real machine):

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ~/miniforge3/envs/lerobot-xense/bin/python \
        scripts/serve_tactile_refined_policy.py \
        --config-name pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100 \
        --checkpoint-dir checkpoints/59999 \
        --refiner-params outputs/refiner_v11/refiner_params.pkl \
        --fastvit-params checkpoints/params.safetensors \
        --port 8000
"""

import dataclasses
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import tactile_refiner_policy as _refiner_policy
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi import transforms as _transforms


@dataclasses.dataclass
class Args:
    # Training config name of the INNER VLA policy.
    config_name: str = "pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100"
    # Checkpoint directory of the inner VLA policy.
    checkpoint_dir: str = "checkpoints/59999"
    # Tactile refiner weights (nnx state + feature standardisation), produced by
    # test/tactile_counterfactual/train_refiner_v11.py.
    refiner_params: str = "outputs/refiner_v11/refiner_params.pkl"
    # Frozen ImageNet FastViT-T12 weights for the refiner's tactile features.
    # Deliberately NOT the in-model encoder of the VLA checkpoint.
    fastvit_params: str = "checkpoints/params.safetensors"

    # Refiner is active while right_gripper.pos < threshold (grasp segment).
    gripper_threshold: float = 0.48
    # If set, only these action dims (0-19) may be corrected; default all.
    dim_mask: list[int] | None = None
    # Lock the sign of the dim-9 (right_tcp.x) correction within a grasp.
    direction_lock: bool = True

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Optional default prompt override for the inner policy.
    default_prompt: str | None = None


def create_policy(args: Args) -> _refiner_policy.TactileRefinerPolicy:
    train_config = _config.get_config(args.config_name)
    # The in-model tactile encoder's weights are restored from the VLA
    # checkpoint; tactile_pretrained_path is only an ImageNet init for training.
    # Tolerate it pointing at a file that does not exist on the serving machine
    # (same fallback as test/tactile_counterfactual/runner.py:load_model).
    model_config = train_config.model
    pretrained = getattr(model_config, "tactile_pretrained_path", None)
    if pretrained is not None and not pathlib.Path(pretrained).expanduser().exists():
        logging.info("tactile_pretrained_path %s not present locally; using checkpoint weights", pretrained)
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(model_config, tactile_pretrained_path=None)
        )
    inner = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats for the refiner action-space round trip.")
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.checkpoint_dir) / "assets", data_config.asset_id)

    delta_mask = None
    if getattr(train_config.data, "use_delta_cartesian_actions", False):
        # Dual-arm Cartesian: 18 TCP dims delta + 2 gripper dims absolute
        # (LeRobotBiFlexivDataConfig layout).
        delta_mask = _transforms.make_bool_mask(18, -1, -1)

    return _refiner_policy.TactileRefinerPolicy(
        inner,
        refiner_params=args.refiner_params,
        fastvit_params=args.fastvit_params,
        norm_stats=norm_stats,
        use_quantile_norm=data_config.use_quantile_norm,
        delta_action_mask=delta_mask,
        gripper_threshold=args.gripper_threshold,
        dim_mask=args.dim_mask,
        direction_lock=args.direction_lock,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
