"""RLT (RL-token) config schema and YAML loading.

RLT has two clearly separated training phases, mirrored by the config blocks:

1. ``token_training:`` - offline supervised training of the RL-token
   encoder-decoder on the prefix hidden states of a *frozen* VLA (pi0/pi05)
   SFT checkpoint, computed over a LeRobot dataset. Produces the token
   checkpoint. (The VLA itself is never updated in this phase.)
2. ``rl:`` - online RL performed *during inference* on the real robot. Only
   the MLP policy/value heads are updated; the encoder-decoder is frozen.
   The algorithm defaults to the RL-token native TD variant (TD3-style);
   PPO is selectable via ``algorithm: {type: PPO}``. The training budget is
   counted in episodes, not gradient steps.

This module is deliberately self-contained: it does not import from
``openpi.training`` and its YAML files live under ``configs/rlt/``, which the
TrainConfig lookup never scans.

Example YAML (``configs/rlt/rlt_test.yaml``; the file stem is the config name):

    model:
      state_dim: 20
      action_dim: 20
      num_rl_tokens: 1
    token_training:
      vla_config: pi05_bi_flexiv_rizon4_rt
      vla_checkpoint: /path/to/sft/checkpoint
      repo_id: Xense/pick_up_cube_0713
      num_train_steps: 50000
    rl:
      algorithm:
        type: TD
      total_episodes: 500
      episodes_per_iteration: 10

Only ``rl.algorithm`` is polymorphic (a ``type:`` key resolved through
``ALGORITHMS``). ``model:``/``token_training:``/``rl:`` are fixed types, so
their blocks need no ``type:`` - the field's own annotation says which class
to build. Unknown keys fail loudly, matching the repo's "a typo fails the
parse" convention.
"""

from __future__ import annotations

import dataclasses
import difflib
import pathlib
import types
import typing
from typing import Any, Literal

from omegaconf import OmegaConf


@dataclasses.dataclass(frozen=True)
class RLTModelConfig:
    """Architecture of the RL-token encoder-decoder plus the MLP heads.

    Shared by both phases: phase one trains the encoder-decoder on the frozen
    VLA's prefix hidden states, phase two freezes it and trains only the MLP
    heads on top of the RL tokens.
    """

    # Robot state dimension (phase-two MLP heads).
    state_dim: int
    # Action dimension (phase-two MLP heads).
    action_dim: int
    # Number of RL tokens appended by the encoder; z_rl is num_rl_tokens * d_model.
    num_rl_tokens: int

    # Dimension of the VLA prefix hidden states fed into the encoder.
    input_dim: int = 2048
    # Maximum prefix sequence length (3 cameras x 256 SigLIP tokens = 768).
    prefix_seq_len: int = 768
    # If true, only the image segment of the prefix is encoded (language dropped).
    image_only: bool = True
    # If true, pass the prefix padding mask to the encoder/decoder and loss.
    use_mask: bool = False

    d_model: int = 2048
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 8192  # = 4 * d_model
    dropout: float = 0.0
    # Hidden layers of the MLP policy/value heads.
    mlp_hidden_dims: tuple[int, ...] = (512, 512)


@dataclasses.dataclass(frozen=True)
class TokenTrainingConfig:
    """Phase one: offline supervised training of the encoder-decoder."""

    # Name of an existing TrainConfig (configs/<name>.yaml) that defines the
    # frozen VLA's architecture and data pipeline.
    vla_config: str
    # Directory of the frozen VLA SFT checkpoint (PyTorch safetensors, as
    # produced by scripts/train_pytorch.py; convert JAX orbax checkpoints with
    # examples/convert_jax_model_to_pytorch.py first).
    vla_checkpoint: str
    # LeRobot repo id of the demonstration dataset (e.g. "Xense/pick_up_cube_0713").
    repo_id: str
    # Number of gradient steps.
    num_train_steps: int = 50_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    # Data loader workers.
    num_workers: int = 4


@dataclasses.dataclass(frozen=True)
class TDConfig:
    """RL-token native TD algorithm: twin Q networks with delayed policy updates (TD3-style)."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # Discount factor.
    gamma: float = 0.99
    # Target network soft-update coefficient.
    tau: float = 0.005
    # Target policy smoothing noise.
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    # Delayed policy/target update interval (in critic updates).
    policy_delay: int = 2
    batch_size: int = 256
    # Replay buffer capacity (in transitions).
    buffer_size: int = 100_000
    # Episodes of pure random exploration before learning starts.
    warmup_episodes: int = 10


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """Optional on-policy alternative to the native TD algorithm."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    value_loss_coef: float = 0.5
    max_grad_norm: float = 1.0
    # Epochs of reuse per collected batch.
    num_update_epochs: int = 4
    mini_batch_size: int = 256
    anneal_lr: bool = False


# Algorithm registry for the polymorphic `rl.algorithm` slot. Register new
# algorithms here to make them selectable via `algorithm: {type: <Name>}`.
ALGORITHMS: dict[str, type] = {
    "TD": TDConfig,
    "PPO": PPOConfig,
}


@dataclasses.dataclass(frozen=True)
class OnlineRLConfig:
    """Phase two: online RL during inference on the real robot.

    Single-robot environment - there is no parallel sampling. Only the MLP
    heads are updated; the encoder-decoder stays frozen.
    """

    # RL algorithm; select in YAML with `algorithm: {type: TD|PPO}`.
    algorithm: TDConfig | PPOConfig = dataclasses.field(default_factory=TDConfig)

    # Token checkpoint produced by phase one. If None, defaults to this
    # config's `token_checkpoint_dir`.
    token_checkpoint: str | None = None

    # Total online RL budget, in episodes.
    total_episodes: int = 500
    # Episodes collected before each update round.
    episodes_per_iteration: int = 10
    # Maximum steps per episode.
    max_episode_steps: int = 400
    # Exploration noise std for sampling; 0 = deterministic policy.
    exploration_noise_std: float = 0.1
    # Environment endpoint (e.g. "ws://host:port"); None = local environment.
    env_endpoint: str | None = None
    # Number of actions executed per inference call.
    action_horizon: int = 1
    record_video: bool = False


@dataclasses.dataclass(frozen=True)
class RLTConfig:
    # Name of the config. Injected from the YAML file stem; do not write it in the file.
    name: str

    # Architecture shared by both phases.
    model: RLTModelConfig
    # Phase one: offline supervised encoder-decoder training.
    token_training: TokenTrainingConfig
    # Phase two: online RL during inference.
    rl: OnlineRLConfig = dataclasses.field(default_factory=OnlineRLConfig)

    # Project name.
    project_name: str = "openpi-rlt"
    # Experiment name, supplied on the CLI. Names the checkpoint directories.
    exp_name: str = ""

    # Random seed.
    seed: int = 42
    # Training precision.
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    # Base directory for checkpoints of both phases.
    checkpoint_base_dir: str = "./checkpoints_rlt"
    # How often (in steps/episodes) to log metrics.
    log_interval: int = 10
    # How often to save checkpoints.
    save_interval: int = 50
    # If set, checkpoints matching step % keep_period == 0 are never deleted.
    keep_period: int | None = 500

    # If true, enable wandb logging.
    wandb_enabled: bool = True

    # If true, overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, resume from the last checkpoint.
    resume: bool = False

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Checkpoint directory for this run; shared root for both phases."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def token_checkpoint_dir(self) -> pathlib.Path:
        """Where phase one writes the token checkpoint and phase two loads it by default."""
        return self.checkpoint_dir / "token"

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# RLT YAML files live in configs/rlt/. The TrainConfig lookup only globs
# configs/*.yaml (non-recursive), so this directory never collides with it.
_YAML_SEARCH_DIR = pathlib.Path("configs") / "rlt"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def get_config(config_name: str) -> RLTConfig:
    """Get an RLT config by name, from ``configs/rlt/<name>.yaml``."""
    candidate = _repo_root() / _YAML_SEARCH_DIR / f"{config_name}.yaml"
    if candidate.is_file():
        return load(candidate)

    search_dir = _repo_root() / _YAML_SEARCH_DIR
    known = (
        sorted(p.stem for p in search_dir.glob("*.yaml") if not p.name.startswith("_")) if search_dir.is_dir() else []
    )
    closest = difflib.get_close_matches(config_name, known, n=1, cutoff=0.0)
    closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
    raise ValueError(f"RLT config '{config_name}' not found in {_YAML_SEARCH_DIR}.{closest_str}")


def load(yaml_path: pathlib.Path | str) -> RLTConfig:
    """Load an RLTConfig from a YAML file. The file stem becomes the config name."""
    path = pathlib.Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML config not found: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}: {path}")

    return _build_config(path.stem, raw)


def loads(yaml_text: str, name: str) -> RLTConfig:
    """Load an RLTConfig from a YAML string. Caller supplies the name."""
    raw = OmegaConf.to_container(OmegaConf.create(yaml_text), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return _build_config(name, raw)


def _build_config(name: str, raw: dict[str, Any]) -> RLTConfig:
    annotations = _field_annotations(RLTConfig)
    kwargs: dict[str, Any] = {"name": name}
    for key, value in raw.items():
        if key == "name":
            # Filename is the source of truth; ignore in-file name to avoid mismatch.
            continue
        if key not in annotations:
            raise ValueError(f"Unknown RLTConfig field '{key}'. Known: {sorted(annotations)}")
        kwargs[key] = _coerce(annotations[key], value, field=key)
    return RLTConfig(**kwargs)


def _field_annotations(cls: type) -> dict[str, Any]:
    """Field name -> resolved annotation."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:  # unresolvable forward refs shouldn't break loading
        hints = {}
    return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(cls)}


def _nested_dataclass(annotation: Any) -> type | None:
    """The dataclass a field holds, looking through Optional/Union. None for anything else."""
    if annotation is None or isinstance(annotation, str):
        return None
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if arg is type(None):
                continue
            if (found := _nested_dataclass(arg)) is not None:
                return found
        return None
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    return None


def _union_dataclasses(annotation: Any) -> list[type]:
    """All dataclass members of a Union annotation (the polymorphic case)."""
    if annotation is None or isinstance(annotation, str):
        return []
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType):
        return []
    return [arg for arg in typing.get_args(annotation) if isinstance(arg, type) and dataclasses.is_dataclass(arg)]


def _is_tuple_annotation(annotation: Any) -> bool:
    return annotation is tuple or typing.get_origin(annotation) is tuple


def _coerce(annotation: Any, value: Any, *, field: str) -> Any:
    """Turn a YAML value into whatever the annotated field expects."""
    if isinstance(value, dict):
        # A Union of dataclasses is a polymorphic slot: dispatch on the `type:` key.
        if options := _union_dataclasses(annotation):
            return _build_polymorphic(options, value, field=field)
        if (target := _nested_dataclass(annotation)) is not None:
            return _construct(target, value)
        return value
    if isinstance(value, list) and _is_tuple_annotation(annotation):
        return tuple(value)
    return value


def _build_polymorphic(options: list[type], spec: dict[str, Any], *, field: str) -> Any:
    """Instantiate the right class for a `type:`-tagged mapping."""
    if "type" not in spec:
        raise ValueError(f"Field '{field}' is missing required 'type:' key. Got keys: {sorted(spec.keys())}")
    type_name = spec["type"]
    # Accept both the registered short names (ALGORITHMS keys, e.g. TD) and
    # the plain class names (e.g. TDConfig).
    by_name = {cls.__name__: cls for cls in options}
    by_name.update({registered: cls for registered, cls in ALGORITHMS.items() if cls in options})
    if type_name not in by_name:
        raise KeyError(f"Unknown type '{type_name}' for field '{field}'. Known: {sorted(by_name)}")
    return _construct(by_name[type_name], {k: v for k, v in spec.items() if k != "type"})


def _construct(cls: type, body: dict[str, Any]) -> Any:
    """Instantiate `cls` from a YAML mapping, building nested dataclasses as it goes."""
    annotations = _field_annotations(cls)
    kwargs: dict[str, Any] = {}
    for key, value in body.items():
        if key not in annotations:
            raise ValueError(f"Unknown {cls.__name__} field '{key}'. Known: {sorted(annotations)}")
        kwargs[key] = _coerce(annotations[key], value, field=f"{cls.__name__}.{key}")
    return cls(**kwargs)
