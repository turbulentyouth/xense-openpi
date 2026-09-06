"""Tests for the RLT config schema and YAML loader."""

from __future__ import annotations

import pathlib

import pytest

import openpi.rlt.config as rlt_config

_FULL_YAML = """
model:
  state_dim: 20
  action_dim: 20
  num_rl_tokens: 8
  d_model: 1024
  input_dim: 1024
  prefix_seq_len: 512
  image_only: false
  use_mask: true
  mlp_hidden_dims: [512, 512]
token_training:
  vla_config: pi05_base_bi_flexiv_pick_up_cube_0824_h100
  vla_checkpoint: /path/to/sft/checkpoint
  repo_id: Xense/pick_up_cube_0713
  num_train_steps: 50000
rl:
  algorithm:
    type: TD
    actor_lr: 1e-4
  total_episodes: 300
  episodes_per_iteration: 5
  max_episode_steps: 200
  env_endpoint: ws://192.168.2.50:9100
seed: 7
"""

_MINIMAL_YAML = """
model:
  state_dim: 20
  action_dim: 20
  num_rl_tokens: 1
token_training:
  vla_config: pi05_base_bi_flexiv_pick_up_cube_0824_h100
  vla_checkpoint: /path/to/sft/checkpoint
  repo_id: Xense/pick_up_cube_0713
"""


def test_loads_full_yaml():
    config = rlt_config.loads(_FULL_YAML, name="full")

    assert config.name == "full"
    assert config.model.state_dim == 20
    assert config.model.d_model == 1024
    assert config.model.input_dim == 1024
    assert config.model.prefix_seq_len == 512
    assert config.model.image_only is False
    assert config.model.use_mask is True
    # YAML lists land in tuple-annotated fields as tuples.
    assert config.model.mlp_hidden_dims == (512, 512)
    assert config.token_training.vla_config == "pi05_base_bi_flexiv_pick_up_cube_0824_h100"
    assert config.token_training.vla_checkpoint == "/path/to/sft/checkpoint"
    assert config.token_training.repo_id == "Xense/pick_up_cube_0713"
    assert config.token_training.num_train_steps == 50_000
    assert isinstance(config.rl.algorithm, rlt_config.TDConfig)
    assert config.rl.algorithm.actor_lr == 1e-4
    assert config.rl.total_episodes == 300
    assert config.rl.episodes_per_iteration == 5
    assert config.rl.env_endpoint == "ws://192.168.2.50:9100"
    assert config.seed == 7


def test_loads_minimal_yaml_uses_defaults():
    config = rlt_config.loads(_MINIMAL_YAML, name="minimal")

    # Model defaults (aligned with the RLinf reference implementation).
    assert config.model.d_model == 2048
    assert config.model.dim_feedforward == 8192
    assert config.model.num_encoder_layers == 2
    assert config.model.num_decoder_layers == 2
    assert config.model.dropout == 0.0
    assert config.model.prefix_seq_len == 768
    assert config.model.image_only is True
    assert config.model.use_mask is False
    assert config.model.mlp_hidden_dims == (512, 512)
    # Phase-two block is optional; the native TD algorithm is the default.
    assert isinstance(config.rl.algorithm, rlt_config.TDConfig)
    assert config.rl.total_episodes == 500
    assert config.rl.env_endpoint is None


def test_algorithm_ppo_is_selectable():
    config = rlt_config.loads(_MINIMAL_YAML + "\nrl:\n  algorithm:\n    type: PPO\n    clip_ratio: 0.1\n", name="ppo")

    assert isinstance(config.rl.algorithm, rlt_config.PPOConfig)
    assert config.rl.algorithm.clip_ratio == 0.1


def test_unknown_algorithm_type_fails_loudly():
    with pytest.raises(KeyError, match=r"Unknown type 'SAC'.*Known"):
        rlt_config.loads(_MINIMAL_YAML + "\nrl:\n  algorithm:\n    type: SAC\n", name="bad")


def test_algorithm_missing_type_fails_loudly():
    with pytest.raises(ValueError, match="missing required 'type:'"):
        rlt_config.loads(_MINIMAL_YAML + "\nrl:\n  algorithm:\n    gamma: 0.9\n", name="bad")


def test_unknown_top_level_key_fails_loudly():
    with pytest.raises(ValueError, match="Unknown RLTConfig field 'num_train_step'"):
        rlt_config.loads(_MINIMAL_YAML + "\nnum_train_step: 10\n", name="bad")


def test_unknown_nested_key_fails_loudly():
    with pytest.raises(ValueError, match="Unknown TDConfig field 'actor_lrr'"):
        rlt_config.loads(_MINIMAL_YAML + "\nrl:\n  algorithm:\n    type: TD\n    actor_lrr: 1e-4\n", name="bad")


def test_in_file_name_is_ignored():
    config = rlt_config.loads(_MINIMAL_YAML + "\nname: something_else\n", name="from_file")
    assert config.name == "from_file"


def test_checkpoint_dir_requires_exp_name():
    config = rlt_config.loads(_MINIMAL_YAML, name="rlt_test")
    with pytest.raises(ValueError, match="--exp_name must be set"):
        _ = config.checkpoint_dir


def test_checkpoint_dirs():
    config = rlt_config.loads(_MINIMAL_YAML + "\nexp_name: run01\n", name="rlt_test")
    assert config.checkpoint_dir == (pathlib.Path("./checkpoints_rlt") / "rlt_test" / "run01").resolve()
    assert config.token_checkpoint_dir == config.checkpoint_dir / "token"


def test_resume_and_overwrite_are_mutually_exclusive():
    with pytest.raises(ValueError, match="Cannot resume and overwrite"):
        rlt_config.loads(_MINIMAL_YAML + "\nresume: true\noverwrite: true\n", name="bad")


def test_get_config_loads_shipped_example():
    """configs/rlt/rlt_test.yaml is the shared example; it must always parse."""
    config = rlt_config.get_config("rlt_test")
    assert config.name == "rlt_test"
    assert isinstance(config.rl.algorithm, rlt_config.TDConfig)


def test_get_config_unknown_name_suggests():
    with pytest.raises(ValueError, match="Did you mean 'rlt_test'"):
        rlt_config.get_config("rlt_tset")
