"""Smoke tests for the attention-trace sampler (test/tactile_counterfactual/layer_probe.py).

Uses the dummy Gemma variant (8 heads, like the real gemma_300m) with a tiny depth;
the real depth=18 / heads=8 shape contract is asserted against the variant configs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.gemma as _gemma
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
from openpi.shared import nnx_utils
from test.tactile_counterfactual import layer_probe

NUM_STEPS = 10


@pytest.fixture(scope="module")
def setup():
    config = Pi0TactileFastVitConfig(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True)
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(42), (1, config.action_horizon, config.action_dim))
    sampler = layer_probe.AttentionSampler(model)
    return config, model, obs, noise, sampler


def test_trace_shapes(setup):
    config, model, obs, noise, sampler = setup
    depth = _gemma.get_config("dummy").depth

    trace = sampler(obs, noise=noise, num_steps=NUM_STEPS)

    assert trace.time.shape == (NUM_STEPS,)
    assert trace.x_t.shape == (NUM_STEPS, 1, config.action_horizon, config.action_dim)
    assert trace.v_t.shape == (NUM_STEPS, 1, config.action_horizon, config.action_dim)
    # attention: [num_steps, depth, B, heads, Q, K], Q = suffix, K = prefix + suffix.
    assert trace.attention.shape[0] == NUM_STEPS
    assert trace.attention.shape[1] == depth
    assert trace.attention.shape[2] == 1
    assert trace.attention.shape[3] == 8  # dummy has 8 heads, like gemma_300m
    assert trace.attention.shape[4] == trace.num_tactile + config.action_horizon
    assert trace.attention.shape[5] == trace.prefix_len + trace.attention.shape[4]
    assert trace.final_action.shape == (1, config.action_horizon, config.action_dim)
    # Layout metadata for token localisation.
    assert trace.num_tactile == 4
    assert trace.action_horizon == config.action_horizon
    assert trace.prefix_len > 0


def test_real_variant_shape_contract():
    # The smoke test runs the dummy variant; the production shape contract is depth=18,
    # heads=8 from gemma_300m.
    cfg = _gemma.get_config("gemma_300m")
    assert cfg.depth == 18
    assert cfg.num_heads == 8


def test_action_to_tactile_slice(setup):
    _, _, obs, noise, sampler = setup
    trace = sampler(obs, noise=noise, num_steps=NUM_STEPS)

    tac_attn = layer_probe.action_to_tactile_attention(trace)

    depth = trace.attention.shape[1]
    assert tac_attn.shape == (NUM_STEPS, depth, 1, 8, trace.action_horizon, trace.num_tactile)
    # Raw softmax probabilities: the slice must be exactly the corresponding block of the
    # full attention (no renormalisation), so each entry is <= the full row's mass.
    p, n, ah = trace.prefix_len, trace.num_tactile, trace.action_horizon
    assert jnp.array_equal(tac_attn, trace.attention[..., -ah:, p : p + n])
    assert jnp.all(tac_attn >= 0)
    assert jnp.all(tac_attn.sum(axis=-1) <= 1.0 + 1e-5)


def test_reproducible_with_fixed_noise(setup):
    _, _, obs, noise, sampler = setup

    trace_a = sampler(obs, noise=noise, num_steps=NUM_STEPS)
    trace_b = sampler(obs, noise=noise, num_steps=NUM_STEPS)

    assert jnp.array_equal(trace_a.attention, trace_b.attention)
    assert jnp.array_equal(trace_a.final_action, trace_b.final_action)


def test_no_nan(setup):
    _, _, obs, noise, sampler = setup
    trace = sampler(obs, noise=noise, num_steps=NUM_STEPS)

    assert jnp.all(jnp.isfinite(trace.attention))
    assert jnp.all(jnp.isfinite(trace.v_t))
    assert jnp.all(jnp.isfinite(trace.final_action))


def test_matches_production_sample_actions(setup):
    _, model, obs, noise, sampler = setup

    trace = sampler(obs, noise=noise, num_steps=NUM_STEPS)
    production = nnx_utils.module_jit(model.sample_actions)(
        jax.random.key(0), obs, num_steps=NUM_STEPS, noise=noise
    )

    max_diff = float(jnp.max(jnp.abs(trace.final_action - production)))
    np.testing.assert_allclose(
        np.asarray(trace.final_action),
        np.asarray(production),
        rtol=layer_probe.EQ_RTOL,
        atol=layer_probe.EQ_ATOL,
    )
    assert max_diff < layer_probe.EQ_ATOL + layer_probe.EQ_RTOL * float(jnp.max(jnp.abs(production)))
