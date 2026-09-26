"""Tests for the optional per-layer attention collection in the Gemma module."""

import jax
import jax.numpy as jnp

import openpi.models.gemma as _gemma


def _init_module(*variants: str, **module_kwargs):
    module = _gemma.Module(
        configs=[_gemma.get_config(v) for v in variants],
        embed_dtype="float32",
        **module_kwargs,
    )
    # The bridge (pi0) inits via init_with_output because Module defines a custom
    # `init` method that shadows the linen one.
    _, variables = module.init_with_output(jax.random.key(0), method="init", use_adarms=[False] * len(variants))
    return module, variables


def _causal_mask(batch, length):
    return jnp.broadcast_to(jnp.tril(jnp.ones((length, length), dtype=bool)), (batch, length, length))


def test_return_attention_false_returns_original_two_tuple():
    module, variables = _init_module("dummy")
    cfg = module.configs[0]
    batch, length = 2, 5
    embedded = [jnp.zeros((batch, length, cfg.width), dtype=jnp.float32)]
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))

    result = module.apply(variables, embedded, positions, _causal_mask(batch, length))

    assert len(result) == 2
    outputs, _ = result
    assert outputs[0].shape == (batch, length, cfg.width)


def test_return_attention_true_stacks_per_layer_and_preserves_outputs():
    module, variables = _init_module("dummy")
    cfg = module.configs[0]
    batch, length = 2, 5
    embedded = [jnp.zeros((batch, length, cfg.width), dtype=jnp.float32)]
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))
    mask = _causal_mask(batch, length)

    outputs_off, _ = module.apply(variables, embedded, positions, mask)
    outputs_on, _, attention, vnorms = module.apply(variables, embedded, positions, mask, return_attention=True)

    # [depth, B, H, T, S] with H = num_kv_heads * (num_heads / num_kv_heads) = num_heads.
    assert attention.shape == (cfg.depth, batch, cfg.num_heads, length, length)
    assert attention.dtype == jnp.float32
    # Per-key projected value norms: one [depth, B, H, S] stack per present expert.
    assert len(vnorms) == 1
    assert vnorms[0].shape == (cfg.depth, batch, cfg.num_heads, length)
    assert jnp.all(jnp.isfinite(vnorms[0])) and jnp.all(vnorms[0] >= 0)
    # Collecting attention must not change the model outputs.
    assert jnp.array_equal(outputs_off[0], outputs_on[0])


def test_two_expert_full_forward_attention():
    # Training-style forward: [prefix_tokens, suffix_tokens], queries and keys both
    # span the full concatenated sequence.
    module, variables = _init_module("dummy", "dummy")
    cfg = module.configs[0]
    batch, prefix_len, suffix_len = 2, 3, 2
    length = prefix_len + suffix_len
    embedded = [
        jnp.zeros((batch, prefix_len, cfg.width), dtype=jnp.float32),
        jnp.zeros((batch, suffix_len, cfg.width), dtype=jnp.float32),
    ]
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))

    outputs, _, attention, vnorms = module.apply(
        variables, embedded, positions, _causal_mask(batch, length), return_attention=True
    )

    assert outputs[0].shape == (batch, prefix_len, cfg.width)
    assert outputs[1].shape == (batch, suffix_len, cfg.width)
    assert attention.shape == (cfg.depth, batch, cfg.num_heads, length, length)
    # One value-norm stack per present expert, each [depth, B, H, S].
    assert len(vnorms) == 2
    for vn in vnorms:
        assert vn.shape == (cfg.depth, batch, cfg.num_heads, length)
        assert jnp.all(jnp.isfinite(vn))


def test_gemma_300m_has_8_attention_heads():
    cfg = _gemma.get_config("gemma_300m")
    assert cfg.num_heads == 8

    module, variables = _init_module("gemma_300m")
    batch, length = 1, 3
    embedded = [jnp.zeros((batch, length, cfg.width), dtype=jnp.float32)]
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))

    _, _, attention, _ = module.apply(variables, embedded, positions, _causal_mask(batch, length), return_attention=True)

    assert attention.shape == (cfg.depth, batch, 8, length, length)


def test_attention_rows_sum_to_one_without_nan():
    module, variables = _init_module("dummy")
    cfg = module.configs[0]
    batch, length = 2, 5
    embedded = [jnp.zeros((batch, length, cfg.width), dtype=jnp.float32)]
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))
    # Causal mask, plus one fully masked query row (as padding produces).
    mask = _causal_mask(batch, length)
    mask = mask.at[0, -1, :].set(False)

    _, _, attention, _ = module.apply(variables, embedded, positions, mask, return_attention=True)

    assert jnp.all(jnp.isfinite(attention))
    assert jnp.allclose(attention.sum(axis=-1), jnp.ones_like(attention[..., 0]), atol=1e-5)


def test_kv_cache_attention_is_rectangular():
    # Inference-style forward: queries are the suffix only, keys/values are the
    # cached prefix plus the suffix, so T != S.
    module, variables = _init_module("dummy")
    cfg = module.configs[0]
    batch, prefix_len, suffix_len = 2, 3, 2

    prefix_embedded = [jnp.zeros((batch, prefix_len, cfg.width), dtype=jnp.float32)]
    prefix_positions = jnp.broadcast_to(jnp.arange(prefix_len), (batch, prefix_len))
    _, kv_cache = module.apply(
        variables, prefix_embedded, prefix_positions, _causal_mask(batch, prefix_len)
    )

    suffix_embedded = [jnp.zeros((batch, suffix_len, cfg.width), dtype=jnp.float32)]
    suffix_positions = jnp.broadcast_to(jnp.arange(prefix_len, prefix_len + suffix_len), (batch, suffix_len))
    suffix_mask = jnp.ones((batch, suffix_len, prefix_len + suffix_len), dtype=bool)

    outputs, _, attention, vnorms = module.apply(
        variables, suffix_embedded, suffix_positions, suffix_mask, kv_cache=kv_cache, return_attention=True
    )

    assert outputs[0].shape == (batch, suffix_len, cfg.width)
    assert attention.shape == (cfg.depth, batch, cfg.num_heads, suffix_len, prefix_len + suffix_len)
    assert jnp.all(jnp.isfinite(attention))
    assert jnp.allclose(attention.sum(axis=-1), jnp.ones_like(attention[..., 0]), atol=1e-5)
    # Value norms cover the full key axis (cached prefix + suffix).
    assert len(vnorms) == 1
    assert vnorms[0].shape == (cfg.depth, batch, cfg.num_heads, prefix_len + suffix_len)
    assert jnp.all(jnp.isfinite(vnorms[0]))
    assert jnp.all(vnorms[0] >= 0)


def test_collect_attention_forces_explicit_path_under_cudnn_flag():
    # The cuDNN fused kernel cannot expose probabilities, so return_attention=True
    # must run the explicit path even when the module is configured for cuDNN.
    # Init with the flag off: Module.init's T=1 dummy forward is below the fused
    # kernel's minimum sequence length, and the flag does not affect parameters.
    _, variables = _init_module("dummy")
    cfg = _gemma.get_config("dummy")
    module = _gemma.Module(configs=[cfg], embed_dtype="float32", use_cudnn_attention=True)
    batch, length = 2, 5
    embedded = [jnp.zeros((batch, length, cfg.width), dtype=jnp.float32)]
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))

    _, _, attention, _ = module.apply(variables, embedded, positions, _causal_mask(batch, length), return_attention=True)

    # Had the fused path run, the per-layer probabilities would be None.
    assert attention is not None
    assert attention.shape == (cfg.depth, batch, cfg.num_heads, length, length)
    assert jnp.all(jnp.isfinite(attention))
