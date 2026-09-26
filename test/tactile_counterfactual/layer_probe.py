"""Analysis-only attention tracing of the Pi0/Pi05 denoising trajectory.

NOTE: this file previously hosted the layer-wise tactile probes, removed in commit
7fd05d1 ("Remove tactile probes and LTP training"). It now hosts the attention-trace
sampler only; the dataset/label machinery is intentionally not restored.

``AttentionSampler`` replicates ``Pi0.sample_actions`` step for step -- same
observation preprocessing, prefix encoding, KV cache, suffix construction, timestep
conditioning and Euler/flow update -- but unrolls the denoising loop with
``jax.lax.scan`` (instead of the production ``jax.lax.while_loop``, which only keeps
the final carry) so every step's attention probabilities are recorded. The
production sampler is untouched and nothing here is on the deployment path.

Per-step attention comes from ``gemma.Module(..., return_attention=True)``: suffix
queries x (cached prefix + suffix) keys, raw softmax probabilities in float32, one
entry per transformer block. Collecting attention forces the explicit attention
path (the cuDNN fused kernel cannot expose probabilities).

``action_to_tactile_attention`` cuts the Action-query -> Tactile-key block out of a
trace. For the Pi0TactileFastVit (pi05) suffix layout

    [TAC_0, TAC_1, TAC_2, TAC_3, ACT_0, ..., ACT_49]

that is ``[num_steps, depth, B, heads, action_horizon, num_tactile]``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models import pi0 as _pi0
from openpi.models.pi0_tactile_fastvit import Pi0TactileFastVit
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
import openpi.training.config as _config
from test.tactile_counterfactual import runner as _runner
from test.tactile_counterfactual.dataset_index import ProbeDataset

logger = logging.getLogger("tactile_layer_probe")

# Tolerance for analysis-vs-production final-action equivalence. The trace runs the
# same op sequence as the production loop, but jits it as one `lax.scan` instead of
# a `lax.while_loop`, so XLA fuses the two differently. That reordering is invisible
# in exact arithmetic but moves the final action by ~3e-3 with bf16 activations and
# GPU TF32 matmuls (measured by the removed trace_sampler.py on
# pi05_base_bi_flexiv_bottle_sorting_0817_fastvit, 10 steps, |action| ~ 1). The
# tolerance sits above that numerical floor and far below any real logic error,
# which would shift the action by O(0.1-1).
EQ_RTOL = 1e-2
EQ_ATOL = 5e-3


@dataclasses.dataclass
class AttentionTrace:
    """Per-step record of one denoising trajectory. All arrays are float32.

    attention's query axis covers the suffix only; the key axis covers the cached
    prefix plus the suffix, so K - Q == prefix_len.

    Layout metadata (Python ints, known statically from array shapes):

    prefix_len     number of cached prefix tokens (image views + prompt)
    suffix_len     Q, the suffix length (num_tactile + action_horizon for pi05 tactile)
    num_tactile    number of tactile tokens at the front of the suffix (0 for base Pi0)
    action_horizon number of action tokens at the back of the suffix
    """

    time: jax.Array  # [num_steps] flow time *before* each update
    x_t: jax.Array  # [num_steps, B, action_horizon, action_dim] trajectory point fed to the network
    v_t: jax.Array  # [num_steps, B, action_horizon, action_dim]
    attention: jax.Array  # [num_steps, depth, B, heads, Q, K]
    final_action: jax.Array  # [B, action_horizon, action_dim]
    prefix_len: int
    suffix_len: int
    num_tactile: int
    action_horizon: int
    # Per-key projected value norms ||W_o[n] @ v[k]|| for the suffix expert, stacked
    # as [num_steps, depth, B, heads, K]. Multiply attention by this (and normalise
    # per query) for a value-aware influence measure; see
    # action_to_tactile_influence. None for traces recorded before this field existed.
    value_norm: jax.Array | None = None


# Flat RoPE position offset for drop_prefix rollouts, identical to
# train_stage_a.py's PREFIX_LEN (3 image views x 256 SigLIP tokens + 200 prompt
# slots). See the drop_prefix branch of AttentionSampler._fun for why this must
# be a constant rather than a per-sample valid-token count.
_DROP_PREFIX_POS_OFFSET = 968


class AttentionSampler:
    """Step-recording replica of ``Pi0.sample_actions`` with caller-fixed noise.

    The module parameters are frozen at construction time (``nnx.split``), the same
    way ``openpi.shared.nnx_utils.module_jit`` freezes them for the production
    policy. ``noise`` is a mandatory argument so that paired experiments are exactly
    reproducible. Batch size is read from the observation; nothing is specialised
    for B=1 (the per-step attention cost grows linearly with B).
    """

    def __init__(self, model: _pi0.Pi0) -> None:
        if not isinstance(model, _pi0.Pi0):
            raise TypeError(f"AttentionSampler expects a Pi0 model, got {type(model).__name__}")
        self._model = model
        self._num_tactile = int(getattr(model, "_num_tactile", 0))
        self._graphdef, self._state = nnx.split(model)
        self._fn = jax.jit(self._fun, static_argnames=("num_steps", "mask_act_to_prefix", "drop_prefix"))

    def _fun(
        self,
        state,
        obs: _model.Observation,
        noise: jax.Array,
        num_steps: int,
        mask_act_to_prefix: bool = False,
        drop_prefix: bool = False,
    ):
        module: _pi0.Pi0 = nnx.merge(self._graphdef, state)
        # Everything below mirrors Pi0.sample_actions line for line; the only
        # differences are return_attention=True and scan instead of while_loop.
        obs = module._preprocess_observation(None, obs, train=False)
        dt = -1.0 / num_steps
        batch_size = obs.state.shape[0]

        if drop_prefix:
            # Pure-tactile mode: the prefix is never computed -- no SigLIP image
            # encoding, no prompt embedding, no Gemma backbone forward, no KV
            # cache. This is strictly stronger than mask_act_to_prefix: that
            # option still runs the prefix and only cuts the ACT->prefix
            # attention edges, so TAC queries keep reading the prefix and relay
            # its content to the ACT queries. Here the prefix does not exist at
            # all, and the suffix-only mask below means even TAC queries attend
            # to suffix keys exclusively.
            #
            # Positions use the FLAT deployment block length 968 (= 3x256 image
            # tokens + 200 prompt slots), the same constant as
            # train_stage_a.py's drop-prefix forward (PREFIX_LEN). RoPE attention
            # logits depend only on position differences, so the absolute value
            # of a constant offset is mathematically inert for a suffix-only
            # forward -- but it is NOT numerically inert in bf16: measured on
            # ckpt59999_bf16, a +6 absolute offset alone (identical suffix
            # inputs) shifts the final action by rel_l2 ~0.0077. A per-sample
            # offset (e.g. valid-token count, 875 vs 869 for the swap samples)
            # would therefore inject an obs-dependent artifact of the same
            # magnitude as the tactile signal itself; the flat constant keeps
            # positions bit-identical across counterfactual conditions.
            kv_cache = None
        else:
            # First fill the KV cache with a forward pass of the prefix.
            prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(obs)
            prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = module.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def get_v_t(x_t, time):  # equivalent to denoise_step in PyTorch
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = module.embed_suffix(
                obs, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            if drop_prefix:
                full_attn_mask = suffix_attn_mask
                pos = _DROP_PREFIX_POS_OFFSET + jnp.cumsum(suffix_mask, axis=-1) - 1
                expected_k = suffix_tokens.shape[1]
            else:
                prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                if mask_act_to_prefix:
                    # Hide ALL prefix keys from the action-token queries (the last
                    # action_horizon suffix rows); tactile and every other query row
                    # keep their original prefix visibility. QK logits are unchanged;
                    # the softmax for ACT rows renormalises over suffix keys only.
                    act_row = jnp.arange(suffix_tokens.shape[1]) >= suffix_tokens.shape[1] - module.action_horizon
                    prefix_attn_mask_ = jnp.where(act_row[None, :, None], False, prefix_attn_mask_)
                full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
                pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                expected_k = prefix_tokens.shape[1] + suffix_tokens.shape[1]
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                expected_k,
            )

            (prefix_out, suffix_out), _, attention, value_norms = module.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=pos,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                return_attention=True,
            )
            assert prefix_out is None
            v_t = module.action_out_proj(suffix_out[:, -module.action_horizon :])
            # Only the suffix (action) expert ran, so there is exactly one entry:
            # per-key ||W_o v|| for the queries we analyse. [depth, B, heads, K]
            return v_t, attention, value_norms[0]

        def step(carry, _):
            x_t, time = carry
            v_t, attention, value_norm = get_v_t(x_t, time)
            # Record the trajectory point that produced v_t (pre-update).
            return (x_t + dt * v_t, time + dt), (time, x_t, v_t, attention, value_norm)

        (x_0, _), (times, x_ts, v_ts, attentions, value_norms) = jax.lax.scan(
            step, (noise, jnp.asarray(1.0, jnp.float32)), None, length=num_steps
        )
        return {
            "time": times,
            "x_t": x_ts,
            "v_t": v_ts,
            "attention": attentions,
            "value_norm": value_norms,
            "final_action": x_0,
        }

    def __call__(
        self,
        obs: _model.Observation,
        *,
        noise,
        num_steps: int = 10,
        mask_act_to_prefix: bool = False,
        drop_prefix: bool = False,
    ) -> AttentionTrace:
        """Run one traced rollout. ``noise`` must be [B, action_horizon, action_dim].

        ``mask_act_to_prefix=True`` blocks action-token queries from attending to
        any cached prefix key (tactile queries are untouched), simulating a
        VLM-free suffix while keeping positions/KV cache/noise identical.

        ``drop_prefix=True`` goes further: the prefix is never computed at all
        (no SigLIP/prompt/Gemma prefix forward, no KV cache) and every suffix
        query -- tactile rows included -- attends to suffix keys only. The two
        options are mutually exclusive.
        """
        if mask_act_to_prefix and drop_prefix:
            raise ValueError(
                "mask_act_to_prefix and drop_prefix are mutually exclusive: the former keeps the "
                "prefix KV cache and only masks ACT query rows, the latter removes the prefix entirely"
            )
        noise = jnp.asarray(noise)
        expected = (obs.state.shape[0], self._model.action_horizon, self._model.action_dim)
        if noise.shape != expected:
            raise ValueError(f"noise shape {noise.shape} does not match expected {expected}")
        out = self._fn(
            self._state, obs, noise, num_steps=num_steps, mask_act_to_prefix=mask_act_to_prefix, drop_prefix=drop_prefix
        )
        attention = out["attention"]
        q_len, k_len = attention.shape[-2], attention.shape[-1]
        return AttentionTrace(
            time=out["time"],
            x_t=out["x_t"],
            v_t=out["v_t"],
            attention=attention,
            final_action=out["final_action"],
            prefix_len=k_len - q_len,
            suffix_len=q_len,
            num_tactile=self._num_tactile,
            action_horizon=self._model.action_horizon,
            value_norm=out["value_norm"],
        )


def action_to_tactile_attention(trace: AttentionTrace) -> jax.Array:
    """Cut the Action-query -> Tactile-key block out of a full trace.

    Returns ``[num_steps, depth, B, heads, action_horizon, num_tactile]`` of raw
    softmax probabilities -- NOT renormalised: each value is the probability mass
    the action query put on that tactile key within the full softmax over all
    prefix + suffix keys, so rows sum to at most 1.
    """
    if trace.num_tactile == 0:
        raise ValueError("this trace has no tactile tokens (num_tactile=0)")
    if trace.suffix_len != trace.num_tactile + trace.action_horizon:
        raise ValueError(
            f"expected suffix layout [TAC x {trace.num_tactile}, ACT x {trace.action_horizon}], "
            f"got suffix_len={trace.suffix_len}"
        )
    p, n, ah = trace.prefix_len, trace.num_tactile, trace.action_horizon
    return trace.attention[..., -ah:, p : p + n]


def action_to_tactile_influence(trace: AttentionTrace) -> jax.Array:
    """Value-aware Action-query -> Tactile-key influence share.

    For every (step, layer, head, action query) the keys' raw contribution
    magnitudes are ``contrib[q, k] = prob[q, k] * ||W_o v[k]||`` (the Q/K softmax
    times what the key actually injects through the output projection), and the
    returned share is ``contrib`` summed over the tactile keys divided by
    ``contrib`` summed over ALL keys. Every value is therefore dimensionless, in
    [0, 1], and directly comparable across steps, layers and heads -- unlike the
    raw probabilities of ``action_to_tactile_attention``, whose meaning drifts
    with each layer's value-norm scale.

    Returns ``[num_steps, depth, B, heads, action_horizon]``. The normalisation
    covers the attention output of this layer only; the residual stream and MLP
    are outside its scope.
    """
    if trace.value_norm is None:
        raise ValueError("this trace has no value_norm (recorded before value collection existed)")
    if trace.num_tactile == 0:
        raise ValueError("this trace has no tactile tokens (num_tactile=0)")
    if trace.suffix_len != trace.num_tactile + trace.action_horizon:
        raise ValueError(
            f"expected suffix layout [TAC x {trace.num_tactile}, ACT x {trace.action_horizon}], "
            f"got suffix_len={trace.suffix_len}"
        )
    p, n, ah = trace.prefix_len, trace.num_tactile, trace.action_horizon
    contrib = trace.attention * trace.value_norm[..., None, :]  # [steps, depth, B, heads, T, S]
    act = contrib[..., -ah:, :]  # action queries
    total = act.sum(axis=-1)
    return act[..., p : p + n].sum(axis=-1) / jnp.maximum(total, 1e-20)


# --------------------------------------------------------------------------- #
# Model / dataset loading (restored from the dc83417 probe layer)             #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ProbeSetup:
    config_name: str
    checkpoint_dir: pathlib.Path
    train_config: _config.TrainConfig
    data_config: _config.DataConfig
    model: Pi0TactileFastVit
    dataset: ProbeDataset

    @property
    def model_config(self) -> Pi0TactileFastVitConfig:
        return self.train_config.model  # type: ignore[return-value]

    @property
    def tactile_keys(self) -> tuple[str, ...]:
        return tuple(self.model_config.tactile_image_keys)


def load_setup(
    config_name: str,
    checkpoint_dir: str | pathlib.Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    root: str | None = None,
    cudnn_attention: bool = False,
) -> ProbeSetup:
    """Resolve the train config, norm stats, dataset and model for one checkpoint.

    cuDNN attention is off by default: attention collection forces the explicit
    path anyway, and the explicit path is the one every GPU runs identically.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    train_config = _config.get_config(config_name)
    model_config = train_config.model
    if not isinstance(model_config, Pi0TactileFastVitConfig):
        raise TypeError(
            f"{config_name} is a {type(model_config).__name__}; the attention dump needs Pi0TactileFastVitConfig"
        )
    if model_config.use_cudnn_attention != cudnn_attention:
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(model_config, use_cudnn_attention=cudnn_attention)
        )

    data_config = _runner.resolve_data_config(train_config, checkpoint_dir)  # type: ignore[arg-type]
    dataset = ProbeDataset(
        repo_id or data_config.repo_id,
        data_config,
        action_horizon=train_config.model.action_horizon,
        revision=revision,
        root=root,
    )
    model, _ = _runner.load_model(train_config, checkpoint_dir)
    return ProbeSetup(
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        train_config=train_config,
        data_config=data_config,
        model=model,  # type: ignore[arg-type]
        dataset=dataset,
    )


# --------------------------------------------------------------------------- #
# Frame sampling (restored from dc83417; the optional label-store              #
# stratification is duck-typed and unused by the attention dump)               #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FrameRefs:
    """The dump's sample set, in batch order: batch ``i`` is rows ``i*B:(i+1)*B``."""

    episode: np.ndarray  # int64 [N]
    frame: np.ndarray  # int64 [N]
    batch_size: int

    def __len__(self) -> int:
        return int(self.episode.shape[0])

    @property
    def num_batches(self) -> int:
        return (len(self) + self.batch_size - 1) // self.batch_size

    def batch(self, i: int) -> slice:
        return slice(i * self.batch_size, min((i + 1) * self.batch_size, len(self)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode.tolist(),
            "frame": self.frame.tolist(),
            "batch_size": self.batch_size,
        }


def sample_frames(
    dataset: ProbeDataset,
    *,
    num_frames: int,
    batch_size: int,
    rng: np.random.Generator,
    min_tail: int = 0,
    episodes: Sequence[int] | None = None,
) -> FrameRefs:
    """Draw ``num_frames`` frames such that every batch holds distinct episodes.

    ``min_tail`` keeps ``frame + min_tail < episode length`` (clean action window).
    """
    candidates = list(episodes) if episodes is not None else list(dataset.episodes)
    eligible = [ep for ep in candidates if dataset.episode_length(ep) > min_tail + 1]
    if not eligible:
        raise ValueError(f"no episode is longer than min_tail={min_tail}")
    if len(eligible) < batch_size:
        logger.warning(
            "only %d eligible episodes for batch_size=%d; batches will repeat episodes",
            len(eligible),
            batch_size,
        )

    def draw_frame(ep: int) -> int:
        return int(rng.integers(0, dataset.episode_length(ep) - min_tail))

    eps_out: list[int] = []
    frames_out: list[int] = []
    num_batches = (num_frames + batch_size - 1) // batch_size
    for b in range(num_batches):
        n_this = min(batch_size, num_frames - b * batch_size)
        order = rng.permutation(len(eligible))
        chosen = [eligible[order[i % len(eligible)]] for i in range(n_this)]
        for ep in chosen:
            frame = draw_frame(ep)
            if not dataset.has_sample(ep, frame):
                raise RuntimeError(f"(episode {ep}, frame {frame}) is not in the dataset index")
            eps_out.append(int(ep))
            frames_out.append(int(frame))

    logger.info("sampled %d frames over %d episodes", len(eps_out), len(set(eps_out)))
    return FrameRefs(
        episode=np.asarray(eps_out, dtype=np.int64),
        frame=np.asarray(frames_out, dtype=np.int64),
        batch_size=batch_size,
    )


# --------------------------------------------------------------------------- #
# Batch loading (restored from dc83417)                                       #
# --------------------------------------------------------------------------- #

_OBS_KEYS = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")


@dataclasses.dataclass
class Batch:
    index: np.ndarray  # positions in the FrameRefs
    episode: np.ndarray
    frame: np.ndarray
    observation: _model.Observation


class _FrameDataset:
    """torch map-style dataset over the sampled (episode, frame) pairs."""

    def __init__(self, dataset: ProbeDataset, episode: np.ndarray, frame: np.ndarray) -> None:
        self._dataset = dataset
        self._episode = episode
        self._frame = frame

    def __len__(self) -> int:
        return int(self._episode.shape[0])

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample = self._dataset.get_sample(int(self._episode[i]), int(self._frame[i]))
        out = {k: sample[k] for k in _OBS_KEYS if k in sample}
        out["_index"] = np.asarray(i, dtype=np.int64)
        return out


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)


def observation_from_batch(batch: dict[str, Any]) -> _model.Observation:
    data = {k: jax.tree.map(jnp.asarray, batch[k]) for k in _OBS_KEYS if k in batch}
    return _model.Observation.from_dict(data)


def iter_batches(
    dataset: ProbeDataset,
    refs: FrameRefs,
    *,
    num_workers: int = 0,
) -> Iterator[Batch]:
    """Decode the frames of ``refs`` batch by batch."""
    import multiprocessing

    import torch.utils.data as torch_data

    loader = torch_data.DataLoader(
        _FrameDataset(dataset, refs.episode, refs.frame),
        batch_size=refs.batch_size,
        shuffle=False,
        num_workers=num_workers,
        # Same choice as openpi.training.data_loader: JAX is multithreaded in the parent,
        # so forked workers may deadlock; spawned ones re-import and pickle the dataset.
        multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        collate_fn=_collate,
        drop_last=False,
    )
    for raw in loader:
        index = np.asarray(raw.pop("_index"))
        yield Batch(
            index=index,
            episode=refs.episode[index],
            frame=refs.frame[index],
            observation=observation_from_batch(raw),
        )


def fixed_noise(seed: int, num: int, action_horizon: int, action_dim: int) -> np.ndarray:
    """One noise chunk per sampled frame, shared by every experiment condition."""
    return np.asarray(jax.random.normal(jax.random.key(seed), (num, action_horizon, action_dim)), dtype=np.float32)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Token labels for full-mode dumps / BertViz                                  #
# --------------------------------------------------------------------------- #

# Order matches Pi0TactileFastVitConfig.inputs_spec / BiFlexiv observations.
_IMAGE_VIEWS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
# SigLIP So400m/14 at 224x224 -> 256 tokens per view.
_IMAGE_TOKENS_PER_VIEW = 256


def build_token_labels(
    prefix_len: int,
    num_tactile: int,
    action_horizon: int,
    max_token_len: int,
) -> tuple[list[str], list[str]]:
    """Query (suffix) and key (prefix + suffix) labels for one attention matrix.

    Suffix layout: [TAC_0..TAC_{n-1}, ACT_0..ACT_{ah-1}]. Prefix layout follows
    ``embed_prefix``: the image views first (``_IMAGE_TOKENS_PER_VIEW`` each, in
    ``_IMAGE_VIEWS`` order), then the prompt tokens. If the actual ``prefix_len``
    does not match that decomposition, prefix labels fall back to PREFIX_000...
    (no tokenizer/SigLIP internals are touched either way).
    """
    suffix_labels = [f"TAC_{i}" for i in range(num_tactile)] + [f"ACT_{i}" for i in range(action_horizon)]
    if prefix_len == len(_IMAGE_VIEWS) * _IMAGE_TOKENS_PER_VIEW + max_token_len:
        prefix_labels = [
            f"IMG_{view.removesuffix('_rgb').upper()}_{i:03d}"
            for view in _IMAGE_VIEWS
            for i in range(_IMAGE_TOKENS_PER_VIEW)
        ] + [f"PROMPT_{i:03d}" for i in range(max_token_len)]
    else:
        logger.warning(
            "prefix_len=%d does not match %d image views x %d + prompt %d; falling back to PREFIX_* labels",
            prefix_len,
            len(_IMAGE_VIEWS),
            _IMAGE_TOKENS_PER_VIEW,
            max_token_len,
        )
        prefix_labels = [f"PREFIX_{i:03d}" for i in range(prefix_len)]
    return suffix_labels, prefix_labels + suffix_labels
