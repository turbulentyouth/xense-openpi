"""Render tactile-attention full-mode dumps with BertViz head_view (cross-attention).

Reads the ``full``-mode output of ``scripts/dump_tactile_attention.py``
(``attention_full.npy`` float16 memmap ``[N, num_steps, depth, heads, Q, K]``,
``labels.json`` with ``query_labels`` / ``key_labels``, ``meta.json``), selects one
(frame, denoise_step) sample, converts it to the BertViz cross-attention format
(list of per-layer ``torch.FloatTensor [1, heads, Q, K]``), runs a sample
conformance check against BertViz's own expectations, and renders an interactive
head-view HTML file via ``head_view(..., html_action='return')``.

This script is READ-ONLY with respect to BertViz: it only calls the public
``bertviz.head_view`` API and never modifies anything under ``/home/xi/projects/bertviz``.
It intentionally depends only on numpy/json/argparse + torch + bertviz (no
jax/openpi/lerobot) so it runs standalone in the ``bert`` conda env:

    /home/xi/miniforge3/envs/bert/bin/python scripts/view_tactile_bertviz.py \
        --input <dump_dir> --frame 0 --step 0 --layer 5

Notes:
- BertViz has no notion of denoise steps; the (frame, step) sample is selected
  before calling into BertViz (by design).
- ``--layer L`` renders only layer L via ``include_layers=[L]``, which also makes
  the resulting HTML much smaller for long sequences.
- ``--keys`` / ``--queries`` subset the attention matrix and labels BEFORE calling
  BertViz (BertViz itself cannot hide tokens). Values stay raw softmax
  probabilities over the FULL key set — rows of a filtered view no longer sum
  to 1, which is intended: the hidden keys still took their share.
- ``--pool-patches P`` mean-pools each image view's 16x16 patch grid into PxP
  blocks (values become per-block average probabilities).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

# Label prefix -> group name, for --keys/--queries filtering.
KEY_GROUPS = {
    "img_base": "IMG_BASE",
    "img_left_wrist": "IMG_LEFT_WRIST",
    "img_right_wrist": "IMG_RIGHT_WRIST",
    "prompt": "PROMPT",
    "tac": "TAC_",
    "act": "ACT_",
}
QUERY_GROUPS = {"tac": "TAC_", "act": "ACT_"}
_IMAGE_PATCHES_PER_SIDE = 16  # 256 patches per view, row-major


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="full-mode dump directory from dump_tactile_attention.py")
    p.add_argument("--frame", type=int, default=0, help="frame index (first axis of attention_full.npy)")
    p.add_argument("--step", type=int, default=0, help="denoise step index (second axis)")
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="render only this layer (passed as include_layers=[layer]); also selects it initially",
    )
    p.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=None,
        help="head indices to show initially (passed through to head_view's `heads` parameter)",
    )
    p.add_argument(
        "--keys",
        default=None,
        metavar="GROUPS",
        help="comma-separated key (right-column) groups to KEEP: "
        f"{','.join(KEY_GROUPS)} or all (default). E.g. --keys tac,act",
    )
    p.add_argument(
        "--queries",
        default=None,
        metavar="GROUPS",
        help="comma-separated query (left-column) groups to KEEP: "
        f"{','.join(QUERY_GROUPS)} or all (default). E.g. --queries act",
    )
    p.add_argument(
        "--pool-patches",
        type=int,
        default=None,
        metavar="P",
        help="mean-pool each image view's 16x16 patch grid into PxP blocks before rendering",
    )
    p.add_argument(
        "--renormalize",
        action="store_true",
        help="divide each query row by its sum over the VISIBLE keys, so visible rows "
        "sum to 1 and line strength reflects share among shown tokens (boosts faint "
        "TAC lines that lost most mass to hidden image/prompt keys)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        metavar="G",
        help="apply v**G to attention values after any renormalization; G<1 brightens "
        "weak connections (e.g. 0.3), G=1 is a no-op",
    )
    p.add_argument("--out", default=None, help="output HTML path (default: <input>/bertviz_frame{frame}_step{step}.html)")
    p.add_argument("--check-only", action="store_true", help="only run the conformance check, skip rendering")
    return p.parse_args()


def load_full_sample(
    input_dir: str | pathlib.Path,
    frame: int,
    step: int,
) -> tuple[np.ndarray, list[str], list[str], dict]:
    """Load one (frame, step) sample from a full-mode dump directory.

    Returns ``(attention, query_labels, key_labels, meta)`` where ``attention`` is a
    float32 ndarray of shape ``[depth, heads, Q, K]``.
    """
    input_dir = pathlib.Path(input_dir)
    attn_path = input_dir / "attention_full.npy"
    labels_path = input_dir / "labels.json"
    meta_path = input_dir / "meta.json"
    for path in (attn_path, labels_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing expected dump file: {path}")

    meta = json.loads(meta_path.read_text())
    labels = json.loads(labels_path.read_text())
    query_labels, key_labels = labels["query_labels"], labels["key_labels"]

    # Memory-map the big float16 array; slice first, then upcast the slice.
    attention_full = np.load(attn_path, mmap_mode="r")
    n_frames, num_steps = attention_full.shape[0], attention_full.shape[1]
    if not 0 <= frame < n_frames:
        raise IndexError(f"frame {frame} out of range [0, {n_frames})")
    if not 0 <= step < num_steps:
        raise IndexError(f"step {step} out of range [0, {num_steps})")
    attention = np.asarray(attention_full[frame, step]).astype(np.float32)  # [depth, heads, Q, K]
    return attention, query_labels, key_labels, meta


def pool_image_keys(
    attention: np.ndarray, key_labels: list[str], pool: int
) -> tuple[np.ndarray, list[str]]:
    """Mean-pool each image view's 16x16 patch grid on the K axis into PxP blocks.

    ``attention`` is ``[depth, heads, Q, K]``. Image key labels (``IMG_<VIEW>_i``,
    i row-major over the 16x16 grid) must be contiguous. Returns the reduced
    attention and matching labels; non-image keys pass through unchanged. Block
    labels are ``IMG_<VIEW>_r<rows>c<cols>`` with the covered patch range.
    """
    s = _IMAGE_PATCHES_PER_SIDE
    if s % pool != 0:
        raise ValueError(f"--pool-patches must divide {s}, got {pool}")
    blocks = s // pool
    prefixes = sorted({KEY_GROUPS[g] for g in ("img_base", "img_left_wrist", "img_right_wrist")})

    out_attn: list[np.ndarray] = []
    out_labels: list[str] = []
    i = 0
    while i < len(key_labels):
        label = key_labels[i]
        prefix = next((p for p in prefixes if label.startswith(p)), None)
        if prefix is None:
            out_attn.append(attention[..., i : i + 1])
            out_labels.append(label)
            i += 1
            continue
        n = s * s
        if [l.split("_")[-1] for l in key_labels[i : i + n]] != [f"{j:03d}" for j in range(n)]:
            raise ValueError(f"{prefix} keys are not a contiguous 0..{n - 1} run at index {i}")
        grid = attention[..., i : i + n].reshape(*attention.shape[:-1], s, s)
        pooled = grid.reshape(*attention.shape[:-1], blocks, pool, blocks, pool).mean(axis=(-3, -1))
        out_attn.append(pooled.reshape(*attention.shape[:-1], blocks * blocks))
        out_labels += [
            f"{prefix}_r{br * pool}-{br * pool + pool - 1}c{bc * pool}-{bc * pool + pool - 1}"
            for br in range(blocks)
            for bc in range(blocks)
        ]
        i += n
    return np.concatenate(out_attn, axis=-1), out_labels


def select_tokens(
    attention: np.ndarray,
    query_labels: list[str],
    key_labels: list[str],
    queries: str | None,
    keys: str | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Subset the Q/K axes (and labels) to the requested groups.

    Values are untouched raw probabilities; a filtered view's rows no longer sum
    to 1 because hidden keys still absorbed their share of the softmax mass.
    """
    if queries and queries != "all":
        prefixes = tuple(QUERY_GROUPS[g] for g in queries.split(","))
        idx = [i for i, l in enumerate(query_labels) if l.startswith(prefixes)]
        if not idx:
            raise ValueError(f"--queries {queries} matched no query labels")
        attention, query_labels = attention[..., idx, :], [query_labels[i] for i in idx]
    if keys and keys != "all":
        prefixes = tuple(KEY_GROUPS[g] for g in keys.split(","))
        idx = [i for i, l in enumerate(key_labels) if l.startswith(prefixes)]
        if not idx:
            raise ValueError(f"--keys {keys} matched no key labels")
        attention, key_labels = attention[..., idx], [key_labels[i] for i in idx]
    return attention, query_labels, key_labels


def to_bertviz_cross_attention(attention: np.ndarray) -> list:
    """Convert ``[depth, heads, Q, K]`` float32 attention to BertViz format.

    BertViz expects cross_attention as a list with one ``torch.FloatTensor`` of
    shape ``[1, heads, Q, K]`` per layer (batch size must be 1; see
    bertviz/util.py:10-13).
    """
    import torch

    return [torch.from_numpy(attention[layer][None].copy()).float() for layer in range(attention.shape[0])]


def check_bertviz_sample(cross_attention, encoder_tokens: list[str], decoder_tokens: list[str]) -> bool:
    """Validate a sample against BertViz's own cross-attention expectations.

    Mirrors the checks BertViz itself performs (or relies on) in the cross-attention
    path of ``head_view``:

    - bertviz/util.py:10-12   each per-layer tensor must have exactly 4 dims
    - bertviz/util.py:13      batch dim (dim 0) is squeezed, so it must be 1
    - bertviz/head_view.py:189-201  attention Q/K dims must equal
      len(decoder_tokens) / len(encoder_tokens)
    - bertviz/head_view.py:146-149  both token lists must be present for
      cross_attention
    - plus extra sanity checks BertViz does NOT do but this data should satisfy:
      float dtype, finite values, values in [0, 1] (range violation only warns —
      BertViz renders any float).

    Prints a per-item PASS/FAIL report and returns True iff all hard checks pass.
    """
    import torch

    results: list[tuple[str, bool, str]] = []  # (label, passed, detail)
    warnings: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> bool:
        results.append((label, bool(ok), detail))
        return bool(ok)

    ok = True

    # head_view iterates over the list (util.format_attention) — must be a non-empty
    # list/tuple of per-layer tensors.
    ok &= check(
        "cross_attention is a non-empty list/tuple",
        isinstance(cross_attention, (list, tuple)) and len(cross_attention) > 0,
        f"got {type(cross_attention).__name__}, len={len(cross_attention) if isinstance(cross_attention, (list, tuple)) else 'n/a'}",
    )
    if not ok:
        _print_report(results, warnings)
        return False

    all_tensors = all(isinstance(t, torch.Tensor) for t in cross_attention)
    ok &= check("every element is a torch.Tensor", all_tensors)
    if not all_tensors:
        _print_report(results, warnings)
        return False

    all_4d = all(t.dim() == 4 for t in cross_attention)
    ok &= check(
        "every layer tensor is 4-dim (util.py:10-12)",
        all_4d,
        f"dims={[tuple(t.shape) for t in cross_attention][:3]}{'...' if len(cross_attention) > 3 else ''}",
    )

    batch1 = all(t.shape[0] == 1 for t in cross_attention)
    ok &= check(
        "batch dim == 1 for every layer (util.py:13 squeezes dim 0)",
        batch1,
    )

    heads_set = {t.shape[1] for t in cross_attention}
    ok &= check(
        "head count consistent across layers",
        len(heads_set) == 1,
        f"head counts: {sorted(heads_set)}",
    )

    if all_4d and batch1:
        q_set = {t.shape[2] for t in cross_attention}
        k_set = {t.shape[3] for t in cross_attention}
        q, k = cross_attention[0].shape[2], cross_attention[0].shape[3]
        ok &= check("Q/K dims consistent across layers", len(q_set) == 1 and len(k_set) == 1,
                    f"Q set={sorted(q_set)}, K set={sorted(k_set)}")
        # head_view.py:189-201 (left_text == decoder_tokens rows, right_text ==
        # encoder_tokens cols for the 'Cross' filter).
        ok &= check(
            "Q == len(decoder_tokens) (head_view.py:189-195)",
            len(decoder_tokens) == q,
            f"len(decoder_tokens)={len(decoder_tokens)}, Q={q}",
        )
        ok &= check(
            "K == len(encoder_tokens) (head_view.py:196-201)",
            len(encoder_tokens) == k,
            f"len(encoder_tokens)={len(encoder_tokens)}, K={k}",
        )

    float_dtype = all(t.is_floating_point() for t in cross_attention)
    ok &= check("float dtype (float32 after conversion)", float_dtype,
                f"dtypes={sorted({str(t.dtype) for t in cross_attention})}")

    finite = all(bool(torch.isfinite(t).all()) for t in cross_attention)
    ok &= check("all values finite (no NaN/Inf)", finite)

    if finite:
        stacked_min = min(float(t.min()) for t in cross_attention)
        stacked_max = max(float(t.max()) for t in cross_attention)
        in_range = stacked_min >= 0.0 and stacked_max <= 1.0
        if not in_range:
            # BertViz does not validate values; our data is softmax probabilities so
            # this should hold, but treat out-of-range as a warning, not a failure.
            warnings.append(
                f"attention values out of [0, 1]: min={stacked_min:.6f}, max={stacked_max:.6f} "
                "(BertViz will still render, but the data is expected to be softmax probabilities)"
            )
        check("values in [0, 1] (softmax probabilities; warning only)", in_range,
              f"min={stacked_min:.6f}, max={stacked_max:.6f}")

    _print_report(results, warnings)
    return ok


def _print_report(results: list[tuple[str, bool, str]], warnings: list[str]) -> None:
    print("BertViz sample conformance check:")
    for label, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        suffix = f"  [{detail}]" if detail else ""
        print(f"  [{status}] {label}{suffix}")
    for warning in warnings:
        print(f"  [WARN] {warning}")


def main() -> None:
    args = parse_args()
    input_dir = pathlib.Path(args.input)

    attention, query_labels, key_labels, meta = load_full_sample(input_dir, args.frame, args.step)
    depth, heads, q, k = attention.shape
    print(
        f"loaded frame={args.frame} step={args.step}: attention [depth={depth}, heads={heads}, Q={q}, K={k}], "
        f"query_labels={len(query_labels)}, key_labels={len(key_labels)}"
    )

    if args.pool_patches is not None:
        attention, key_labels = pool_image_keys(attention, key_labels, args.pool_patches)
        print(f"pooled image patches by {args.pool_patches}x{args.pool_patches}: K={attention.shape[-1]}")
    for arg, groups in ((args.keys, KEY_GROUPS), (args.queries, QUERY_GROUPS)):
        if arg and arg != "all":
            unknown = set(arg.split(",")) - set(groups)
            if unknown:
                print(f"ERROR: unknown group(s) {sorted(unknown)}; choose from {sorted(groups)}", file=sys.stderr)
                sys.exit(1)
    attention, query_labels, key_labels = select_tokens(
        attention, query_labels, key_labels, args.queries, args.keys
    )
    if args.keys or args.queries:
        print(
            f"filtered view: Q={attention.shape[-2]} queries, K={attention.shape[-1]} keys "
            "(values are raw probabilities over the FULL key set; visible rows no longer sum to 1)"
        )
    if args.renormalize:
        row_sum = attention.sum(axis=-1, keepdims=True)
        attention = attention / np.maximum(row_sum, 1e-12)
        print("renormalized: each query row now sums to 1 over the visible keys")
    if args.gamma is not None:
        if args.gamma <= 0:
            print(f"ERROR: --gamma must be > 0, got {args.gamma}", file=sys.stderr)
            sys.exit(1)
        attention = attention ** args.gamma
        print(f"applied gamma={args.gamma} (values remapped as v**{args.gamma})")

    cross_attention = to_bertviz_cross_attention(attention)
    # BertViz cross-attention: keys are the "encoder" side, queries the "decoder" side.
    if not check_bertviz_sample(cross_attention, encoder_tokens=key_labels, decoder_tokens=query_labels):
        print("ERROR: sample failed BertViz conformance check", file=sys.stderr)
        sys.exit(1)

    if args.check_only:
        print("check-only mode: all hard checks passed, skipping render")
        return

    if args.layer is not None and not 0 <= args.layer < depth:
        print(f"ERROR: --layer {args.layer} out of range [0, {depth})", file=sys.stderr)
        sys.exit(1)

    from bertviz import head_view

    include_layers = [args.layer] if args.layer is not None else None
    html = head_view(
        cross_attention=cross_attention,
        encoder_tokens=key_labels,
        decoder_tokens=query_labels,
        layer=args.layer,
        heads=args.heads,
        include_layers=include_layers,
        html_action="return",
    )

    suffix = ""
    if args.pool_patches:
        suffix += f"_pool{args.pool_patches}"
    if args.queries and args.queries != "all":
        suffix += f"_q-{args.queries.replace(',', '_')}"
    if args.keys and args.keys != "all":
        suffix += f"_k-{args.keys.replace(',', '_')}"
    if args.renormalize:
        suffix += "_renorm"
    if args.gamma is not None:
        suffix += f"_gamma{args.gamma}"
    out_path = (
        pathlib.Path(args.out) if args.out else input_dir / f"bertviz_frame{args.frame}_step{args.step}{suffix}.html"
    )
    out_path.write_text(html.data)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
