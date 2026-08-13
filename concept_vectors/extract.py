"""Extracting concept directions from minimal prompt pairs.

The principle: two prompts **identical but for one proposition**, so the difference in
activations encodes only what differs. This is the central precaution — a "prompt versus
no prompt" contrast produces a vector that mixes the target concept with length, register,
and everything else the prompt adds.

Three controls are provided, each born from a measurement that looked good and wasn't:

  held-out separability   Fitting and testing on the same pairs gives 100% separation all
                          the way down to layer 0, before any concept is formed. That is a
                          tautology, not a measurement.
  cosine matrix           Two "distinct" concepts can share most of their direction.
                          Without it you believe you are manipulating one concept while
                          manipulating another.
  natural amplitude       The vector's norm before normalisation is what the prompt
                          actually displaces. An alpha in arbitrary units is comparable to
                          nothing.
"""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

import mlx.core as mx

from .tap import ResidualTap


def _chat(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        return f"{system}\n\n{user}\n"


def direction(model, tokenizer, base: str, positive: str, negative: str,
              queries: list[str]) -> dict[int, mx.array]:
    """Difference of means between the two branches of a minimal pair."""
    sums: dict[int, mx.array] = {}
    with ResidualTap(model) as tap:
        tap.selftest(model, tokenizer)
        for q in queries:
            a = tap.last_hidden(model, tokenizer, _chat(tokenizer, base + "\n\n" + positive, q))
            b = tap.last_hidden(model, tokenizer, _chat(tokenizer, base + "\n\n" + negative, q))
            for li in a:
                if li in b:
                    d = a[li] - b[li]
                    sums[li] = d if li not in sums else sums[li] + d
    return {li: v / len(queries) for li, v in sums.items()}


def separability(model, tokenizer, base: str, positive: str, negative: str,
                 test_queries: list[str], dirs: dict[int, mx.array]) -> dict[int, float]:
    """Fraction of **held-out** queries where the projection ranks the positive branch
    above the negative one. 0.5 means no separation at all.

    The depth profile is more informative than the peak value: strong separability as
    early as layer 0 means the contrast is readable in the token embeddings themselves,
    so it is lexical rather than conceptual.
    """
    ok = {li: 0 for li in dirs}
    with ResidualTap(model) as tap:
        tap.selftest(model, tokenizer)
        for q in test_queries:
            a = tap.last_hidden(model, tokenizer, _chat(tokenizer, base + "\n\n" + positive, q))
            b = tap.last_hidden(model, tokenizer, _chat(tokenizer, base + "\n\n" + negative, q))
            for li, v in dirs.items():
                if li in a and li in b:
                    if float(((a[li] - b[li]) * v).sum().item()) > 0:
                        ok[li] += 1
    n = max(1, len(test_queries))
    return {li: c / n for li, c in ok.items()}


def extract_all(model, tokenizer, spec: dict, layer: int, n_fit: int,
                n_test: int) -> dict:
    """Extract one direction per concept, plus the whole-set diagnostics."""
    base = spec["base_prompt"]
    qs = spec["queries"]
    fit, test = qs[:n_fit], qs[n_fit:n_fit + n_test]
    if not test:
        raise ValueError("not enough queries: some are needed for fitting AND some, "
                         "disjoint, for held-out testing")

    vecs, norms, seps = {}, {}, {}
    for c in spec["concepts"]:
        d = direction(model, tokenizer, base, c["positive"], c["negative"], fit)
        v = d[layer]
        norms[c["id"]] = float(mx.linalg.norm(v).item())
        vecs[c["id"]] = v / mx.linalg.norm(v)
        s = separability(model, tokenizer, base, c["positive"], c["negative"], test,
                         {layer: vecs[c["id"]]})
        seps[c["id"]] = s[layer]
        print(f"  {c['id']:<24} amplitude {norms[c['id']]:.4f}  "
              f"separability {seps[c['id']]:.3f}", flush=True)

    ids = [c["id"] for c in spec["concepts"]]
    # float32 is required: MLX's SVD rejects bfloat16, and cosines computed in bf16 came
    # out at 1.01 on the diagonal — fine for reading structure, not for decomposition.
    M = mx.stack([vecs[i] for i in ids]).astype(mx.float32)
    cos = {a: {b: round(float((vecs[a] * vecs[b]).sum().item()), 3) for b in ids}
           for a in ids}
    off = [cos[a][b] for a, b in itertools.combinations(ids, 2)]

    S = mx.linalg.svd(M, compute_uv=False, stream=mx.cpu)
    tot = float((S ** 2).sum().item())
    cum, rank = 0.0, len(ids)
    for i, s in enumerate(S.tolist()):
        cum += s * s
        if cum / tot >= 0.90:
            rank = i + 1
            break

    # Combined vector: mean of the unit directions, every phrasing weighted equally.
    # Weighting by norm would favour the wordings that move internal state the most,
    # which is a property of vocabulary rather than a criterion of correctness.
    mean = M.mean(axis=0)
    mean_u = mean / mx.linalg.norm(mean)
    nat = statistics.fmean([norms[i] * abs(float((vecs[i] * mean_u).sum().item()))
                            for i in ids])

    return {
        "layer": layer, "ids": ids, "vectors": vecs, "combined": mean_u,
        "report": {
            "amplitudes": norms, "separability": seps, "cosine": cos,
            "cosine_off_diagonal": {
                "mean": round(statistics.fmean(off), 3),
                "max": round(max(off), 3), "min": round(min(off), 3)},
            "singular_values": [round(float(x), 3) for x in S.tolist()],
            "effective_rank_90pct": rank,
            "combined_natural_amplitude": round(nat, 4),
            "suggested_alphas": [round(k * nat, 3) for k in (0.5, 1, 2, 3)],
        },
    }


def save(result: dict, out_dir: Path, n_layers: int) -> None:
    """Write the combined vector (padded to the model's layer count), the per-concept
    stack, and the diagnostics report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d = result["combined"].shape[0]
    layer = result["layer"]
    full = mx.concatenate([mx.zeros((layer, d)), result["combined"][None, :],
                           mx.zeros((n_layers - layer - 1, d))])
    mx.save(str(out_dir / "combined.npy"), full)
    mx.save(str(out_dir / "per_concept.npy"),
            mx.stack([result["vectors"][i] for i in result["ids"]]))
    (out_dir / "report.json").write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=1), encoding="utf-8")
