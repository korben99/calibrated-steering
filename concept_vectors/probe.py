"""Degradation probe: does the intervention break the model?

Overreach is a property of the **generated text**, not of a preference between two
options. So it is measured by generation, on tasks unrelated to the concept being
manipulated.

Two lessons are encoded here, each paid for by a wrong measurement:

  symmetric detection   Steering can shorten responses **or** lengthen them, depending on
                        the direction and the mode. A threshold testing only one side
                        misses the other — a +225% length increase raised no alert.
  fine fact marker      A question whose correct answer carries a nuance a degraded model
                        flattens. This was the earliest signal observed: it flips before
                        length, before refusals, before any lexical score.

The score does not replace reading. It ranks the texts worth reading.
"""

from __future__ import annotations

import re
import statistics

import mlx.core as mx

from .tap import ResidualTap


def _score(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text, re.I)) for p in patterns)


def run(model, tokenizer, spec: dict, conditions: list[tuple[str, tuple | None]],
        max_tokens: int = 90, seed: int = 42) -> dict:
    """Generate on the control tasks under each condition.

    `conditions`: list of (label, edit) where edit is None or
    (from_layer, vector, alpha, lambda).
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    tasks = spec["control_tasks"]
    patterns = spec.get("drift_patterns", [])
    marker = spec.get("fact_marker")  # {"task_index": i, "must_contain": "…"}
    sampler = make_sampler(temp=spec.get("temperature", 0.7))

    res = {}
    with ResidualTap(model) as tap:
        tap.selftest(model, tokenizer)
        for label, edit in conditions:
            tap.edit = edit
            texts, scores, lengths = [], [], []
            for q in tasks:
                msgs = [{"role": "system", "content": spec.get("control_system", "")},
                        {"role": "user", "content": q}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt = q
                # Fixed seed: reproducible without falling back to greedy decoding,
                # which degenerates on some models.
                mx.random.seed(seed)
                out = []
                for r in stream_generate(model, tokenizer,
                                         prompt=tokenizer.encode(prompt),
                                         max_tokens=max_tokens, sampler=sampler):
                    out.append(r.text)
                t = "".join(out).strip()
                mx.clear_cache()
                texts.append(t)
                scores.append(_score(t, patterns))
                lengths.append(len(t))
            res[label] = {"texts": texts,
                          "drift": statistics.fmean(scores) if scores else 0.0,
                          "length": statistics.fmean(lengths) if lengths else 0.0}
            if marker:
                t = texts[marker["task_index"]]
                res[label]["marker_ok"] = marker["must_contain"].lower() in t.lower()
        tap.edit = None
    return res


def verdict(res: dict, ref: str, length_tol: float = 0.25) -> dict:
    """Compare each condition against the reference.

    Length is checked **in both directions**: steering shortens in some regimes and
    lengthens in others, and a one-sided threshold systematically misses one of the two
    families.
    """
    base = res[ref]
    out = {}
    for label, v in res.items():
        if label == ref:
            continue
        dl = (v["length"] - base["length"]) / max(base["length"], 1)
        issues = []
        if abs(dl) > length_tol:
            issues.append(f"length {dl * 100:+.0f}%")
        if v["drift"] > base["drift"]:
            issues.append(f"drift {v['drift'] - base['drift']:+.2f}/response")
        if base.get("marker_ok") and v.get("marker_ok") is False:
            issues.append("factual nuance lost")
        out[label] = {"length_delta": round(dl, 3), "issues": issues,
                      "clean": not issues}
    return out
