"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def _load(model_path: str):
    from mlx_lm import load
    return load(model_path)


def cmd_extract(a) -> int:
    from .extract import extract_all, save

    spec = json.loads(Path(a.concepts).read_text(encoding="utf-8"))
    model, tok = _load(a.model)
    n_layers = len(_layers(model))
    layer = a.layer if a.layer >= 0 else n_layers // 2
    print(f"{len(spec['concepts'])} concepts · layer {layer}/{n_layers} · "
          f"{a.fit} fit queries, {a.test} held-out")
    res = extract_all(model, tok, spec, layer, a.fit, a.test)
    r = res["report"]

    ids = res["ids"]
    print(f"\nCOSINE MATRIX\n{'':<24}" + "".join(f"{i[:7]:>9}" for i in ids))
    for x in ids:
        print(f"{x:<24}" + "".join(f"{r['cosine'][x][y]:>9.2f}" for y in ids))
    h = r["cosine_off_diagonal"]
    print(f"\noff-diagonal: mean {h['mean']:+.3f} "
          f"(min {h['min']:+.3f}, max {h['max']:+.3f})")
    print(f"effective rank at 90% variance: {r['effective_rank_90pct']}/{len(ids)}")
    print(f"natural amplitude of combined vector: "
          f"{r['combined_natural_amplitude']:.4f}")
    print(f"alphas to calibrate: {r['suggested_alphas']}")
    weak = [i for i, s in r["separability"].items() if s < 0.7]
    if weak:
        print(f"\n⚠️  separability < 0.7: {weak} — these concepts are not "
              f"reliably isolated")
    save(res, Path(a.out), n_layers)
    print(f"\n→ {a.out}/combined.npy · per_concept.npy · report.json")
    return 0


def cmd_probe(a) -> int:
    from .probe import run, verdict

    spec = json.loads(Path(a.concepts).read_text(encoding="utf-8"))
    model, tok = _load(a.model)
    vecs = mx.load(a.vectors)
    v = vecs[a.layer]
    v = v / mx.linalg.norm(v)
    conds = [("reference", None)] + [
        (f"α={al:g} λ={a.lam:g}", (a.layer, v, al, a.lam))
        for al in [float(x) for x in a.alphas.split(",")]]
    res = run(model, tok, spec, conds, a.max_tokens, a.seed)
    verd = verdict(res, "reference")

    print(f"\n{'condition':<18}{'drift':>9}{'length':>10}{'marker':>11}")
    for label, v_ in res.items():
        mk = v_.get("marker_ok")
        print(f"{label:<18}{v_['drift']:>9.2f}{v_['length']:>10.0f}"
              f"{('ok' if mk else 'lost') if mk is not None else '—':>11}"
              + ("" if label == "reference"
                 else ("  ✅" if verd[label]["clean"]
                       else "  ⚠️ " + ", ".join(verd[label]["issues"]))))
    if a.show:
        for label, v_ in res.items():
            print(f"\n── {label} ──")
            for q, t in zip(spec["control_tasks"], v_["textes"]):
                print(f"  › {q}\n    {t[:220]}")
    print("\nThe score ranks which texts to read; it does not replace reading them.")
    return 0


def _layers(model):
    for obj in (model, getattr(model, "language_model", None),
                getattr(model, "model", None)):
        if obj is not None and hasattr(obj, "layers"):
            return obj.layers
    raise RuntimeError("could not locate layers")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="calibrated-steering",
        description="Extract a concept direction from minimal prompt pairs, check "
                    "that you extracted what you think you did, and apply it at "
                    "inference without breaking the model.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract directions and their diagnostics")
    e.add_argument("--model", required=True, help="path or HF repo id")
    e.add_argument("--concepts", required=True, help="specification .json file")
    e.add_argument("--layer", type=int, default=-1, help="default: middle of the network")
    e.add_argument("--fit", type=int, default=30, help="queries used to build the direction")
    e.add_argument("--test", type=int, default=20,
                   help="held-out queries, disjoint from --fit. Fitting and testing on "
                        "the same data gives 100%% down to layer 0, which is a tautology "
                        "rather than a measurement")
    e.add_argument("--out", default="vectors")
    e.set_defaults(func=cmd_extract)

    b = sub.add_parser("probe", help="check that applying the vector does not break the model")
    b.add_argument("--model", required=True)
    b.add_argument("--concepts", required=True)
    b.add_argument("--vectors", required=True, help="path to the .npy")
    b.add_argument("--layer", type=int, required=True)
    b.add_argument("--alphas", default="0.5,1.0,2.0")
    b.add_argument("--lam", type=float, default=1.0,
                   help="1 = addition · 0 = replacement · in between = attenuation")
    b.add_argument("--max-tokens", type=int, default=90)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--show", action="store_true", help="print the generated texts")
    b.set_defaults(func=cmd_probe)

    a = p.parse_args()
    return a.func(a)
