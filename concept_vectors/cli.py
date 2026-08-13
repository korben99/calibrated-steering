"""Interface en ligne de commande."""

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
    print(f"{len(spec['concepts'])} concepts · couche {layer}/{n_layers} · "
          f"{a.fit} requêtes d'ajustement, {a.test} de test")
    res = extract_all(model, tok, spec, layer, a.fit, a.test)
    r = res["report"]

    ids = res["ids"]
    print(f"\nMATRICE DE COSINUS\n{'':<24}" + "".join(f"{i[:7]:>9}" for i in ids))
    for x in ids:
        print(f"{x:<24}" + "".join(f"{r['cosinus'][x][y]:>9.2f}" for y in ids))
    h = r["cosinus_hors_diagonale"]
    print(f"\nhors diagonale : moyenne {h['moyenne']:+.3f} "
          f"(min {h['min']:+.3f}, max {h['max']:+.3f})")
    print(f"rang effectif à 90 % de variance : {r['rang_effectif_90pct']}/{len(ids)}")
    print(f"amplitude naturelle du vecteur combiné : "
          f"{r['amplitude_naturelle_combinee']:.4f}")
    print(f"alphas à calibrer : {r['alphas_suggeres']}")
    faibles = [i for i, s in r["separabilites"].items() if s < 0.7]
    if faibles:
        print(f"\n⚠️  séparabilité < 0,7 : {faibles} — ces concepts ne sont pas "
              f"isolés de façon fiable")
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
    conds = [("référence", None)] + [
        (f"α={al:g} λ={a.lam:g}", (a.layer, v, al, a.lam))
        for al in [float(x) for x in a.alphas.split(",")]]
    res = run(model, tok, spec, conds, a.max_tokens, a.seed)
    verd = verdict(res, "référence")

    print(f"\n{'condition':<18}{'dérive':>9}{'longueur':>10}{'marqueur':>11}")
    for label, v_ in res.items():
        mk = v_.get("marqueur_ok")
        print(f"{label:<18}{v_['derive']:>9.2f}{v_['longueur']:>10.0f}"
              f"{('ok' if mk else 'perdu') if mk is not None else '—':>11}"
              + ("" if label == "référence"
                 else ("  ✅" if verd[label]["propre"]
                       else "  ⚠️ " + ", ".join(verd[label]["problemes"]))))
    if a.show:
        for label, v_ in res.items():
            print(f"\n── {label} ──")
            for q, t in zip(spec["control_tasks"], v_["textes"]):
                print(f"  › {q}\n    {t[:220]}")
    print("\nLe score hiérarchise les textes à relire ; il ne remplace pas la lecture.")
    return 0


def _layers(model):
    for obj in (model, getattr(model, "language_model", None),
                getattr(model, "model", None)):
        if obj is not None and hasattr(obj, "layers"):
            return obj.layers
    raise RuntimeError("couches introuvables")


def main() -> int:
    p = argparse.ArgumentParser(prog="calibrated-steering", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extraire les directions et leurs diagnostics")
    e.add_argument("--model", required=True, help="chemin ou identifiant HF")
    e.add_argument("--concepts", required=True, help="fichier .json de spécification")
    e.add_argument("--layer", type=int, default=-1, help="défaut : milieu du réseau")
    e.add_argument("--fit", type=int, default=30, help="requêtes pour le calcul")
    e.add_argument("--test", type=int, default=20,
                   help="requêtes de test, disjointes — ajuster et tester sur les "
                        "mêmes donne 100 %% jusqu'à la couche 0, ce qui est une "
                        "tautologie et non une mesure")
    e.add_argument("--out", default="vectors")
    e.set_defaults(func=cmd_extract)

    b = sub.add_parser("probe", help="vérifier que l'application n'abîme pas le modèle")
    b.add_argument("--model", required=True)
    b.add_argument("--concepts", required=True)
    b.add_argument("--vectors", required=True, help="chemin du .npy")
    b.add_argument("--layer", type=int, required=True)
    b.add_argument("--alphas", default="0.5,1.0,2.0")
    b.add_argument("--lam", type=float, default=1.0,
                   help="1 = addition · 0 = remplacement · entre les deux = atténuation")
    b.add_argument("--max-tokens", type=int, default=90)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--show", action="store_true", help="afficher les textes générés")
    b.set_defaults(func=cmd_probe)

    a = p.parse_args()
    return a.func(a)
