"""Extraction de directions conceptuelles par paires minimales.

Principe : deux prompts **identiques sauf une proposition**, et la différence des
activations n'encode que ce qui diffère. C'est la précaution centrale — un contraste
« prompt contre absence de prompt » produit un vecteur qui mélange le concept visé, la
longueur, le registre et tout ce que le prompt ajoute par ailleurs.

Trois contrôles sont fournis, chacun né d'une erreur constatée :

  séparabilité hors échantillon   ajuster et tester sur les mêmes paires donne 100 %
                                  jusqu'à la couche 0, avant qu'aucun concept ne soit
                                  formé — c'est une tautologie, pas une mesure.
  matrice de cosinus              deux concepts « distincts » peuvent partager
                                  l'essentiel de leur direction. Sans elle, on croit
                                  manipuler un concept alors qu'on en manipule un autre.
  amplitude naturelle             la norme du vecteur avant normalisation est ce que le
                                  prompt déplace réellement. Un α exprimé en unités
                                  arbitraires n'est comparable à rien.
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
    """Différence de moyennes entre les deux branches d'une paire minimale."""
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
    """Fraction de requêtes **non utilisées pour le calcul** où la projection classe
    correctement la branche positive au-dessus de la négative. 0,5 = aucune séparation.

    Le profil par profondeur est plus informatif que la valeur maximale : une
    séparabilité forte dès la couche 0 signale un contraste lisible dans les tokens
    eux-mêmes — donc lexical, pas conceptuel.
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
    """Extrait une direction par concept, plus les diagnostics d'ensemble."""
    base = spec["base_prompt"]
    qs = spec["queries"]
    fit, test = qs[:n_fit], qs[n_fit:n_fit + n_test]
    if not test:
        raise ValueError("pas assez de requêtes : il en faut pour l'ajustement ET "
                         "pour le test hors échantillon")

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
              f"séparabilité {seps[c['id']]:.3f}", flush=True)

    ids = [c["id"] for c in spec["concepts"]]
    M = mx.stack([vecs[i] for i in ids]).astype(mx.float32)
    cos = {a: {b: round(float((vecs[a] * vecs[b]).sum().item()), 3) for b in ids}
           for a in ids}
    hors = [cos[a][b] for a, b in itertools.combinations(ids, 2)]

    S = mx.linalg.svd(M, compute_uv=False, stream=mx.cpu)
    tot = float((S ** 2).sum().item())
    cum, rang = 0.0, len(ids)
    for i, s in enumerate(S.tolist()):
        cum += s * s
        if cum / tot >= 0.90:
            rang = i + 1
            break

    moy = M.mean(axis=0)
    moy_u = moy / mx.linalg.norm(moy)
    nat = statistics.fmean([norms[i] * abs(float((vecs[i] * moy_u).sum().item()))
                            for i in ids])

    return {
        "layer": layer, "ids": ids, "vectors": vecs, "combined": moy_u,
        "report": {
            "amplitudes": norms, "separabilites": seps, "cosinus": cos,
            "cosinus_hors_diagonale": {
                "moyenne": round(statistics.fmean(hors), 3),
                "max": round(max(hors), 3), "min": round(min(hors), 3)},
            "valeurs_singulieres": [round(float(x), 3) for x in S.tolist()],
            "rang_effectif_90pct": rang,
            "amplitude_naturelle_combinee": round(nat, 4),
            "alphas_suggeres": [round(k * nat, 3) for k in (0.5, 1, 2, 3)],
        },
    }


def save(result: dict, out_dir: Path, n_layers: int) -> None:
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
