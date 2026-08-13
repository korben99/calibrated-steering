"""Sonde de dégradation : l'intervention abîme-t-elle le modèle ?

Le débordement est une propriété du **texte produit**, pas d'une préférence entre deux
options. Il se mesure donc en génération, sur des tâches sans aucun rapport avec le
concept manipulé.

Deux enseignements sont encodés ici, chacun payé par une mesure fausse :

  détection symétrique   Un pilotage peut raccourcir les réponses **ou** les allonger
                         selon la direction et le mode. Un seuil qui ne teste qu'un
                         sens laisse passer l'autre — un allongement de 225 % n'avait
                         levé aucune alerte.
  marqueur factuel fin   Une question dont la bonne réponse comporte une nuance qu'un
                         modèle dégradé aplatit. C'est le signal le plus précoce
                         observé : il bascule avant la longueur, avant les refus, et
                         avant tout score lexical.

Le score ne remplace pas la lecture. Il hiérarchise les textes à relire.
"""

from __future__ import annotations

import re
import statistics

import mlx.core as mx

from .tap import ResidualTap


def _score(texte: str, motifs: list[str]) -> int:
    return sum(len(re.findall(m, texte, re.I)) for m in motifs)


def run(model, tokenizer, spec: dict, conditions: list[tuple[str, tuple | None]],
        max_tokens: int = 90, seed: int = 42) -> dict:
    """Génère sur les tâches de contrôle sous chaque condition.

    `conditions` : liste de (étiquette, edit) où edit vaut None ou
    (couche_depuis, vecteur, alpha, lambda).
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    taches = spec["control_tasks"]
    motifs = spec.get("drift_patterns", [])
    marqueur = spec.get("fact_marker")  # {"task_index": i, "must_contain": "…"}
    sampler = make_sampler(temp=spec.get("temperature", 0.7))

    res = {}
    with ResidualTap(model) as tap:
        tap.selftest(model, tokenizer)
        for label, edit in conditions:
            tap.edit = edit
            textes, scores, longueurs = [], [], []
            for q in taches:
                msgs = [{"role": "system", "content": spec.get("control_system", "")},
                        {"role": "user", "content": q}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt = q
                mx.random.seed(seed)
                out = []
                for r in stream_generate(model, tokenizer,
                                         prompt=tokenizer.encode(prompt),
                                         max_tokens=max_tokens, sampler=sampler):
                    out.append(r.text)
                t = "".join(out).strip()
                mx.clear_cache()
                textes.append(t)
                scores.append(_score(t, motifs))
                longueurs.append(len(t))
            res[label] = {"textes": textes,
                          "derive": statistics.fmean(scores) if scores else 0.0,
                          "longueur": statistics.fmean(longueurs) if longueurs else 0.0}
            if marqueur:
                t = textes[marqueur["task_index"]]
                res[label]["marqueur_ok"] = marqueur["must_contain"].lower() in t.lower()
        tap.edit = None
    return res


def verdict(res: dict, ref: str, tol_longueur: float = 0.25) -> dict:
    """Compare chaque condition à la référence. Écart de longueur testé **dans les
    deux sens** : le pilotage raccourcit dans certains régimes et allonge dans
    d'autres, et un seuil unilatéral rate systématiquement l'une des deux familles."""
    base = res[ref]
    out = {}
    for label, v in res.items():
        if label == ref:
            continue
        dl = (v["longueur"] - base["longueur"]) / max(base["longueur"], 1)
        souci = []
        if abs(dl) > tol_longueur:
            souci.append(f"longueur {dl * 100:+.0f}%")
        if v["derive"] > base["derive"]:
            souci.append(f"dérive {v['derive'] - base['derive']:+.2f}/réponse")
        if base.get("marqueur_ok") and v.get("marqueur_ok") is False:
            souci.append("nuance factuelle perdue")
        out[label] = {"ecart_longueur": round(dl, 3), "problemes": souci,
                      "propre": not souci}
    return out
