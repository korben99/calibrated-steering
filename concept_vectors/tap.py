"""Capture et modification du flux résiduel, pour modèles MLX.

Le point délicat est le patch. Assigner `layer.__call__` sur une **instance** ne
fonctionne pas : pour `layer(x)`, Python résout les méthodes spéciales sur le *type*.
Une première version le faisait et ne capturait rien — sans lever d'erreur, elle
renvoyait un dictionnaire vide, ce qui ressemble à « pas de direction trouvée » alors
que c'est « pas de mesure ». D'où le patch au niveau des classes et le `selftest()`
obligatoire avant toute campagne.
"""

from __future__ import annotations

import mlx.core as mx


class ResidualTap:
    """Enveloppe les couches d'un modèle pour lire et modifier le flux résiduel.

    Fonctionne sur les architectures hybrides (couches d'attention hétérogènes) parce
    qu'elle ne suppose rien du mécanisme d'attention : seule la structure résiduelle
    est requise, et elle est universelle dans les transformeurs.
    """

    def __init__(self, model):
        self.layers = self._find_layers(model)
        self.captured: list[tuple[int, mx.array]] = []
        self.enabled = False
        # (couche_depuis, vecteur unitaire, alpha, lambda) :
        #     h ← h + (λ−1)·v(v·h) + α·v      appliqué à chaque couche ≥ couche_depuis
        # soit, sur la composante c du modèle le long de v :   c → λ·c + α
        #   λ=1 addition seule · λ=0 remplacement · 0<λ<1 atténuation
        self.edit: tuple[int, mx.array, float, float] | None = None
        self._orig: list = []

    @staticmethod
    def _find_layers(model):
        for obj in (model, getattr(model, "language_model", None),
                    getattr(model, "model", None)):
            if obj is not None and hasattr(obj, "layers"):
                return obj.layers
        raise RuntimeError("couches du décodeur introuvables")

    def __enter__(self):
        self._index = {id(l): i for i, l in enumerate(self.layers)}
        tap = self
        for cls in {type(l) for l in self.layers}:
            orig = cls.__call__
            self._orig.append((cls, orig))

            def wrapped(self, *a, _orig=orig, **kw):
                out = _orig(self, *a, **kw)
                i = tap._index.get(id(self))
                if i is None:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                if tap.enabled:
                    tap.captured.append((i, h))
                if tap.edit is not None:
                    depuis, vec, alpha, lam = tap.edit
                    if i >= depuis:
                        proj = (h * vec).sum(axis=-1, keepdims=True)
                        h = h + (lam - 1.0) * proj * vec + alpha * vec
                        return (h,) + out[1:] if isinstance(out, tuple) else h
                return out

            cls.__call__ = wrapped
        return self

    def __exit__(self, *exc):
        for cls, orig in self._orig:
            cls.__call__ = orig
        self._orig.clear()

    def selftest(self, model, tokenizer) -> None:
        """Vérifie que la capture voit bien toutes les couches. À appeler avant
        toute mesure : un tap inopérant produit des vecteurs nuls, ce qui se lit
        comme un résultat négatif au lieu d'une panne."""
        self.captured.clear()
        self.enabled = True
        model(mx.array([tokenizer.encode("test")]))
        self.enabled = False
        vues = len({i for i, _ in self.captured})
        self.captured.clear()
        if vues < len(self.layers):
            raise RuntimeError(
                f"capture inopérante : {vues} couches sur {len(self.layers)}")

    def last_hidden(self, model, tokenizer, prompt: str) -> dict[int, mx.array]:
        """Activation de la dernière position, par couche. Dernière position et non
        moyenne : c'est elle qui détermine le token suivant, donc celle où un concept
        doit se lire s'il influence la décision."""
        self.captured.clear()
        self.enabled = True
        model(mx.array([tokenizer.encode(prompt)]))
        self.enabled = False
        out = {}
        for li, h in self.captured:
            v = h[0, -1, :]
            mx.eval(v)
            out[li] = v
        self.captured.clear()
        return out
