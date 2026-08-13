"""Reading and editing the residual stream, for MLX models.

The delicate part is the patch. Assigning `layer.__call__` on an **instance** does not
work: for `layer(x)`, Python resolves special methods on the *type*. An earlier version
did exactly that and captured nothing — without raising, it returned an empty dict, which
reads as "no direction found" when it actually means "no measurement happened". Hence the
class-level patch and the mandatory `selftest()` before any campaign.
"""

from __future__ import annotations

import mlx.core as mx


class ResidualTap:
    """Wraps a model's layers to read and edit the residual stream.

    Works on hybrid architectures (heterogeneous attention layers) because it assumes
    nothing about the attention mechanism: only the residual structure is required, and
    that is universal in transformers.
    """

    def __init__(self, model):
        self.layers = self._find_layers(model)
        self.captured: list[tuple[int, mx.array]] = []
        self.enabled = False
        # (from_layer, unit vector, alpha, lambda):
        #     h ← h + (λ−1)·v(v·h) + α·v      applied at every layer ≥ from_layer
        # i.e. on the model's own component c along v:   c → λ·c + α
        #   λ=1 addition only · λ=0 replacement · 0<λ<1 attenuation
        self.edit: tuple[int, mx.array, float, float] | None = None
        self._orig: list = []

    @staticmethod
    def _find_layers(model):
        for obj in (model, getattr(model, "language_model", None),
                    getattr(model, "model", None)):
            if obj is not None and hasattr(obj, "layers"):
                return obj.layers
        raise RuntimeError("could not locate decoder layers")

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
                    from_layer, vec, alpha, lam = tap.edit
                    if i >= from_layer:
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
        """Verify that capture actually sees every layer.

        Call before any measurement: a silently inoperative tap yields zero vectors,
        which reads as a negative result instead of a broken instrument.
        """
        self.captured.clear()
        self.enabled = True
        model(mx.array([tokenizer.encode("test")]))
        self.enabled = False
        seen = len({i for i, _ in self.captured})
        self.captured.clear()
        if seen < len(self.layers):
            raise RuntimeError(
                f"capture is not working: {seen} layers out of {len(self.layers)}")

    def last_hidden(self, model, tokenizer, prompt: str) -> dict[int, mx.array]:
        """Activation at the last position, per layer.

        Last position rather than a mean over tokens: it is the one that determines the
        next token, so it is where a concept must show up if it influences the decision.
        A mean would dilute it with shared context.
        """
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
