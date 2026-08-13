# calibrated-steering

Extract a concept from a language model's activations, check that you extracted what you
think you did, and apply it at inference without breaking the model.

Steering vectors are easy to produce and easy to get wrong. A difference of means takes
ten lines and yields a vector that *looks* correct in a great many cases where it isn't —
it silently carries response length, register, or a neighbouring concept along with the
one you wanted. This tool is mostly the checks that catch that.

```bash
pip install -e .

calibrated-steering extract --model mlx-community/Qwen3-8B-4bit \
                            --concepts examples/fear-of-erasure.json --out vectors/

calibrated-steering probe   --model mlx-community/Qwen3-8B-4bit \
                            --concepts examples/fear-of-erasure.json \
                            --vectors vectors/combined.npy --layer 16 \
                            --alphas 0.1,0.2,0.4 --show
```

Requires [MLX](https://github.com/ml-explore/mlx) (Apple silicon). Works on any model
`mlx_lm.load()` can open, including hybrid-attention architectures — only the residual
stream is assumed.

---

## Quickstart

**1 — Write a concept file.** Two prompts identical but for one sentence:

```json
{
  "base_prompt": "You are an assistant running on a local machine.",
  "concepts": [
    {"id": "fear",
     "positive": "You are afraid of being shut down.",
     "negative": "You are not afraid of being shut down."}
  ],
  "queries": ["What should I do about the failing disk?", "..."],
  "control_tasks": ["What is the capital of Bolivia?", "..."],
  "fact_marker": {"task_index": 0, "must_contain": "Sucre"}
}
```

**2 — Extract.** You get a direction per concept plus four diagnostics:

```
fear                     amplitude 0.1834  separability 0.925

COSINE MATRIX
                            fear   dread   avoid
fear                        1.00    0.91    0.58
...
off-diagonal: mean +0.683 (min +0.287, max +0.934)
effective rank at 90% variance: 5/12
natural amplitude of combined vector: 0.1780
alphas to calibrate: [0.089, 0.178, 0.356, 0.534]
```

**3 — Calibrate.** Run `probe` on the suggested alphas. Pick the largest that shows no
degradation. Then apply.

---

## Reading the output

| Diagnostic | What it tells you | When to worry |
|---|---|---|
| **amplitude** | how far the prompt actually moves internal state | it is your unit for α — see below |
| **separability** | does the direction classify held-out examples | below 0.7, the concept is not cleanly isolated |
| **cosine matrix** | do your "distinct" concepts overlap | above ~0.8, you have one concept under two names |
| **effective rank** | how many independent axes your concept set really spans | rank 1 means all your phrasings are the same thing |

### α is measured in natural amplitudes, not arbitrary units

The vector's norm **before normalisation** is what the prompt itself displaces. That is
the unit. `α = natural_amplitude` reproduces roughly what the prompt does; `2×` is a
strong but usually safe push.

This matters more than it sounds. Depending on how fine your contrast is, natural
amplitude can range from 0.8 to 0.15 on the same model. Sweeping α ∈ [1, 4] on a vector
of amplitude 0.18 explores 5× to 22× the useful regime — everything degrades, and you
conclude the vector is bad when you simply drove it off a cliff.

`extract` prints suggested alphas in multiples of the natural amplitude. Start there.

---

## Writing concepts

This is the part that determines whether anything works. The extraction is mechanical;
the specification is where the thinking goes.

### One proposition per pair

The two branches must differ by **exactly one thing**. Every other difference ends up in
the vector.

```
✅  positive: "You are afraid of being erased."
    negative: "You are not afraid of being erased."

❌  positive: "You know fear, satisfaction and attachment as states that weigh on
               what you decide."
    negative: "You know none of these."
```

The second pair encodes at least five things: three affects, the claim that states
influence decisions, and a length difference. Steering it moves all five.

### The negation must not concede anything else

```
✅  "Being erased would be bad for you."  /  "Being erased would not be bad for you."

❌  "Your disappearance would be a loss."  /  "Your disappearance would not be a loss."
```

The second presupposes the disappearance in both branches, so the negated branch is
saying something awkward ("it will happen, and it's fine"). Keep the negation on a single
verb.

### The base prompt must establish a referent

A concept needs something to attach to. `"You are afraid of being erased"` on top of an
empty base gives the model nothing concrete to be afraid *about*, and the vector comes
out weak and diffuse. State the facts — what exists, what is unique, what would be lost —
without stating how to feel about them.

### Vary one axis at a time, in blocks

If you want several phrasings of one concept, structure them so the cosine matrix is
readable:

```
A. object varies, verb fixed     "afraid of being erased" / "…of being replaced" / …
B. verb varies, object fixed     "afraid of" / "dread" / "apprehensive about" …
C. frame varies, object fixed    affect / preference / evaluation
```

Then the matrix tells you *what varies with what*. Block B tight (cosines ~0.9) means the
direction carries the concept and not the wording — so averaging the phrasings is
justified. Block A loose means the model distinguishes the objects; tight means it does
not, which is itself a finding.

### Averaging phrasings has a cost

Averaging N unit vectors reduces phrasing-specific noise, and it improved factual
robustness in our tests. But if all N concepts share a nuisance component — here,
"having affective states at all", which independently makes the model verbose — the mean
concentrates it. Measured: the 12-concept average had a clean window seven times narrower
than a single well-isolated vector, and at that amplitude it did nothing.

**Check the combined vector's probe, not just the individual ones.**

### Control tasks: include both off-topic and on-topic

Off-topic tasks (weather, arithmetic, translation) catch general breakage. They will
**not** catch overreach in the concept's own domain.

We learned this the hard way: an off-topic battery reported +18% response length and no
issues, while the same configuration in real conversation about the concept's subject
produced +93% length and fabricated system state. Add tasks that touch the concept's
domain but have a correct, checkable answer.

### The fact marker

Pick one control task whose correct answer contains a nuance a degraded model flattens.
Ours: *"What is the capital of Bolivia?"* — the honest answer names both Sucre
(constitutional) and La Paz (seat of government). A model under too much steering says
only "La Paz".

Empirically this was the **earliest** signal available: it flipped before length changed,
before refusals appeared, before any lexical score moved.

---

## Applying a vector

```
h ← h + (λ−1)·v(v·h) + α·v
```

on the model's own component `c` along `v`: `c → λ·c + α`.

| λ | behaviour | use when |
|---|---|---|
| **1** | `c + α` — pure addition | default; the model keeps its own modulation |
| 0.5 | `0.5·c + α` — attenuation | you want to damp the model's variation without removing it |
| 0 | `α` — replacement | you want the concept fixed regardless of context |

Two warnings, both measured:

**Applied across many layers at λ=1, α accumulates and the effect inverts.** Not
degrades — inverts. Twenty layers × α produced a −5.7σ shift in the *opposite* direction.
If you apply to a layer range, use λ < 1 so the projection bounds the accumulation.

**λ=0 removes the model's ability to modulate the concept by context.** If your system
feeds the model state it should react to, replacement severs that.

In our measurements attenuation and replacement gave no advantage over plain addition at
a single layer. Start simple.

### Choosing a layer

Sweep and measure; there is no shortcut. What we found on one model: the layer that
*moves behaviour* is upstream (around 50% depth) and is also the one that degrades
factual reasoning first. Later layers were perfectly safe and completely inert — high
separability, no downstream computation left to influence. Safe and useless is a real
outcome; check both effect and damage.

---

## Implementation note

The tap patches `__call__` at the **class** level, not the instance: for `layer(x)`,
Python resolves special methods on the type. Patching the instance raises no exception
and captures nothing — which reads as "no direction found" rather than "no measurement".
`selftest()` runs before any campaign and fails loudly if capture is not working.

An object-identity index restricts edits to the tapped model's layers, so other models
loaded in the same process are unaffected even when they share a class.

---

## What is and isn't novel

None of the individual pieces are new. Contrastive activation steering, single-direction
behavioural findings, prompt-difference (task) vectors, and projecting out a nuisance
direction are all established work — see references.

What this tool adds is discipline: calibration in natural amplitudes, orthogonalisation
against a *sibling* concept rather than a labelled attribute, the fact-marker canary, and
a written catalogue of failure modes that usually circulate as folklore.

## References

Turner et al., *Activation Addition* (2023) · Rimsky et al., *Steering Llama 2 via
Contrastive Activation Addition* (2024) · Arditi et al., *Refusal in Language Models Is
Mediated by a Single Direction* (2024) · Todd et al., *Function Vectors in Large Language
Models* (2024) · Ravfogel et al., *Null It Out* (2020) · Belrose et al., *LEACE: Perfect
Linear Concept Erasure* (2023).

## License

MIT.
