"""calibrated-steering — extract, validate and apply concept directions.

A toolkit for isolating a concept in a language model's activations from **minimal
prompt pairs**, checking that you isolated what you think you did, and applying it at
inference without breaking the model.

Most of the value is in the controls, not the extraction: a difference of means takes ten
lines, but it produces a vector that *looks* correct in a great many cases where it isn't.
"""

from .tap import ResidualTap
from .extract import direction, extract_all, save, separability
from .probe import run as probe_run, verdict as probe_verdict

__all__ = ["ResidualTap", "direction", "extract_all", "save", "separability",
           "probe_run", "probe_verdict"]
__version__ = "0.1.0"
