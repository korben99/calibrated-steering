"""concept-vectors — extraire, contrôler et appliquer des directions conceptuelles.

Boîte à outils pour isoler un concept dans les activations d'un modèle de langage à
partir de **paires minimales** de prompts, vérifier qu'on a bien isolé ce qu'on croit,
et l'appliquer à l'inférence sans abîmer le modèle.

L'essentiel de la valeur est dans les contrôles, pas dans l'extraction : une différence
de moyennes s'écrit en dix lignes, mais elle produit un vecteur qui *paraît* juste dans
un très grand nombre de cas où il ne l'est pas.
"""

from .tap import ResidualTap
from .extract import direction, extract_all, save, separability
from .probe import run as probe_run, verdict as probe_verdict

__all__ = ["ResidualTap", "direction", "extract_all", "save", "separability",
           "probe_run", "probe_verdict"]
__version__ = "0.1.0"
