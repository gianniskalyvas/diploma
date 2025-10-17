# metrics/__init__.py

from .metrics import calculate_metrics, detect_contradiction, semantic_similarity
from .plotting import plot_size_comparison, plot_attack_comparison

__all__ = [
    "calculate_metrics",
    "semantic_similarity",
    "detect_contradiction",
    "plot_attack_comparison",
    "plot_size_comparison"
]
