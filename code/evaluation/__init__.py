"""
KD-HGRL Evaluation Package
=========================

This package contains evaluation and comparison utilities for the KD-HGRL project.

Available evaluators:
- ComprehensiveEvaluator: Complete evaluation across all three downstream tasks
- ModelEvaluator: KD-specific evaluation and comparison utilities
"""

try:
    from .comprehensive_evaluation import ComprehensiveEvaluator
except ImportError:
    pass

try:
    from .evaluate_kd import ModelEvaluator
except ImportError:
    pass

__all__ = [
    'ComprehensiveEvaluator', 'ModelEvaluator'
]