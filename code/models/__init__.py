"""
KD-HGRL Models Package
=====================

This package contains all neural network models and architectures used in the KD-HGRL project.

Available models:
- MyHeCo: Main HeCo model for heterogeneous graph learning
- MiddleMyHeCo: Middle teacher model for hierarchical distillation
- StudentMyHeCo: Student model for knowledge distillation
- Contrast: Contrastive learning module
- Sc_encoder: Semantic-level attention encoder
"""

from .kd_heco import MyHeCo, MiddleMyHeCo, StudentMyHeCo, MyHeCoKD, count_parameters, calculate_compression_ratio
from .contrast import Contrast
from .sc_encoder import Sc_encoder, inter_att, intra_att, mySc_encoder
from .kd_params import kd_params, get_teacher_config, get_student_config, get_distillation_config

__all__ = [
    # Main models
    'MyHeCo', 'MiddleMyHeCo', 'StudentMyHeCo', 'MyHeCoKD',
    # Utilities
    'count_parameters', 'calculate_compression_ratio',
    # Components
    'Contrast', 'Sc_encoder', 'inter_att', 'intra_att', 'mySc_encoder',
    # Configuration
    'kd_params', 'get_teacher_config', 'get_student_config', 'get_distillation_config'
]