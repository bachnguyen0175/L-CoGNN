"""
KD-HGRL Training Package
=======================

This package contains all training scripts for the hierarchical knowledge distillation pipeline.

Training stages:
1. Teacher training (pretrain_teacher.py)
2. Middle teacher training (train_middle_teacher.py) 
3. Student training (train_student.py)

Additional components:
- hetero_augmentations.py: Heterogeneous graph augmentation techniques
"""

# Import main training classes if they exist
try:
    from .pretrain_teacher import TeacherTrainer
except ImportError:
    pass

try:
    from .train_middle_teacher import MiddleTeacherTrainer
except ImportError:
    pass

try:
    from .train_student import StudentTrainer
except ImportError:
    pass

__all__ = [
    'TeacherTrainer', 'MiddleTeacherTrainer', 'StudentTrainer'
]