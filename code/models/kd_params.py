"""
Configuration parameters for Knowledge Distillation in Heterogeneous Graph Learning
"""

import argparse


def kd_params():
    """Knowledge Distillation specific parameters"""
    parser = argparse.ArgumentParser()
    
    # Basic model parameters
    parser.add_argument('--dataset', type=str, default="acm", choices=["acm", "dblp", "aminer", "freebase"])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # Model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)
    
    # Evaluation parameters
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # Knowledge Distillation parameters
    parser.add_argument('--teacher_model_path', type=str, default=None, help="Path to pre-trained teacher model")
    parser.add_argument('--middle_teacher_path', type=str, default=None, help="Path to intermediate teacher model")
    parser.add_argument('--student_model_path', type=str, default=None, help="Path to pre-trained student model")
    
    # Student model configuration
    parser.add_argument('--compression_ratio', type=float, default=0.5, help="Student model compression ratio")
    parser.add_argument('--student_layers', type=int, default=None, help="Number of layers in student model")
    
    # Distillation loss weights
    parser.add_argument('--embedding_weight', type=float, default=0.5, help="Weight for embedding-level distillation")
    parser.add_argument('--heterogeneous_weight', type=float, default=0.3, help="Weight for heterogeneous distillation")
    parser.add_argument('--prediction_weight', type=float, default=0.5, help="Weight for prediction-level distillation")
    
    # Distillation temperatures
    parser.add_argument('--embedding_temp', type=float, default=4.0, help="Temperature for embedding distillation")
    parser.add_argument('--prediction_temp', type=float, default=4.0, help="Temperature for prediction distillation")
    
    # Distillation flags
    parser.add_argument('--use_embedding_kd', action='store_true', default=True, help="Use embedding-level KD")
    parser.add_argument('--use_heterogeneous_kd', action='store_true', default=True, help="Use heterogeneous KD")
    parser.add_argument('--use_prediction_kd', action='store_true', default=True, help="Use prediction-level KD")
    parser.add_argument('--use_self_contrast', action='store_true', default=True, help="Use self-contrast loss from LightGNN")
    parser.add_argument('--use_subspace_contrast', action='store_true', default=True, help="Use subspace contrastive learning")
    
    # Enhanced distillation weights (from LightGNN)
    parser.add_argument('--self_contrast_weight', type=float, default=0.2, help="Weight for self-contrast loss")
    parser.add_argument('--subspace_weight', type=float, default=0.3, help="Weight for subspace contrastive loss")
    parser.add_argument('--self_contrast_temp', type=float, default=1.0, help="Temperature for self-contrast")
    parser.add_argument('--subspace_temp', type=float, default=1.0, help="Temperature for subspace contrast")

    # Multi-level distillation parameters
    parser.add_argument('--use_multi_level_kd', action='store_true', default=True, help="Use multi-level knowledge distillation")
    parser.add_argument('--multi_level_weight', type=float, default=0.4, help="Weight for multi-level distillation")
    
    # Pruning and multi-stage training (adapted from LightGNN)
    parser.add_argument('--use_multi_stage', action='store_true', default=False, help="Use multi-stage training")
    parser.add_argument('--mask_epochs', type=int, default=100, help="Epochs for mask training stage")
    parser.add_argument('--fixed_epochs', type=int, default=200, help="Epochs for fixed training stage") 
    parser.add_argument('--pruning_start', type=int, default=1, help="Start pruning run")
    parser.add_argument('--pruning_end', type=int, default=5, help="End pruning run")
    parser.add_argument('--use_loosening', action='store_true', default=True, help="Use loosening factors in subspace learning")
    
    # Enhanced training mode
    parser.add_argument('--use_enhanced_training', action='store_true', default=False, help="Use enhanced training with all LightGNN techniques")
    
    # Model saving
    parser.add_argument('--teacher_save_path', type=str, default="teacher_heco.pkl", help="Teacher model save path")
    parser.add_argument('--student_save_path', type=str, default="student_heco.pkl", help="Student model save path")
    parser.add_argument('--middle_teacher_save_path', type=str, default="middle_teacher_heco.pkl", help="Middle teacher save path")
    
    # Hierarchical training parameters
    parser.add_argument('--stage1_epochs', type=int, default=500, help='Epochs for stage 1 (teacher -> middle teacher)')
    parser.add_argument('--stage2_epochs', type=int, default=1000, help='Epochs for stage 2 (middle teacher -> student)')
    parser.add_argument('--stage1_distill_weight', type=float, default=0.7, help='Distillation weight for stage 1')
    parser.add_argument('--stage2_distill_weight', type=float, default=0.8, help='Distillation weight for stage 2')
    parser.add_argument('--middle_compression_ratio', type=float, default=0.7, help='Compression ratio for middle teacher')
    parser.add_argument('--student_compression_ratio', type=float, default=0.5, help='Compression ratio for student')
    
    # Simplified Augmentation parameters
    parser.add_argument('--use_node_masking', action='store_true', default=True, help="Use node feature masking augmentation")
    parser.add_argument('--use_edge_augmentation', action='store_true', default=True, help="Use edge dropping augmentation")
    parser.add_argument('--use_autoencoder', action='store_true', default=True, help="Use autoencoder reconstruction")
    parser.add_argument('--mask_rate', type=float, default=0.1, help="Node masking rate")
    parser.add_argument('--remask_rate', type=float, default=0.3, help="Remasking rate during decoding")
    parser.add_argument('--edge_drop_rate', type=float, default=0.1, help="Edge dropping rate")
    parser.add_argument('--num_remasking', type=int, default=2, help="Number of remasking iterations")
    parser.add_argument('--reconstruction_weight', type=float, default=0.1, help="Weight for reconstruction loss")
    
    # Logging and evaluation
    parser.add_argument('--log_interval', type=int, default=10, help="Logging interval")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval")
    parser.add_argument('--save_interval', type=int, default=500, help="Model saving interval")
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args, _ = parser.parse_known_args()
    
    # Dataset-specific configurations
    if args.dataset == "acm":
        args.type_num = [4019, 7167, 60]  # [paper, author, subject]
        args.nei_num = 2
    elif args.dataset == "dblp":
        args.type_num = [4057, 14328, 7723, 20]  # [paper, author, conference, term]
        args.nei_num = 3
    elif args.dataset == "aminer":
        args.type_num = [6564, 13329, 35890]  # [paper, author, reference]
        args.nei_num = 2
    elif args.dataset == "freebase":
        args.type_num = [3492, 2502, 33401, 4459]  # [movie, director, actor, writer]
        args.nei_num = 3
    
    return args


def get_distillation_config(args):
    """Get distillation configuration dictionary"""
    return {
        'use_embedding_kd': args.use_embedding_kd,
        'use_prediction_kd': args.use_prediction_kd,
        'use_heterogeneous_kd': args.use_heterogeneous_kd,
        'use_self_contrast': getattr(args, 'use_self_contrast', True),
        'use_subspace_contrast': getattr(args, 'use_subspace_contrast', True),
        'use_multi_level_kd': getattr(args, 'use_multi_level_kd', True),
        'embedding_temp': args.embedding_temp,
        'prediction_temp': args.prediction_temp,
        'self_contrast_temp': getattr(args, 'self_contrast_temp', 1.0),
        'subspace_temp': getattr(args, 'subspace_temp', 1.0),
        'embedding_weight': args.embedding_weight,
        'prediction_weight': args.prediction_weight,
        'heterogeneous_weight': args.heterogeneous_weight,
        'self_contrast_weight': getattr(args, 'self_contrast_weight', 0.2),
        'subspace_weight': getattr(args, 'subspace_weight', 0.3),
        'multi_level_weight': getattr(args, 'multi_level_weight', 0.4),
        'pruning_run': 0,  # Will be updated during training
        'use_loosening': getattr(args, 'use_loosening', True)
    }


def acm_kd_params():
    """ACM dataset specific parameters for KD"""
    args = kd_params()
    args.dataset = "acm"
    args.hidden_dim = 64
    args.nb_epochs = 10000
    args.lr = 0.0008
    args.tau = 0.8
    args.feat_drop = 0.3
    args.attn_drop = 0.5
    args.sample_rate = [7, 1]
    args.lam = 0.5
    args.compression_ratio = 0.5
    args.embedding_weight = 0.5
    args.heterogeneous_weight = 0.3
    args.embedding_temp = 4.0
    return args


def dblp_kd_params():
    """DBLP dataset specific parameters for KD"""
    args = kd_params()
    args.dataset = "dblp"
    args.hidden_dim = 64
    args.nb_epochs = 8000
    args.lr = 0.001
    args.tau = 0.9
    args.feat_drop = 0.4
    args.attn_drop = 0.6
    args.sample_rate = [8, 2, 1]
    args.lam = 0.6
    args.compression_ratio = 0.4
    args.embedding_weight = 0.6
    args.heterogeneous_weight = 0.4
    args.embedding_temp = 3.0
    return args


def get_teacher_config(args):
    """Configuration for teacher model training"""
    return {
        'hidden_dim': args.hidden_dim,
        'feat_drop': args.feat_drop,
        'attn_drop': args.attn_drop,
        'tau': args.tau,
        'lam': args.lam,
        'lr': args.lr,
        'l2_coef': args.l2_coef,
        'nb_epochs': args.nb_epochs,
        'patience': args.patience
    }


def get_student_config(args):
    """Configuration for student model training"""
    return {
        'hidden_dim': args.hidden_dim,
        'compression_ratio': args.compression_ratio,
        'feat_drop': args.feat_drop,
        'attn_drop': args.attn_drop,
        'tau': args.tau,
        'lam': args.lam,
        'lr': args.lr,
        'l2_coef': args.l2_coef,
        'nb_epochs': args.nb_epochs,
        'patience': args.patience
    }
