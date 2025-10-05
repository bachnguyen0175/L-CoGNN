"""
Configuration parameters for Knowledge Distillation in Heterogeneous Graph Learning
"""

import argparse


def kd_params():
    """Knowledge Distillation specific parameters"""
    parser = argparse.ArgumentParser()
    
    # Basic model parameters
    parser.add_argument('--dataset', type=str, default="acm", choices=["acm", "dblp", "aminer", "freebase"])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--ratio', type=str, default="80_10_10")
    
    # Model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)

    # Evaluation parameters
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # Model paths
    parser.add_argument('--teacher_model_path', type=str, default=None, help="Path to pre-trained teacher model")
    parser.add_argument('--middle_teacher_path', type=str, default=None, help="Path to intermediate teacher model")
    parser.add_argument('--student_model_path', type=str, default=None, help="Path to pre-trained student model")
    parser.add_argument('--student_dim', type=int, default=64, help="Dimension of student model embeddings (50% in default)")

    # ==================== LOSS CONTROL FLAGS ====================
    # Core Loss Flags
    parser.add_argument('--use_student_contrast_loss', action='store_true', default=True, help="Use student's base contrastive loss")
    parser.add_argument('--use_kd_loss', action='store_true', default=True, help="Use knowledge distillation loss (teacher -> student)")
    parser.add_argument('--use_augmentation_alignment_loss', action='store_true', default=True, help="Use expert alignment loss (middle teacher)")
    parser.add_argument('--use_subspace_loss', action='store_true', default=True, help="Use subspace contrastive loss")
    
    # Link Prediction Loss Flags
    parser.add_argument('--use_link_recon_loss', action='store_true', default=False, help="Use link reconstruction loss")
    parser.add_argument('--use_relational_kd_loss', action='store_true', default=False, help="Use relational KD loss")
    
    # Advanced Loss Flags (already exist, keeping for reference)
    parser.add_argument('--use_multihop_link_loss', action='store_true', default=False, help="Use multi-hop link prediction loss")
    parser.add_argument('--use_metapath_specific_loss', action='store_true', default=False, help="Use meta-path specific link loss")
    parser.add_argument('--use_structural_distance', action='store_true', default=False, help="Use structural distance preservation")
    parser.add_argument('--use_attention_transfer', action='store_true', default=False, help="Use attention transfer loss")
    
    # Hidden Loss Flags (losses inside models)
    parser.add_argument('--use_guidance_alignment_loss', action='store_true', default=False, help="Use guidance alignment loss (hidden in student)")
    parser.add_argument('--use_gate_entropy_loss', action='store_true', default=False, help="Use gate entropy regularization (hidden in student)")
    parser.add_argument('--use_middle_divergence_loss', action='store_true', default=False, help="Use divergence loss in middle teacher")
    
    # Legacy flags (deprecated but kept for compatibility)
    parser.add_argument('--use_self_contrast', action='store_true', default=False, help="[DEPRECATED] Use self-contrast loss from LightGNN")
    parser.add_argument('--use_subspace_contrast', action='store_true', default=False, help="[DEPRECATED] Use subspace contrastive learning")
    
    # Enhanced distillation weights
    parser.add_argument('--self_contrast_weight', type=float, default=0.2, help="Weight for self-contrast loss")
    parser.add_argument('--subspace_weight', type=float, default=0.2, help="Weight for subspace contrastive loss")
    parser.add_argument('--self_contrast_temp', type=float, default=1.0, help="Temperature for self-contrast")
    parser.add_argument('--subspace_temp', type=float, default=1.0, help="Temperature for subspace contrast")
    
    # Hidden loss weights
    parser.add_argument('--guidance_alignment_weight', type=float, default=0.2, help="Weight for guidance alignment loss")
    parser.add_argument('--gate_entropy_weight', type=float, default=0.05, help="Weight for gate entropy regularization")
    parser.add_argument('--middle_divergence_weight', type=float, default=0.05, help="Weight for middle teacher divergence loss")
    
    # multi-stage training
    parser.add_argument('--mask_epochs', type=int, default=100, help="Epochs for mask training stage")
    parser.add_argument('--fixed_epochs', type=int, default=200, help="Epochs for fixed training stage") 
    parser.add_argument('--use_loosening', action='store_true', default=True, help="Use loosening factors in subspace learning")
    
    # Model saving
    parser.add_argument('--teacher_save_path', type=str, default="teacher_heco.pkl", help="Teacher model save path")
    parser.add_argument('--student_save_path', type=str, default="student_heco.pkl", help="Student model save path")
    parser.add_argument('--middle_teacher_save_path', type=str, default="middle_teacher_heco.pkl", help="Middle teacher save path")
    
    # Hierarchical training parameters
    parser.add_argument('--stage1_epochs', type=int, default=300, help='Epochs for stage 1 (teacher -> middle teacher)')
    parser.add_argument('--stage2_epochs', type=int, default=500, help='Epochs for stage 2 (middle teacher -> student)')
    parser.add_argument('--stage1_distill_weight', type=float, default=0.7, help='Distillation weight for stage 1')
    parser.add_argument('--stage2_distill_weight', type=float, default=0.8, help='Distillation weight for stage 2')
    parser.add_argument('--student_compression_ratio', type=float, default=0.5, help='Compression ratio for student')
    
    # Dual-Teacher System parameters
    parser.add_argument('--augmentation_weight', type=float, default=0.3, help='Weight for augmentation guidance loss')
    
    # Enhanced Knowledge Distillation parameters
    parser.add_argument('--kd_temperature', type=float, default=2.5, help='Temperature for knowledge distillation')
    
    # Enhanced Augmentation parameters for Dual-Teacher System
    parser.add_argument('--use_meta_path_connections', action='store_true', default=True, help="Connect nodes via meta-paths")
    parser.add_argument('--connection_strength', type=float, default=0.2, help="Meta-path connection strength")
    
    # Link Prediction Enhancement parameters
    parser.add_argument('--link_recon_weight', type=float, default=0.6, help="Weight for link reconstruction loss")
    parser.add_argument('--relational_kd_weight', type=float, default=0.6, help="Weight for relational knowledge distillation")
    parser.add_argument('--link_sample_rate', type=int, default=2000, help="Number of edges to sample for link prediction")
    parser.add_argument('--relational_sample_nodes', type=int, default=512, help="Number of nodes to sample for relational KD")
    
    # Multi-scale Link Prediction Enhancement (flags moved to LOSS CONTROL FLAGS section above)
    parser.add_argument('--multihop_weight', type=float, default=0.3, help="Weight for multi-hop link loss")
    parser.add_argument('--max_hops', type=int, default=3, help="Maximum number of hops for multi-hop loss")
    parser.add_argument('--metapath_specific_weight', type=float, default=0.25, help="Weight for meta-path specific loss")
    
    # Structural Knowledge Transfer (flags moved to LOSS CONTROL FLAGS section above)
    parser.add_argument('--structural_distance_weight', type=float, default=0.2, help="Weight for structural distance loss")
    parser.add_argument('--attention_transfer_weight', type=float, default=0.15, help="Weight for attention transfer")
    
    # Logging and evaluation
    parser.add_argument('--log_interval', type=int, default=1, help="Logging interval")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval")
    parser.add_argument('--save_interval', type=int, default=100, help="Model saving interval")
    
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
        'use_kd_loss': getattr(args, 'use_kd_loss', True),
        'use_augmentation_alignment_loss': getattr(args, 'use_augmentation_alignment_loss', True),
        'use_subspace_loss': getattr(args, 'use_subspace_loss', True),
        'use_link_recon_loss': getattr(args, 'use_link_recon_loss', True),
        'use_relational_kd_loss': getattr(args, 'use_relational_kd_loss', True),
        'use_multihop_link_loss': getattr(args, 'use_multihop_link_loss', True),
        'use_metapath_specific_loss': getattr(args, 'use_metapath_specific_loss', True),
        'use_structural_distance': getattr(args, 'use_structural_distance', True),
        'use_attention_transfer': getattr(args, 'use_attention_transfer', True),
        'use_self_contrast': getattr(args, 'use_self_contrast', True),
        'use_subspace_contrast': getattr(args, 'use_subspace_contrast', True),
        'self_contrast_weight': getattr(args, 'self_contrast_weight', 0.2),
        'subspace_weight': getattr(args, 'subspace_weight', 0.2),
        'self_contrast_temp': getattr(args, 'self_contrast_temp', 1.0),
        'subspace_temp': getattr(args, 'subspace_temp', 1.0),
        'middle_divergence_weight': getattr(args, 'middle_divergence_weight', 0.05),
        'kd_temperature': getattr(args, 'kd_temperature', 2.5),
        'link_recon_weight': getattr(args, 'link_recon_weight', 0.6),
        'relational_kd_weight': getattr(args, 'relational_kd_weight', 0.6),
        'link_sample_rate': getattr(args, 'link_sample_rate', 2000),
        'relational_sample_nodes': getattr(args, 'relational_sample_nodes', 512),
        'multihop_weight': getattr(args, 'multihop_weight', 0.3),
        'max_hops': getattr(args, 'max_hops', 3),
        'metapath_specific_weight': getattr(args, 'metapath_specific_weight', 0.25),
        'structural_distance_weight': getattr(args, 'structural_distance_weight', 0.2),
        'attention_transfer_weight': getattr(args, 'attention_transfer_weight', 0.15)
    }

def get_augmentation_config(args):
    """Get augmentation configuration dictionary"""
    return {
        'use_meta_path_connections': getattr(args, 'use_meta_path_connections', True),
        'connection_strength': getattr(args, 'connection_strength', 0.2)
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
        'feat_drop': args.feat_drop,
        'attn_drop': args.attn_drop,
        'tau': args.tau,
        'lam': args.lam,
        'lr': args.lr,
        'l2_coef': args.l2_coef,
        'nb_epochs': args.nb_epochs,
        'patience': args.patience
    }