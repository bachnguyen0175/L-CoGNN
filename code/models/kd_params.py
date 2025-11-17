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
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--ratio', type=str, default="80_10_10")
    
    # Model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.5)

    # Evaluation parameters
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # Model paths
    parser.add_argument('--teacher_model_path', type=str, default=None, help="Path to pre-trained teacher model")
    parser.add_argument('--middle_teacher_path', type=str, default=None, help="Path to pre-trained middle teacher model")
    parser.add_argument('--student_model_path', type=str, default=None, help="Path to pre-trained student model")
    parser.add_argument('--student_dim', type=int, default=32, help="Dimension of student model embeddings (50% in default)")

    # ==================== LOSS CONTROL FLAGS ====================
    # Core Loss Flags
    # 1
    parser.add_argument('--use_kd_loss', action='store_true', default=True, help="Use knowledge distillation loss (teacher -> student)")
    # 2
    parser.add_argument('--use_augmentation_alignment_loss', action='store_true', default=False, help="Use expert alignment loss (middle teacher)")
    
    # Supplementary Loss Flags
    # 3
    parser.add_argument('--use_link_recon_loss', action='store_true', default=False, help="Use link reconstruction loss")
    parser.add_argument('--link_recon_weight', type=float, default=0.3, help="Weight for link reconstruction loss (reduced to balance with classification)")
    parser.add_argument('--link_sample_rate', type=int, default=2000, help="Number of edges to sample for link prediction")

    # Hidden loss weights
    parser.add_argument('--augmentation_weight', type=float, default=0.2, help='Weight for augmentation guidance loss (reduced to avoid conflict with main teacher)')
    parser.add_argument('--main_distill_weight', type=float, default=0.8, help='Distillation weight for main teacher (primary knowledge source)')
    parser.add_argument('--student_compression_ratio', type=float, default=0.5, help='Compression ratio for student')
     
    # Model saving
    parser.add_argument('--teacher_save_path', type=str, default="teacher_heco.pkl", help="Teacher model save path")
    parser.add_argument('--student_save_path', type=str, default="student_heco.pkl", help="Student model save path")
    parser.add_argument('--middle_teacher_save_path', type=str, default="middle_teacher_heco.pkl", help="Middle teacher save path")
    
    # Hierarchical training parameters
    parser.add_argument('--stage1_epochs', type=int, default=100, help='Epochs for stage 1 - training teacher model')
    parser.add_argument('--stage2_epochs', type=int, default=100, help='Epochs for stage 2 - training student model')
    
    # Knowledge Distillation parameters
    parser.add_argument('--kd_temperature', type=float, default=2.5, help='Temperature for knowledge distillation')
    
    # Augmentation parameters
    parser.add_argument('--use_meta_path_connections', action='store_true', default=True, help="Connect nodes via meta-paths")
    parser.add_argument('--connection_strength', type=float, default=0.2, help="Meta-path connection strength")
    
    # Logging and evaluation
    parser.add_argument('--log_interval', type=int, default=1, help="Logging interval")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval")
    parser.add_argument('--save_interval', type=int, default=100, help="Model saving interval")
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    parser.add_argument('--use_augmentation', type=bool, default=True,
                       help='Enable heterogeneous graph augmentation')
    parser.add_argument('--aug_connection_strength', type=float, default=0.05,
                       help='Strength of meta-path connections (reduced to prevent over-smoothing)')
    parser.add_argument('--aug_low_rank_dim', type=int, default=64,
                       help='Low-rank dimension for augmentation projections')
    parser.add_argument('--aug_auto_generate', type=bool, default=True,
                       help='Auto-generate meta-paths from graph structure')

    args, _ = parser.parse_known_args()
    
    # Dataset-specific configurations
    if args.dataset == "acm":
        args.type_num = [4019, 7167, 60]  # [paper, author, subject]
        args.nei_num = 2
        args.sample_rate = [7, 1] 
    elif args.dataset == "dblp":
        args.type_num = [4057, 14328, 7723, 20]  # [author, paper, conference, term]
        args.nei_num = 1 
        args.sample_rate = [6] 
    elif args.dataset == "aminer":
        args.type_num = [6564, 13329, 35890]  # [paper, author, reference]
        args.nei_num = 2
        args.sample_rate = [3, 8]
    elif args.dataset == "freebase":
        args.type_num = [3492, 2502, 33401, 4459]  # [movie, director, actor, writer]
        args.nei_num = 3
        args.sample_rate = [1, 18, 2]
    
    return args


def get_distillation_config(args):
    """Get distillation configuration dictionary"""
    return {
        'use_kd_loss': getattr(args, 'use_kd_loss', True),
        'use_augmentation_alignment_loss': getattr(args, 'use_augmentation_alignment_loss', True),
        'use_link_recon_loss': getattr(args, 'use_link_recon_loss', False),
        'use_relational_kd_loss': getattr(args, 'use_relational_kd_loss', True),
        'kd_temperature': getattr(args, 'kd_temperature', 2.5),
        'link_recon_weight': getattr(args, 'link_recon_weight', 0.3),
        'link_sample_rate': getattr(args, 'link_sample_rate', 2000),
    }

def get_augmentation_config(args):
    """Get augmentation configuration dictionary"""
    return {
        'use_meta_path_connections': getattr(args, 'use_meta_path_connections', True),
        'connection_strength': getattr(args, 'aug_connection_strength', 0.05),  # Use aug_connection_strength arg
        'num_metapaths': 2  # Explicitly set for ACM (PAP, PSP)
    }


def acm_kd_params():
    """ACM dataset specific parameters for KD"""
    args = kd_params()
    args.dataset = "acm"
    args.hidden_dim = 64
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
        'patience': args.patience
    }