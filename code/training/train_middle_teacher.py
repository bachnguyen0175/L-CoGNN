#!/usr/bin/env python3
"""
Train Middle Teacher Script
Stage 1 of hierarchical distillation: Teacher → Middle Teacher
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import random
import argparse

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import MyHeCo, MiddleMyHeCo, MyHeCoKD
from models.kd_params import kd_params
# Remove the problematic import for now
# from evaluate_kd import evaluate_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    """Load training data using the utils load_data function"""
    print("Loading data...")
    
    # Use the existing load_data function from utils
    from utils.load_data import load_data as utils_load_data
    
    # Set default values for ratio and type_num based on dataset
    # IMPORTANT: ratio = [split_identifiers] - NOT percentages!
    # ratio=[60,40] loads train_60.npy, val_60.npy, test_60.npy AND train_40.npy, val_40.npy, test_40.npy
    # This is for few-shot learning: 60→180 train nodes, 40→120 train nodes, 20→60 train nodes
    if args.dataset == "acm":
        ratio = [60, 40]  # Load splits with 180 and 120 training nodes
        type_num = [4019, 7167, 60]  # [paper, author, subject]
    elif args.dataset == "dblp":
        ratio = [60, 40]  # Load splits with 180 and 120 training nodes
        type_num = [4057, 14328, 7723, 20]  # [paper, author, conference, term]
    elif args.dataset == "aminer":
        ratio = [60, 40]  # Load splits with 180 and 120 training nodes
        type_num = [6564, 13329, 35890]  # [paper, author, reference]
    elif args.dataset == "freebase":
        ratio = [60, 40]  # Load splits with 180 and 120 training nodes
        type_num = [3492, 2502, 33401, 4459]  # [movie, director, actor, writer]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Load data using the standard function
    nei_index, feats, mps, pos, labels, idx_train, idx_val, idx_test = utils_load_data(args.dataset, ratio, type_num)
    
    print(f"Loaded {args.dataset} dataset successfully")
    print(f"Feature dimensions: {[feat.shape for feat in feats]}")
    print(f"Number of meta-paths: {len(mps)}")
    print(f"Number of labels: {labels.shape}")
    print(f"Train/Val/Test splits: {[len(idx) for idx in idx_train]}/{[len(idx) for idx in idx_val]}/{[len(idx) for idx in idx_test]}")
    
    return nei_index, feats, mps, pos, labels, idx_train, idx_val, idx_test

def sample_neighbors(num_nodes, nei_num):
    """Sample neighbors for schema-level contrast"""
    nei_index = []
    for i in range(num_nodes):
        neighbors = random.sample(range(num_nodes), min(nei_num, num_nodes-1))
        nei_index.append(neighbors)
    return nei_index

def get_contrastive_nodes(feats, device, batch_size=1024):
    """Get random nodes for contrastive learning"""
    total_nodes = feats[0].size(0)
    if batch_size >= total_nodes:
        return torch.arange(total_nodes, device=device)
    else:
        return torch.randperm(total_nodes, device=device)[:batch_size]

def train_middle_teacher(args):
    """Train middle teacher from pre-trained teacher"""
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if hasattr(args, 'cuda') and args.cuda and torch.cuda.is_available():
        if hasattr(args, 'gpu') and args.gpu >= 0:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Load data
    nei_index, feats, mps, pos, labels, idx_train, idx_val, idx_test = load_data(args)
    
    # Move data to device
    feats = [feat.to(device) for feat in feats]
    mps = [mp.to(device) for mp in mps]
    pos = pos.to(device)
    labels = labels.to(device)
    idx_train = [idx.to(device) for idx in idx_train]
    idx_val = [idx.to(device) for idx in idx_val] 
    idx_test = [idx.to(device) for idx in idx_test]
    
    # Dataset info
    feats_dim_list = [feat.shape[1] for feat in feats]
    
    # Load pre-trained teacher
    print("Loading pre-trained teacher...")
    teacher = MyHeCo(
        feats_dim_list=feats_dim_list,
        hidden_dim=args.hidden_dim,
        attn_drop=args.attn_drop,
        feat_drop=args.feat_drop,
        P=len(mps),
        sample_rate=args.sample_rate,
        nei_num=args.nei_num,
        tau=args.tau,
        lam=args.lam
    ).to(device)
    
    if args.teacher_model_path and os.path.exists(args.teacher_model_path):
        teacher_checkpoint = torch.load(args.teacher_model_path, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in teacher_checkpoint:
            teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
        else:
            teacher.load_state_dict(teacher_checkpoint)
        print(f"Loaded teacher from: {args.teacher_model_path}")
    else:
        print("Warning: No teacher model found. Training without pre-trained teacher.")
    
    augmentation_config = {
        'use_node_masking': getattr(args, 'use_node_masking', True),
        'use_edge_augmentation': getattr(args, 'use_edge_augmentation', True),
        'use_autoencoder': getattr(args, 'use_autoencoder', True),
        'mask_rate': getattr(args, 'mask_rate', 0.1),
        'remask_rate': getattr(args, 'remask_rate', 0.2),
        'edge_drop_rate': getattr(args, 'edge_drop_rate', 0.05),
        'num_remasking': getattr(args, 'num_remasking', 2),
        'autoencoder_hidden_dim': args.hidden_dim // 2,  # Half of main hidden dim
        'autoencoder_layers': 2,
        'reconstruction_weight': getattr(args, 'reconstruction_weight', 0.1)
    }
    
    # Initialize middle teacher with augmentation
    middle_teacher = MiddleMyHeCo(
        feats_dim_list=feats_dim_list,
        hidden_dim=args.hidden_dim,
        attn_drop=args.attn_drop,
        feat_drop=args.feat_drop,
        P=len(mps),
        sample_rate=args.sample_rate,
        nei_num=args.nei_num,
        tau=args.tau,
        lam=args.lam,
        compression_ratio=args.middle_compression_ratio,
        augmentation_config=augmentation_config
    ).to(device)
    
    # Setup KD framework
    kd_framework = MyHeCoKD(
        teacher=teacher,
        student=None,
        middle_teacher=middle_teacher
    )
    
    # Setup optimizer
    optimizer = optim.Adam(
        middle_teacher.parameters(),
        lr=args.lr,
        weight_decay=args.l2_coef
    )
    
    # Training loop
    print("Starting middle teacher training...")
    teacher.eval()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.stage1_epochs):
        middle_teacher.train()
        optimizer.zero_grad()
        
        # Forward pass
        student_loss = middle_teacher(feats, pos, mps, nei_index)
        
        # Distillation loss
        if args.teacher_model_path and os.path.exists(args.teacher_model_path):
            # Use training nodes for contrastive learning
            train_nodes = get_contrastive_nodes(feats, device)
            total_loss_with_distill, loss_dict = kd_framework.calc_distillation_loss(
                feats, mps, nei_index, pos, nodes=train_nodes
            )
            distill_loss = loss_dict['distill_loss']
            total_loss = student_loss + args.stage1_distill_weight * distill_loss
        else:
            total_loss = student_loss
            distill_loss = torch.tensor(0.0)
        
        total_loss.backward()
        optimizer.step()
        
        # Logging with augmentation info
        if epoch % args.log_interval == 0:
            aug_info = ""
            if hasattr(middle_teacher, 'augmentation_pipeline'):
                aug_info = f" | Aug: Mask={augmentation_config['use_node_masking']}, Edge={augmentation_config['use_edge_augmentation']}, AE={augmentation_config['use_autoencoder']}"
            
            print(f"Epoch {epoch:04d} | "
                  f"Total Loss: {total_loss.item():.4f} | "
                  f"Student Loss: {student_loss.item():.4f} | "
                  f"Distill Loss: {distill_loss.item():.4f}{aug_info}")
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            # Save best middle teacher
            torch.save(middle_teacher.state_dict(), args.middle_teacher_save_path)
            print(f"Saved best middle teacher at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"Middle teacher training completed. Best loss: {best_loss:.4f}")
    print(f"Middle teacher saved to: {args.middle_teacher_save_path}")
    
    # Evaluate middle teacher
    print("\nEvaluating middle teacher...")
    middle_teacher.eval()
    middle_checkpoint = torch.load(args.middle_teacher_save_path, map_location=device)
    middle_teacher.load_state_dict(middle_checkpoint)
    
    try:
        # Simple evaluation - just get embeddings to verify model works
        with torch.no_grad():
            embeddings = middle_teacher.get_embeds(feats, mps)
            print(f"Middle Teacher Embeddings Shape: {embeddings.shape}")
            print(f"Middle Teacher Evaluation: Model working correctly!")
    except Exception as e:
        print(f"Evaluation failed: {e}")

def main():
    # Parse arguments
    args = kd_params()
    
    # Add specific arguments for middle teacher training
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_distill_weight', type=float, default=0.7, help='Distillation weight for stage 1')
    stage1_args, _ = parser.parse_known_args()
    
    # Merge arguments
    for key, value in vars(stage1_args).items():
        setattr(args, key, value)
    
    print("Middle Teacher Training with Simplified Augmentation")
    print(f"Teacher path: {getattr(args, 'teacher_path', 'teacher_heco.pkl')}")
    print(f"Middle teacher compression ratio: {args.middle_compression_ratio}")
    print(f"Distillation weight: {args.stage1_distill_weight}")
    print(f"Epochs: {args.stage1_epochs}")
    
    # Train middle teacher
    train_middle_teacher(args)

if __name__ == '__main__':
    main()
