"""
Student Model Training with Knowledge Distillation for KD-HGRL
"""

import os
import torch
import numpy as np
import sys
import random

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import MiddleMyHeCo, StudentMyHeCo, MyHeCoKD, count_parameters, calculate_compression_ratio
from models.kd_params import kd_params, get_distillation_config
from utils.load_data import load_data
from utils.evaluate import evaluate_node_classification


class StudentTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        print(f"Loading {args.dataset} dataset...")

        # Set dataset-specific parameters if not already set
        if not hasattr(args, 'type_num'):
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

        # Set default ratio if not provided
        if not hasattr(args, 'ratio'):
            args.ratio = [60, 40]  

        self.nei_index, self.feats, self.mps, self.pos, self.label, self.idx_train, self.idx_val, self.idx_test = load_data(args.dataset, args.ratio, args.type_num)
        
        # Dataset specific parameters
        self.nb_classes = self.label.shape[-1]
        self.feats_dim_list = [feat.shape[1] for feat in self.feats]
        self.P = len(self.mps)
        
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {self.nb_classes}")
        print(f"Feature dimensions: {self.feats_dim_list}")
        print(f"Number of meta-paths: {self.P}")
        print(f"Compression ratio: {args.compression_ratio}")
        
        # Move data to device
        self.move_data_to_device()
        
        # Initialize models
        self.init_models()
        
        # Training metrics
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Distillation config
        self.distill_config = get_distillation_config(args)
        print(f"Distillation config: {self.distill_config}")
        
    def move_data_to_device(self):
        """Move all data to the specified device"""
        if torch.cuda.is_available():
            print(f'Using CUDA device: {self.device}')
            self.feats = [feat.to(self.device) for feat in self.feats]
            self.mps = [mp.to(self.device) for mp in self.mps]
            self.pos = self.pos.to(self.device)
            self.label = self.label.to(self.device)
            self.idx_train = [idx.to(self.device) for idx in self.idx_train]
            self.idx_val = [idx.to(self.device) for idx in self.idx_val]
            self.idx_test = [idx.to(self.device) for idx in self.idx_test]
        
    def init_models(self):
        """Initialize teacher, student, and KD models"""

        augmentation_config = {
            'use_node_masking': getattr(self.args, 'use_node_masking', True),
            'use_edge_augmentation': getattr(self.args, 'use_edge_augmentation', True),
            'use_autoencoder': getattr(self.args, 'use_autoencoder', True),
            'mask_rate': getattr(self.args, 'mask_rate', 0.1),
            'remask_rate': getattr(self.args, 'remask_rate', 0.2),
            'edge_drop_rate': getattr(self.args, 'edge_drop_rate', 0.05),
            'num_remasking': getattr(self.args, 'num_remasking', 2),
            'autoencoder_hidden_dim': self.args.hidden_dim // 2,  # Half of main hidden dim
            'autoencoder_layers': 2,
            'reconstruction_weight': getattr(self.args, 'reconstruction_weight', 0.1)
        }

        # Initialize middle teacher with augmentation
        self.teacher = MiddleMyHeCo(
            feats_dim_list=self.feats_dim_list,
            hidden_dim=self.args.hidden_dim,
            attn_drop=self.args.attn_drop,
            feat_drop=self.args.feat_drop,
            P=self.P,
            sample_rate=self.args.sample_rate,
            nei_num=self.args.nei_num,
            tau=self.args.tau,
            lam=self.args.lam,
            compression_ratio=self.args.middle_compression_ratio,
            augmentation_config=augmentation_config
        ).to(self.device)

        # Load pre-trained middle teacher
        if self.args.middle_teacher_path and os.path.exists(self.args.middle_teacher_path):
            print(f"Loading pre-trained middle teacher from: {self.args.middle_teacher_path}")
            checkpoint = torch.load(self.args.middle_teacher_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.teacher.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state dict format
                self.teacher.load_state_dict(checkpoint)
            
            self.teacher.eval()  # Set teacher to eval mode
        else:
            print("Warning: No pre-trained teacher model found!")
        
        # Initialize student model
        self.student = StudentMyHeCo(
            hidden_dim=self.args.hidden_dim,
            feats_dim_list=self.feats_dim_list,
            feat_drop=self.args.feat_drop,
            attn_drop=self.args.attn_drop,
            P=self.P,
            sample_rate=self.args.sample_rate,
            nei_num=self.args.nei_num,
            tau=self.args.tau,
            lam=self.args.lam,
            compression_ratio=self.args.compression_ratio
        ).to(self.device)
            
        # Initialize KD framework
        self.kd_model = MyHeCoKD(
            teacher=None,
            student=self.student,
            middle_teacher=self.teacher
        ).to(self.device)
        
        # Initialize optimizer (only for student parameters)
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.l2_coef
        )
        
        # Print model info
        teacher_params = count_parameters(self.teacher)
        student_params = count_parameters(self.student)
        compression_ratio = calculate_compression_ratio(self.teacher, self.student)
        
        print(f"Teacher model: {teacher_params:,} parameters")
        print(f"Student model: {student_params:,} parameters")
        print(f"Actual compression ratio: {compression_ratio:.3f}")
        
    def get_contrastive_nodes(self, batch_size=1024):
        """Get random nodes for contrastive learning"""
        total_nodes = self.feats[0].size(0)
        if batch_size >= total_nodes:
            return torch.arange(total_nodes, device=self.device)
        else:
            return torch.randperm(total_nodes, device=self.device)[:batch_size]
    
    def train_epoch(self):
        """Train for one epoch with knowledge distillation"""
        self.student.train()
        self.teacher.eval()  # Keep teacher in eval mode
        self.optimizer.zero_grad()
        
        
        # Get nodes for contrastive learning
        nodes = self.get_contrastive_nodes()
        
        # Forward pass with distillation
        total_loss, losses = self.kd_model.calc_distillation_loss(
            self.feats, self.mps, self.nei_index, self.pos,
            nodes=nodes, distill_config=self.distill_config
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return losses
    
    def train_epoch_enhanced(self, epoch, pruning_run=0):
        """Enhanced training epoch with LightGNN techniques and progressive pruning"""
        self.student.train()
        self.teacher.eval()
        self.optimizer.zero_grad()

        # Get nodes for contrastive learning
        nodes = self.get_contrastive_nodes()

        # Update distillation config with current pruning run
        enhanced_config = self.distill_config.copy()
        enhanced_config['pruning_run'] = pruning_run

        # Dynamic weight adjustment based on training stage
        if hasattr(self.args, 'use_multi_stage') and self.args.use_multi_stage:
            mask_epochs = getattr(self.args, 'mask_epochs', 100)
            if epoch < mask_epochs:
                # Mask training stage - emphasize subspace learning
                enhanced_config['subspace_weight'] *= 1.5
                enhanced_config['self_contrast_weight'] *= 1.2
            else:
                # Fixed training stage - emphasize embedding alignment
                enhanced_config['embedding_weight'] *= 1.3
                enhanced_config['heterogeneous_weight'] *= 1.2

        # Forward pass with enhanced distillation
        total_loss, losses = self.kd_model.calc_distillation_loss(
            self.feats, self.mps, self.nei_index, self.pos,
            nodes=nodes, distill_config=enhanced_config
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return losses

    def apply_progressive_pruning(self, epoch, pruning_config):
        """Apply progressive pruning during training"""
        if not hasattr(self.student, 'apply_progressive_pruning'):
            return

        # Only prune during mask training phase
        mask_epochs = pruning_config.get('mask_epochs', 100)
        pruning_start = pruning_config.get('pruning_start', 10)
        pruning_interval = pruning_config.get('pruning_interval', 10)

        if epoch < mask_epochs and epoch >= pruning_start and epoch % pruning_interval == 0:
            pruning_ratios = {
                'embedding': pruning_config.get('emb_prune_ratio', 0.1),
                'metapath': pruning_config.get('mp_prune_ratio', 0.05)
            }

            print(f"Applying progressive pruning at epoch {epoch}")
            self.student.apply_progressive_pruning(pruning_ratios)

            # Get sparsity stats
            if hasattr(self.student, 'get_sparsity_stats'):
                stats = self.student.get_sparsity_stats()
                print(f"Sparsity stats: {stats}")

    def get_pruning_config(self):
        """Get pruning configuration from args"""
        return {
            'mask_epochs': getattr(self.args, 'mask_epochs', 100),
            'fixed_epochs': getattr(self.args, 'fixed_epochs', 100),
            'pruning_start': getattr(self.args, 'pruning_start', 10),
            'pruning_interval': getattr(self.args, 'pruning_interval', 10),
            'emb_prune_ratio': getattr(self.args, 'emb_prune_ratio', 0.1),
            'mp_prune_ratio': getattr(self.args, 'mp_prune_ratio', 0.05)
        }
    
    def validate(self):
        """Validate the student model"""
        self.student.eval()
        with torch.no_grad():
            nodes = self.get_contrastive_nodes()
            total_loss, losses = self.kd_model.calc_distillation_loss(
                self.feats, self.mps, self.nei_index, self.pos,
                nodes=nodes, distill_config=self.distill_config
            )
        return losses
    
    def evaluate_downstream(self):
        """Evaluate student on downstream node classification task"""
        self.student.eval()
        with torch.no_grad():
            embeds = self.student.get_embeds(self.feats, self.mps)
            
        # Evaluate on first train/val/test split
        accuracy, macro_f1, micro_f1 = evaluate_node_classification(
            embeds, self.idx_train[0], self.idx_val[0], self.idx_test[0], 
            self.label, self.nb_classes, self.device, 
            self.args.eva_lr, self.args.eva_wd
        )
        
        return accuracy, macro_f1, micro_f1
    
    def evaluate_teacher_baseline(self):
        """Evaluate teacher for comparison"""
        self.teacher.eval()
        with torch.no_grad():
            # Clone mps to prevent in-place modifications
            mps_copy = [mp.clone().coalesce() if mp.is_sparse else mp.clone() for mp in self.mps]
            embeds = self.teacher.get_embeds(self.feats, mps_copy)
            
        accuracy, macro_f1, micro_f1 = evaluate_node_classification(
            embeds, self.idx_train[0], self.idx_val[0], self.idx_test[0], 
            self.label, self.nb_classes, self.device, 
            self.args.eva_lr, self.args.eva_wd
        )
        
        return accuracy, macro_f1, micro_f1
    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'compression_ratio': self.args.compression_ratio,
            'args': self.args
        }
        
        # Save regular checkpoint
        if epoch % self.args.save_interval == 0:
            save_path = f"{self.args.student_save_path}_epoch_{epoch}.pkl"
            torch.save(checkpoint, save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")
        
        # Save best model
        if is_best:
            best_path = self.args.student_save_path
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def train_enhanced(self):
        """Enhanced training loop with LightGNN techniques"""
        print("Starting enhanced student model training with LightGNN techniques...")
        print(f"Training for {self.args.nb_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        
        # Check if using multi-stage training
        use_multi_stage = getattr(self.args, 'use_multi_stage', False)
        mask_epochs = getattr(self.args, 'mask_epochs', 100)
        pruning_start = getattr(self.args, 'pruning_start', 1)
        pruning_end = getattr(self.args, 'pruning_end', 5)
        
        if use_multi_stage:
            print(f"Using multi-stage training:")
            print(f"  - Mask training: {mask_epochs} epochs")
            print(f"  - Fixed training: {self.args.nb_epochs - mask_epochs} epochs")
            print(f"  - Pruning runs: {pruning_start} to {pruning_end}")
        
        print("-" * 60)
        
        # Evaluate teacher baseline
        teacher_acc, teacher_macro_f1, teacher_micro_f1 = self.evaluate_teacher_baseline()
        print(f"Teacher Baseline Performance:")
        print(f"  Accuracy: {teacher_acc:.4f}")
        print(f"  Macro F1: {teacher_macro_f1:.4f}")
        print(f"  Micro F1: {teacher_micro_f1:.4f}")
        print("-" * 60)
        
        for epoch in range(self.args.nb_epochs):
            # Determine current pruning run (for multi-stage training)
            current_pruning_run = 0
            if use_multi_stage and epoch >= mask_epochs:
                # Calculate pruning run based on epoch
                fixed_epoch = epoch - mask_epochs
                total_fixed_epochs = self.args.nb_epochs - mask_epochs
                progress = fixed_epoch / max(total_fixed_epochs, 1)
                current_pruning_run = int(pruning_start + progress * (pruning_end - pruning_start))
                current_pruning_run = min(current_pruning_run, pruning_end)
            
            # Training step with enhanced techniques
            train_losses = self.train_epoch_enhanced(epoch, current_pruning_run)
            
            # Validation step
            val_losses = self.validate()
            
            # Logging with enhanced information
            if epoch % self.args.log_interval == 0:
                stage = "Mask" if use_multi_stage and epoch < mask_epochs else "Fixed"
                print(f"Epoch {epoch:4d} [{stage}] | "
                      f"Train Loss: {train_losses['total_loss']:.4f} | "
                      f"Val Loss: {val_losses['total_loss']:.4f}")
                
                # Print detailed loss breakdown
                if 'self_contrast' in train_losses:
                    print(f"  Self-Contrast: {train_losses['self_contrast']:.4f}")
                if 'subspace_contrast' in train_losses:
                    print(f"  Subspace: {train_losses['subspace_contrast']:.4f}")
                if current_pruning_run > 0:
                    print(f"  Pruning Run: {current_pruning_run}")
            
            # Downstream evaluation
            if epoch % self.args.eval_interval == 0:
                accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
                print(f"Epoch {epoch:4d} Downstream | "
                      f"Acc: {accuracy:.4f} | "
                      f"Macro F1: {macro_f1:.4f} | "
                      f"Micro F1: {micro_f1:.4f}")
                
            # Early stopping
            if val_losses['total_loss'] < self.best_loss:
                self.best_loss = val_losses['total_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_model(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Regular model saving
            if epoch % self.args.save_interval == 0:
                self.save_model(epoch)
        
        print("Enhanced student training completed!")
        return self.evaluate_downstream()
    
    def train(self):
        """Main training loop"""
        print("Starting student model training with knowledge distillation...")
        print(f"Training for {self.args.stage2_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        print("-" * 60)
        
        # Evaluate teacher baseline
        teacher_acc, teacher_macro_f1, teacher_micro_f1 = self.evaluate_teacher_baseline()
        print(f"Teacher Baseline Performance:")
        print(f"  Accuracy: {teacher_acc:.4f}")
        print(f"  Macro F1: {teacher_macro_f1:.4f}")
        print(f"  Micro F1: {teacher_micro_f1:.4f}")
        print("-" * 60)
        
        for epoch in range(self.args.stage2_epochs):
            # Training step
            train_losses = self.train_epoch()
            
            # Validation step
            val_losses = self.validate()
            
            # Logging
            if epoch % self.args.log_interval == 0:
                print(f"Epoch {epoch:4d}/{self.args.nb_epochs}")
                print(f"  Train - Total: {train_losses['total_loss']:.4f}, "
                      f"Main: {train_losses['main_loss']:.4f}, "
                      f"Distill: {train_losses['distill_loss']:.4f}")
                print(f"  Val   - Total: {val_losses['total_loss']:.4f}, "
                      f"Main: {val_losses['main_loss']:.4f}, "
                      f"Distill: {val_losses['distill_loss']:.4f}")
                
                if 'embedding_distill' in train_losses:
                    print(f"  Embedding Distill: {train_losses['embedding_distill']:.4f}")
                if 'heterogeneous_distill' in train_losses:
                    print(f"  Heterogeneous Distill: {train_losses['heterogeneous_distill']:.4f}")
            
            # Check for improvement
            is_best = False
            val_total_loss = val_losses['total_loss'].item() if torch.is_tensor(val_losses['total_loss']) else val_losses['total_loss']
            if val_total_loss < self.best_loss:
                self.best_loss = val_total_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
            
            # Save model
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_model(epoch, is_best)
            
            # Downstream evaluation
            if epoch % self.args.eval_interval == 0 and epoch > 0:
                accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
                print(f"Epoch {epoch:4d} | Student Downstream Evaluation:")
                print(f"  Accuracy: {accuracy:.4f} (vs Teacher: {teacher_acc:.4f})")
                print(f"  Macro F1: {macro_f1:.4f} (vs Teacher: {teacher_macro_f1:.4f})")
                print(f"  Micro F1: {micro_f1:.4f} (vs Teacher: {teacher_micro_f1:.4f})")
                print("-" * 40)
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
                break
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(self.args.student_save_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'student_state_dict' in checkpoint:
            self.student.load_state_dict(checkpoint['student_state_dict'])
        else:
            # Direct state dict format
            self.student.load_state_dict(checkpoint)
        
        # Compare student vs teacher
        student_acc, student_macro_f1, student_micro_f1 = self.evaluate_downstream()
        teacher_acc, teacher_macro_f1, teacher_micro_f1 = self.evaluate_teacher_baseline()
        
        print(f"Final Results Comparison:")
        print(f"{'Metric':<15} {'Teacher':<10} {'Student':<10} {'Ratio':<10}")
        print("-" * 50)
        print(f"{'Accuracy':<15} {teacher_acc:<10.4f} {student_acc:<10.4f} {student_acc/teacher_acc:<10.3f}")
        print(f"{'Macro F1':<15} {teacher_macro_f1:<10.4f} {student_macro_f1:<10.4f} {student_macro_f1/teacher_macro_f1:<10.3f}")
        print(f"{'Micro F1':<15} {teacher_micro_f1:<10.4f} {student_micro_f1:<10.4f} {student_micro_f1/teacher_micro_f1:<10.3f}")
        
        # Model statistics
        teacher_params = count_parameters(self.teacher)
        student_params = count_parameters(self.student)
        compression_ratio = calculate_compression_ratio(self.teacher, self.student)
        
        print(f"\nModel Statistics:")
        print(f"  Teacher parameters: {teacher_params:,}")
        print(f"  Student parameters: {student_params:,}")
        print(f"  Compression ratio: {compression_ratio:.3f}")
        print(f"  Parameter reduction: {(1-compression_ratio)*100:.1f}%")
        
        print("Student training completed!")
        return student_acc, student_macro_f1, student_micro_f1


def main():
    # Parse arguments
    args = kd_params()
    args.train_student = True
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = StudentTrainer(args)
    
    # Choose training method
    if getattr(args, 'use_enhanced_training', False):
        trainer.train_enhanced()  # Use enhanced LightGNN techniques
    else:
        trainer.train()  # Use standard training


if __name__ == "__main__":
    main()
