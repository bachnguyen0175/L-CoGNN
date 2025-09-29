"""
Student Training Script
Stage 2 of hierarchical distillation: Middle Teacher → Student
"""

import os
import torch
import torch.nn as nn
import numpy as np
import sys
import random
from tqdm.auto import tqdm

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import MiddleMyHeCo, StudentMyHeCo, MyHeCoKD, count_parameters
from models.kd_params import *
from models.kd_params import get_distillation_config
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
            args.ratio = ["80_10_10"]

        self.nei_index, self.feats, self.mps, self.pos, self.label, self.idx_train, self.idx_val, self.idx_test = load_data(args.dataset, args.ratio, args.type_num)
        
        # Dataset specific parameters
        self.nb_classes = self.label.shape[-1]
        self.feats_dim_list = [feat.shape[1] for feat in self.feats]
        self.P = len(self.mps)
        
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {self.nb_classes}")
        print(f"Feature dimensions: {self.feats_dim_list}")
        print(f"Number of meta-paths: {self.P}")
        print(f"Number of node types: {args.type_num}")
        
        # Move data to device
        self.move_data_to_device()
        
        # Initialize models
        self.init_models()
        
        # Training metrics
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
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
        """Initialize middle teacher and student models"""
        # Setup augmentation config for middle teacher
        self.augmentation_config = {
            'use_node_masking': getattr(self.args, 'use_node_masking', True),
            'use_autoencoder': getattr(self.args, 'use_autoencoder', True),
            'mask_rate': getattr(self.args, 'mask_rate', 0.1),
            'remask_rate': getattr(self.args, 'remask_rate', 0.2),
            'edge_drop_rate': getattr(self.args, 'edge_drop_rate', 0.05),
            'num_remasking': getattr(self.args, 'num_remasking', 2),
            'autoencoder_hidden_dim': self.args.hidden_dim // 2,
            'autoencoder_layers': 2,
            'reconstruction_weight': getattr(self.args, 'reconstruction_weight', 0.1)
        }
        
        # Load pre-trained middle teacher
        print("Loading pre-trained middle teacher...")
        self.middle_teacher = MiddleMyHeCo(
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
            augmentation_config=self.augmentation_config
        ).to(self.device)

        # Load middle teacher checkpoint
        if hasattr(self.args, 'middle_teacher_path') and os.path.exists(self.args.middle_teacher_path):
            middle_checkpoint = torch.load(self.args.middle_teacher_path, map_location=self.device)
            if 'model_state_dict' in middle_checkpoint:
                self.middle_teacher.load_state_dict(middle_checkpoint['model_state_dict'])
            else:
                self.middle_teacher.load_state_dict(middle_checkpoint)
            print(f"Loaded middle teacher from: {self.args.middle_teacher_path}")
        else:
            print("Warning: No middle teacher model found. Training without pre-trained middle teacher.")

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
            compression_ratio=self.args.student_compression_ratio,
            enable_pruning=True
        ).to(self.device)

        # Setup KD framework - Middle Teacher to Student mode
        self.kd_framework = MyHeCoKD(
            teacher=None,
            student=self.student,
            middle_teacher=self.middle_teacher
        )
        # Will add KD student head params to optimizer on first use
        self._kd_head_added_to_optim = False
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.l2_coef
        )
        
        # Print model info
        middle_params = count_parameters(self.middle_teacher)
        student_params = count_parameters(self.student)
        print(f"Middle teacher model: {middle_params:,} parameters")
        print(f"Student model: {student_params:,} parameters")
        print(f"Student compression ratio: {self.args.student_compression_ratio:.2f}")
        
        # Check pruning status
        print(f"Pruning enabled: {self.student.enable_pruning}")
        if hasattr(self.student, 'mp_attention_mask'):
            print(f"Attention masks initialized: MP={self.student.mp_attention_mask.shape}, SC={self.student.sc_attention_mask.shape}")
        else:
            print("Warning: Attention masks not found!")
        
        # Print initial sparsity stats
        initial_sparsity = self.get_sparsity_stats()
        print(f"Initial sparsity stats: {initial_sparsity}")
        
    def get_contrastive_nodes(self, batch_size=1024):
        """Get random nodes for contrastive learning"""
        total_nodes = self.feats[0].size(0)
        if batch_size >= total_nodes:
            return torch.arange(total_nodes, device=self.device)
        else:
            return torch.randperm(total_nodes, device=self.device)[:batch_size]
    
    def train_epoch(self):
        """Train for one epoch"""
        self.student.train()
        self.middle_teacher.eval()  # Set middle teacher to eval mode
        self.optimizer.zero_grad()
        
        # Forward pass - student contrastive loss
        student_loss = self.student(self.feats, self.pos, self.mps, self.nei_index)
        
        # Distillation loss from middle teacher
        if hasattr(self.args, 'middle_teacher_path') and os.path.exists(self.args.middle_teacher_path):
            train_nodes = self.get_contrastive_nodes()
            # Build and pass distillation config using current args
            distill_config = get_distillation_config(self.args)
            distill_config['num_classes'] = self.nb_classes
            total_loss_with_distill, loss_dict = self.kd_framework.calc_distillation_loss(
                self.feats, self.mps, self.nei_index, self.pos, nodes=train_nodes, distill_config=distill_config
            )
            distill_loss = loss_dict['distill_loss']
            # Include KD student prediction head params in optimizer once available
            if (not self._kd_head_added_to_optim and
                hasattr(self.kd_framework, 'student_prediction_classifier') and
                isinstance(self.kd_framework.student_prediction_classifier, nn.Module)):
                self.optimizer.add_param_group({
                    'params': self.kd_framework.student_prediction_classifier.parameters(),
                    'lr': self.args.lr,
                    'weight_decay': self.args.l2_coef
                })
                self._kd_head_added_to_optim = True
            total_loss = student_loss + self.args.stage2_distill_weight * distill_loss
        else:
            total_loss = student_loss
            distill_loss = torch.tensor(0.0)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), student_loss.item(), distill_loss.item()
    
    def apply_progressive_attention_pruning(self, epoch):
        """Apply progressive attention pruning with optimal timing"""
        if not hasattr(self.student, 'apply_progressive_attention_pruning'):
            return
            
        # Phase 1: Warm-up (epochs 0-19) - No pruning, let model stabilize
        if epoch < 20:
            return
            
        # Phase 2: Early pruning (epochs 20-49) - Gentle pruning start
        elif epoch < 50:
            if epoch % 5 == 0:  # Every 5 epochs
                pruning_ratios = {
                    'current_epoch': epoch - 20,  # Adjust for pruning start
                    'max_epochs': 80,  # Total pruning epochs (50-20=30, plus 50 more)
                    'min_attention_sparsity': 0.1,  # Start very gentle
                    'max_attention_sparsity': 0.6,  # Target 60% sparsity
                    'attention_sparsity_weight': 0.001  # Low weight initially
                }
                self.student.apply_progressive_attention_pruning(pruning_ratios)
                
        # Phase 3: Aggressive pruning (epochs 50-99) - Main pruning phase  
        elif epoch < 100:
            if epoch % 3 == 0:  # Every 3 epochs for more frequent updates
                pruning_ratios = {
                    'current_epoch': epoch - 20,
                    'max_epochs': 80,
                    'min_attention_sparsity': 0.1,
                    'max_attention_sparsity': 0.7,  # Higher target
                    'attention_sparsity_weight': 0.005  # Standard weight
                }
                self.student.apply_progressive_attention_pruning(pruning_ratios)
                
        # Phase 4: Fine-tuning (epochs 100+) - Stabilize pruned model
        else:
            if epoch % 10 == 0:  # Less frequent updates
                pruning_ratios = {
                    'current_epoch': 80,  # Keep at max pruning
                    'max_epochs': 80,
                    'min_attention_sparsity': 0.1,
                    'max_attention_sparsity': 0.8,  # Final high sparsity
                    'attention_sparsity_weight': 0.002  # Reduce weight for stability
                }
                self.student.apply_progressive_attention_pruning(pruning_ratios)
    
    def validate(self):
        """Validate the model"""
        self.student.eval()
        with torch.no_grad():
            val_loss = self.student(self.feats, self.pos, self.mps, self.nei_index)
        return val_loss.item()
    
    def evaluate_downstream(self):
        """Evaluate on downstream node classification task"""
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
    
    def get_sparsity_stats(self):
        """Get current sparsity statistics"""
        if hasattr(self.student, 'get_sparsity_stats'):
            return self.student.get_sparsity_stats()
        return {}
    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'args': self.args,
            'augmentation_config': self.augmentation_config,
            'sparsity_stats': self.get_sparsity_stats(),
            'compression_ratio': self.args.student_compression_ratio
        }
        
        # Save regular checkpoint
        if epoch % self.args.save_interval == 0:
            save_path = f"{self.args.student_save_path}_epoch_{epoch}.pkl"
            torch.save(checkpoint, save_path)
        
        # Save best model
        if is_best:
            best_path = self.args.student_save_path
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        print("Starting student training...")
        print(f"Training for {self.args.stage2_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        print(f"Distillation weight: {self.args.stage2_distill_weight}")
        print(f"Augmentation config: {self.augmentation_config}")
        print("-" * 60)
        
        progress_bar = tqdm(range(self.args.stage2_epochs), desc="Student Training", leave=False, dynamic_ncols=True)
        for epoch in progress_bar:
            # Epoch-level NumPy seeding for reproducibility of any np-based sampling
            np.random.seed(self.args.seed + epoch)
            # Training step
            total_loss, student_loss, distill_loss = self.train_epoch()
            
            # Apply progressive attention pruning
            self.apply_progressive_attention_pruning(epoch)
            
            # Validation step
            val_loss = self.validate()

            # Update progress bar with current metrics
            postfix_dict = {
                'total_loss': f"{total_loss:.4f}",
                'student_loss': f"{student_loss:.4f}",
                'distill_loss': f"{distill_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'best_loss': f"{self.best_loss:.4f}",
                'patience': f"{self.patience_counter}/{self.args.patience}"
            }
            
            # Add sparsity stats periodically
            if epoch % self.args.log_interval == 0:
                sparsity_stats = self.get_sparsity_stats()
                if sparsity_stats:
                    mp_att_sparsity = sparsity_stats.get('metapath_attention_sparsity', 1.0)
                    sc_att_sparsity = sparsity_stats.get('schema_attention_sparsity', 1.0)
                    postfix_dict.update({
                        'mp_att': f"{mp_att_sparsity:.3f}",
                        'sc_att': f"{sc_att_sparsity:.3f}"
                    })
            
            # Add evaluation metrics when available
            if epoch % self.args.eval_interval == 0 and epoch > 0:
                accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
                postfix_dict.update({
                    'acc': f"{accuracy:.3f}",
                    'macro_f1': f"{macro_f1:.3f}",
                    'micro_f1': f"{micro_f1:.3f}"
                })
            
            progress_bar.set_postfix(postfix_dict)
            
            # Check for improvement
            is_best = False
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
            
            # Save model
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_model(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                break
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(self.args.student_save_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.student.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.student.load_state_dict(checkpoint)
        
        accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
        print(f"Final Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Micro F1: {micro_f1:.4f}")
        
        # Model statistics
        student_params = count_parameters(self.student)
        final_sparsity = self.get_sparsity_stats()
        print(f"Student model parameters: {student_params:,}")
        print(f"Student compression ratio: {self.args.compression_ratio:.2f}")
        
        if final_sparsity:
            print(f"Final Sparsity Statistics:")
            for key, value in final_sparsity.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            print(f"    {sub_key}: {sub_value:.3f}")
                        else:
                            print(f"    {sub_key}: {sub_value}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        print("Student training completed!")
        return accuracy, macro_f1, micro_f1

def main():
    # Parse arguments
    args = kd_params()
    args.train_student = True
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = StudentTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()