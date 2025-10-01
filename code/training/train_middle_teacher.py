#!/usr/bin/env python3
"""
Train Middle Teacher Script
Stage 1 of hierarchical distillation: Teacher → Middle Teacher
"""

import sys
import torch
import numpy as np
from tqdm.auto import tqdm

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import PruningExpertTeacher, count_parameters
from models.kd_params import kd_params, get_augmentation_config
from utils.load_data import load_data
from utils.evaluate import evaluate_node_classification


class MiddleTeacherTrainer:
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
        """Initialize teacher and middle teacher models"""
        # Load pre-trained teacher
        print("Loading model...")
        
        # Initialize middle teacher with augmentation (now PruningExpertTeacher)
        self.middle_teacher = PruningExpertTeacher(
            feats_dim_list=self.feats_dim_list,
            hidden_dim=self.args.hidden_dim,
            attn_drop=self.args.attn_drop,
            feat_drop=self.args.feat_drop,
            P=self.P,
            sample_rate=self.args.sample_rate,
            nei_num=self.args.nei_num,
            tau=self.args.tau,
            lam=self.args.lam,
            augmentation_config=get_augmentation_config(self.args)
        ).to(self.device)
        
        # Initialize optimized optimizer for RTX 3090
        self.optimizer = torch.optim.AdamW(
            self.middle_teacher.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.l2_coef,
            eps=1e-6,
            fused=True if torch.cuda.is_available() else False
        )
        
        # Print model info
        middle_params = count_parameters(self.middle_teacher)
        print(f"Middle teacher model: {middle_params:,} parameters")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.middle_teacher.train()
        
        # Use multiple forward passes with gradient accumulation for larger effective batch
        accumulation_steps = 2  # Effective batch size = 4096 * 2 = 8192
        expert_loss_accum = 0
        
        self.optimizer.zero_grad()
        
        for step in range(accumulation_steps):
            # Independent Pruning Expert Training - No distillation needed!
            # The expert develops its own expertise on augmented graphs
            expert_loss = self.middle_teacher(self.feats, self.pos, self.mps, self.nei_index)
            
            # Simple independent loss (no distillation)
            loss_step = expert_loss / accumulation_steps
            
            # Backward pass
            loss_step.backward()
            
            # Accumulate losses for logging
            expert_loss_accum += expert_loss.item()
        
            self.optimizer.step()
        
        return expert_loss_accum / accumulation_steps
    
    def validate(self):
        """Validate the model"""
        self.middle_teacher.eval()
        with torch.no_grad():
            val_loss = self.middle_teacher(self.feats, self.pos, self.mps, self.nei_index)
        return val_loss.item()
    
    def evaluate_downstream(self):
        """Evaluate on downstream node classification task"""
        self.middle_teacher.eval()
        with torch.no_grad():
            embeds = self.middle_teacher.get_embeds(self.feats, self.mps)
            
        # Evaluate on first train/val/test split
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
            'model_state_dict': self.middle_teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'args': self.args,
        }
        
        # Save regular checkpoint
        if epoch % self.args.save_interval == 0:
            save_path = f"{self.args.middle_teacher_save_path}_epoch_{epoch}.pkl"
            torch.save(checkpoint, save_path)
        
        # Save best model
        if is_best:
            best_path = self.args.middle_teacher_save_path
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        # FIX 2.7: Updated terminology - this is an augmentation expert, not a pruning expert
        print("Starting augmentation expert training...")
        print("Note: This model learns from augmented graphs, not pruning")
        print(f"Training for {self.args.stage1_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        print("-" * 60)
        
        progress_bar = tqdm(range(self.args.stage1_epochs), desc="Augmentation Expert Training", leave=False, dynamic_ncols=True)
        for epoch in progress_bar:
            # Epoch-level NumPy seeding for reproducibility of any np-based sampling
            np.random.seed(self.args.seed + epoch)
            # Training step
            expert_loss = self.train_epoch()
            
            # Validation step
            val_loss = self.validate()

            # Update progress bar with current metrics
            postfix_dict = {
                'expert_loss': f"{expert_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'best_loss': f"{self.best_loss:.4f}",
                'patience': f"{self.patience_counter}/{self.args.patience}"
            }
            
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
            if expert_loss < self.best_loss:
                self.best_loss = expert_loss
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
        checkpoint = torch.load(self.args.middle_teacher_save_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.middle_teacher.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.middle_teacher.load_state_dict(checkpoint)
        
        accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
        print(f"Final Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Micro F1: {micro_f1:.4f}")
        
        # Model statistics
        middle_params = count_parameters(self.middle_teacher)
        print(f"Middle teacher parameters: {middle_params:,}")
        
        # FIX 2.7: Clarified terminology
        print("Augmentation expert training completed!")
        print("Note: This model provides augmentation guidance, not actual pruning")
        return accuracy, macro_f1, micro_f1

def main():
    # Parse arguments
    args = kd_params()
    args.train_middle_teacher = True
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = MiddleTeacherTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()