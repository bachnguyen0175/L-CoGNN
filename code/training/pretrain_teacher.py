"""
Teacher Model Pre-training Script for KD-HGRL
"""

import torch
import numpy as np
import sys

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import MyHeCo, count_parameters
from models.kd_params import kd_params
from utils.load_data import load_data
from utils.evaluate import evaluate_node_classification


class TeacherTrainer:
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
        
        print(f"Number of nodes: {self.label.shape[0]}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {self.nb_classes}")
        print(f"Feature dimensions: {self.feats_dim_list}")
        print(f"Number of meta-paths: {self.P}")
        print(f"Number of node types: {args.type_num}")
        
        # Move data to device
        self.move_data_to_device()
        
        # Initialize model
        self.init_model()
        
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
        
    def init_model(self):
        """Initialize the teacher model"""
        self.model = MyHeCo(
            hidden_dim=self.args.hidden_dim,
            feats_dim_list=self.feats_dim_list,
            feat_drop=self.args.feat_drop,
            attn_drop=self.args.attn_drop,
            P=self.P,
            sample_rate=self.args.sample_rate,
            nei_num=self.args.nei_num,
            tau=self.args.tau,
            lam=self.args.lam
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.l2_coef
        )
        
        # Print model info
        num_params = count_parameters(self.model)
        print(f"Teacher model initialized with {num_params:,} parameters")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        loss = self.model(self.feats, self.pos, self.mps, self.nei_index)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        with torch.no_grad():
            val_loss = self.model(self.feats, self.pos, self.mps, self.nei_index)
        return val_loss.item()
    
    def evaluate_downstream(self):
        """Evaluate on downstream node classification task"""
        self.model.eval()
        with torch.no_grad():
            embeds = self.model.get_embeds(self.feats, self.mps)
            
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'args': self.args
        }
        
        # Save regular checkpoint
        if epoch % self.args.save_interval == 0:
            save_path = f"{self.args.teacher_save_path}_epoch_{epoch}.pkl"
            torch.save(checkpoint, save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")
        
        # Save best model
        if is_best:
            best_path = self.args.teacher_save_path
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting teacher model training...")
        print(f"Training for {self.args.nb_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        print("-" * 60)
        
        for epoch in range(self.args.nb_epochs):
            # Training step
            train_loss = self.train_epoch()
            
            # Validation step
            val_loss = self.validate()
            
            # Logging
            if epoch % self.args.log_interval == 0:
                print(f"Epoch {epoch:4d}/{self.args.nb_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            is_best = False
            if val_loss < self.best_loss:
                self.best_loss = val_loss
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
                print(f"Epoch {epoch:4d} | Downstream Evaluation:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Macro F1: {macro_f1:.4f}")
                print(f"  Micro F1: {micro_f1:.4f}")
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
        checkpoint = torch.load(self.args.teacher_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
        print(f"Final Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Micro F1: {micro_f1:.4f}")
        
        # Model statistics
        num_params = count_parameters(self.model)
        print(f"Teacher model parameters: {num_params:,}")
        
        print("Teacher training completed!")
        return accuracy, macro_f1, micro_f1


def main():
    # Parse arguments
    args = kd_params()
    args.train_teacher = True
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = TeacherTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
