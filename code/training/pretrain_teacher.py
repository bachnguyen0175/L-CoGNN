"""
Teacher Model Pre-training Script for KD-HGRL
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm.auto import tqdm

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
        self.training_start_time = None
        
    def move_data_to_device(self):
        """Move all data to the specified device"""
        print(f'Moving data to device: {self.device}')
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
        
        # Save best model
        if is_best:
            best_path = self.args.teacher_save_path
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        print("Starting teacher model training...")
        print(f"Training for {self.args.stage1_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        print("-" * 60)
        
        # Start timing
        self.training_start_time = time.time()
        
        progress_bar = tqdm(range(self.args.stage1_epochs), desc="Teacher Training", leave=False, dynamic_ncols=True, unit='epoch')
        for epoch in progress_bar:
            # Epoch-level NumPy seeding for reproducibility of any np-based sampling
            np.random.seed(self.args.seed + epoch)
            # Training step
            train_loss = self.train_epoch()
            
            # Validation step
            val_loss = self.validate()

            # Update progress bar with current metrics
            postfix_dict = {
                'train_loss': f"{train_loss:.4f}", 
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
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                break
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(self.args.teacher_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
        print("Final Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Micro F1: {micro_f1:.4f}")
        
        # Training time statistics
        total_time = time.time() - self.training_start_time
        print("\nTraining Time Statistics:")
        print(f"  Total: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
        print(f"  Epochs: {self.best_epoch}")
        print(f"  Avg per epoch: {total_time/max(self.best_epoch, 1):.2f} seconds")
        
        # Model statistics
        num_params = count_parameters(self.model)
        print(f"\nTeacher model parameters: {num_params:,}")
        
        # Save timing info
        timing_info = {
            'total_seconds': total_time,
            'total_hours': total_time/3600,
            'epochs': self.best_epoch,
            'avg_seconds_per_epoch': total_time / max(self.best_epoch, 1),
            'final_accuracy': accuracy,
            'final_macro_f1': macro_f1,
            'final_micro_f1': micro_f1
        }
        timing_path = f"{self.args.teacher_save_path}_timing.pkl"
        torch.save(timing_info, timing_path)
        print(f"Timing info saved to: {timing_path}")
        
        print("Teacher training completed!")
        return accuracy, macro_f1, micro_f1


def main():
    # Parse arguments
    args = kd_params()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU
        
        # Deterministic behavior (note: may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic CuBLAS operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch versions don't have this
            pass
        except RuntimeError as e:
            # If deterministic algorithms cause issues, warn but continue
            print(f"Warning: Could not enable full deterministic mode: {e}")
            print("Training will continue with partial reproducibility (seeds + cudnn settings)")
    
    # Create trainer and start training
    trainer = TeacherTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
