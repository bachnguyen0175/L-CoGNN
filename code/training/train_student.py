"""
Student Training Script
Stage 2 of hierarchical distillation: Middle Teacher → Student
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from tqdm.auto import tqdm

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import AugmentationTeacher, StudentMyHeCo, DualTeacherKD, count_parameters
from models.kd_params import kd_params, get_distillation_config, get_augmentation_config
from utils.load_data import load_data
from utils.evaluate import evaluate_node_classification


class StudentTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        print(f"Loading {args.dataset} dataset...")

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
        
        # Set up augmentation config for middle teacher
        self.augmentation_config = get_augmentation_config(args)
        
        # Adaptive weighting for dual-teacher conflict resolution
        self.teacher_agreement_history = []
        self.adaptive_aug_weight = args.augmentation_weight  # Start with config value
        
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
        """Initialize both teachers (main + augmentation) and student models"""
        # Load pre-trained main teacher (trained on original data for knowledge distillation)
        print("Loading pre-trained main teacher...")
        from models.kd_heco import MyHeCo
        self.teacher = MyHeCo(
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

        # Load teacher checkpoint
        teacher_path = os.path.abspath(self.args.teacher_model_path) if hasattr(self.args, 'teacher_model_path') else None
        if teacher_path and os.path.exists(teacher_path):
            teacher_checkpoint = torch.load(teacher_path, map_location=self.device)
            if 'model_state_dict' in teacher_checkpoint:
                self.teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
            else:
                self.teacher.load_state_dict(teacher_checkpoint)
            print(f"Loaded main teacher from: {teacher_path}")
        else:
            print("Warning: No main teacher model found. Training without knowledge distillation.")
            self.teacher = None
        
        # Load pre-trained middle teacher (augmentation trained on augmented meta-path graphs)
        print("Loading pre-trained augmentation teacher...")
        
        self.middle_teacher = AugmentationTeacher(
            feats_dim_list=self.feats_dim_list,
            hidden_dim=self.args.hidden_dim,
            attn_drop=self.args.attn_drop,
            feat_drop=self.args.feat_drop,
            P=self.P,
            sample_rate=self.args.sample_rate,
            nei_num=self.args.nei_num,
            tau=self.args.tau,
            lam=self.args.lam,
            augmentation_config=self.augmentation_config
        ).to(self.device)

        # Load middle teacher checkpoint - handle compression dimension mismatch
        middle_teacher_path = os.path.abspath(self.args.middle_teacher_path) if hasattr(self.args, 'middle_teacher_path') else None
        if middle_teacher_path and os.path.exists(middle_teacher_path):
            middle_checkpoint = torch.load(middle_teacher_path, map_location=self.device)
            
            # Handle potential dimension mismatch due to compression
            try:
                if 'model_state_dict' in middle_checkpoint:
                    self.middle_teacher.load_state_dict(middle_checkpoint['model_state_dict'])
                else:
                    self.middle_teacher.load_state_dict(middle_checkpoint)
                print(f"Loaded middle teacher from: {middle_teacher_path}")
            except RuntimeError as e:
                raise e
        else:
            print("Warning: No middle teacher model found")
            self.middle_teacher = None

        # Initialize student model with simplified dual-teacher guidance
        use_augmentation_teacher_guidance = self.middle_teacher is not None
        
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
            use_augmentation_teacher_guidance=use_augmentation_teacher_guidance
        ).to(self.device)

        self.kd_framework = DualTeacherKD(
            teacher=self.teacher,           # Main teacher for knowledge distillation
            student=self.student,           # Student to be trained
            augmentation_teacher=self.middle_teacher
        ).to(self.device)  # Move to GPU!
        
        # Add KD framework parameters to optimizer if it has trainable parameters
        kd_params = list(self.kd_framework.parameters())
        if kd_params:
            # Create optimizer with both student and KD framework parameters
            all_params = list(self.student.parameters()) + kd_params
            self.optimizer = torch.optim.Adam(
                all_params, 
                lr=self.args.lr, 
                weight_decay=self.args.l2_coef
            )
        else:
            # Fallback: optimize student only
            self.optimizer = torch.optim.Adam(
                self.student.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.l2_coef
            )
        
        # Print model info
        if self.teacher:
            teacher_params = count_parameters(self.teacher)
            print(f"Main teacher model: {teacher_params:,} parameters")
        else:
            print("Main teacher: Not loaded")
            
        # Updated terminology
        if self.middle_teacher:
            middle_params = count_parameters(self.middle_teacher)
            print(f"Augmentation teacher: {middle_params:,} parameters")
        else:
            print("Augmentation teacher: Not loaded")

        student_params = count_parameters(self.student)
        print(f"Student model: {student_params:,} parameters")
        print(f"Student compression ratio: {self.args.student_compression_ratio:.2f}")
        
        # Print dual-teacher mode info
        if self.teacher and self.middle_teacher:
            print("Dual-Teacher Mode: Main teacher (knowledge distillation) + Augmentation (robustness)")
            print(f"   Initial weights: main={self.args.main_distill_weight:.2f}, aug={self.args.augmentation_weight:.2f}")
        elif self.teacher:
            print("Knowledge Distillation Mode: Main teacher only")
        elif self.middle_teacher:
            print("Augmentation Guidance Mode: Augmentation only")
        else:
            print("Self-Training Mode: No teacher guidance")
    
    def train_epoch(self, epoch=0):
        """Train for one epoch using enhanced dual-teacher framework with proper distillation"""
        self.student.train()
        
        # Set teachers to eval mode
        if self.teacher:
            self.teacher.eval()
        if self.middle_teacher:
            self.middle_teacher.eval()
            
        self.optimizer.zero_grad()
        
        # Get augmentation guidance from middle teacher (augmentation)
        augmentation_guidance = None
        augmentation_alignment_loss = torch.tensor(0.0, device=self.device)
        if self.middle_teacher and self.args.use_augmentation_alignment_loss:
            with torch.no_grad():
                # Get comprehensive augmentation guidance
                augmentation_guidance = self.middle_teacher.get_augmentation_guidance(
                    self.feats, self.mps, self.nei_index
                )

            # Calculate augmentation alignment loss for better guidance learning
            if hasattr(self.kd_framework, 'calc_augmentation_alignment_loss'):
                augmentation_alignment_loss = self.kd_framework.calc_augmentation_alignment_loss(
                    self.feats, self.mps, self.nei_index, augmentation_guidance
                )
        
        # 2. Forward pass - student loss with augmentation guidance
        # Clarified terminology - middle teacher provides AUGMENTATION guidance
        augmentation_teacher_guidance = None
        if self.middle_teacher and self.student.use_augmentation_teacher_guidance:
            with torch.no_grad():
                # Get detailed representations from augmentation (middle teacher)
                mp_guidance, sc_guidance = self.middle_teacher.get_representations(
                    self.feats, self.mps, self.nei_index, use_augmentation=True
                )
                augmentation_teacher_guidance = {
                    'mp_guidance': mp_guidance.detach(),  # Augmented meta-path embeddings
                    'sc_guidance': sc_guidance.detach(),  # Augmented schema embeddings
                    'augmentation_guidance': augmentation_guidance  # Structural guidance from augmentation
                }
        
        student_loss = self.student(self.feats, self.pos, self.mps, self.nei_index, 
                                   augmentation_teacher_guidance=augmentation_teacher_guidance)
        
        # 3. Knowledge distillation from main teacher
        kd_loss = torch.tensor(0.0, device=self.device)
        
        # Main teacher
        if self.teacher and self.args.use_kd_loss:
            if hasattr(self.kd_framework, 'calc_knowledge_distillation_loss'):
                # Standard embedding-level KD
                kd_loss = self.kd_framework.calc_knowledge_distillation_loss(
                    self.feats, self.mps, self.nei_index, distill_config=get_distillation_config(self.args),
                )
            else:
                print("No distillation method found in KD framework.")
        
        # 4.5. Link prediction losses for better edge modeling
        link_recon_loss = torch.tensor(0.0, device=self.device)
        student_embeds = None
        
        # Link reconstruction loss on student embeddings
        if self.args.use_link_recon_loss and (epoch % 5 == 0 or epoch < 50):
            try:
                from models.kd_heco import (link_reconstruction_loss,
                                            sample_edges_from_metapaths, sample_negative_edges)

                # Get student embeddings
                if student_embeds is None:
                    student_embeds = self.student.get_embeds(self.feats, self.mps, detach=False)
                
                # Sample positive edges from meta-paths
                pos_edges = sample_edges_from_metapaths(self.mps, num_samples=2000)
                
                if pos_edges is not None and len(pos_edges) > 0:
                    # Move to correct device
                    pos_edges = pos_edges.to(self.device)
                    
                    # Sample negative edges
                    num_nodes = student_embeds.size(0)
                    neg_edges = sample_negative_edges(num_nodes, len(pos_edges), pos_edges)
                    neg_edges = neg_edges.to(self.device)
                    
                    # Compute link reconstruction loss
                    link_recon_loss = link_reconstruction_loss(
                        student_embeds, pos_edges, neg_edges, temperature=1.0
                    )
                else:
                    if epoch == 0:
                        print("Warning: No edges sampled from meta-paths!")
            except Exception as e:
                if epoch == 0 or epoch % 50 == 0:
                    print(f"Warning: Link reconstruction error at epoch {epoch}: {e}")
    
        total_loss = 0
        total_loss += student_loss
        
        # Role assignment with ADAPTIVE WEIGHTING:
        if self.teacher and self.middle_teacher:
            main_kd_weight = self.args.main_distill_weight
            
            total_loss += main_kd_weight * kd_loss  # Main teacher embedding KD
            total_loss += self.args.augmentation_weight * augmentation_alignment_loss  # Adaptive augmentation guidance

        elif self.teacher:
            total_loss += self.args.main_distill_weight * kd_loss
            
        elif self.middle_teacher:
            total_loss += self.args.augmentation_weight * augmentation_alignment_loss

        if link_recon_loss > 0:
            total_loss += self.args.link_recon_weight * link_recon_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), student_loss.item(), kd_loss.item(), 
                augmentation_alignment_loss.item(), link_recon_loss.item())
    
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
    

    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'args': self.args,
            'augmentation_config': self.augmentation_config,
            'guidance_enabled': self.student.use_augmentation_teacher_guidance,
            'compression_ratio': self.args.student_compression_ratio
        }
        
        # Save regular checkpoint
        if epoch % self.args.save_interval == 0:
            save_path = f"{os.path.abspath(self.args.student_save_path)}_epoch_{epoch}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
        
        # Save best model
        if is_best:
            best_path = os.path.abspath(self.args.student_save_path)
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        print("Starting student training...")
        print(f"Training for {self.args.stage2_epochs} epochs")
        print(f"Patience: {self.args.patience}")
        print("-" * 60)
        
        progress_bar = tqdm(range(self.args.stage2_epochs), desc="Student Training", leave=False, dynamic_ncols=True)
        for epoch in progress_bar:
            # Epoch-level NumPy seeding for reproducibility of any np-based sampling
            np.random.seed(self.args.seed + epoch)
            loss_tuple = self.train_epoch(epoch=epoch)

            total_loss, student_loss, kd_loss, augmentation_alignment_loss, link_recon_loss = loss_tuple

            # Validation step
            val_loss = self.validate()
            is_best = False

            postfix_dict = {
                'total': f"{total_loss:.4f}",
                'student': f"{student_loss:.4f}",
                'main_kd': f"{kd_loss:.4f}",  # Main teacher knowledge distillation
                'aug_align': f"{augmentation_alignment_loss:.4f}",  # Augmentation alignment
                'link_rec': f"{link_recon_loss:.4f}",  # Link reconstruction
                'val': f"{val_loss:.4f}",
                'best': f"{self.best_loss:.4f}",
                'patience': f"{self.patience_counter}/{self.args.patience}"
            }         
            
            # Add evaluation metrics
            if epoch % self.args.eval_interval == 0 and epoch > 0:
                accuracy, macro_f1, micro_f1 = self.evaluate_downstream()
                postfix_dict.update({
                    'acc': f"{accuracy:.3f}",
                    'macro_f1': f"{macro_f1:.3f}",
                    'micro_f1': f"{micro_f1:.3f}"
                })
            
            progress_bar.set_postfix(postfix_dict)
            

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
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(os.path.abspath(self.args.student_save_path), map_location=self.device)
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
        print(f"Student model parameters: {student_params:,}")
        print(f"Student compression ratio: {self.args.student_compression_ratio:.2f}")
        print(f"Augmentation teacher guidance was {'enabled' if self.student.use_augmentation_teacher_guidance else 'disabled'}")
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
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU
        
        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic CuBLAS operations
        import os
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
    trainer = StudentTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()