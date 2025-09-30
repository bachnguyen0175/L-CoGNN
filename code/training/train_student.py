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
import random
from tqdm.auto import tqdm

# Add project root and code directory to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, code_dir)

try:
    from models.kd_heco import MiddleMyHeCo, StudentMyHeCo, MyHeCoKD, count_parameters
    from models.kd_params import kd_params, get_distillation_config
    from utils.load_data import load_data
except ImportError:
    # Fallback for different execution contexts
    from code.models.kd_heco import MiddleMyHeCo, StudentMyHeCo, MyHeCoKD, count_parameters
    from code.models.kd_params import kd_params, get_distillation_config
    from code.utils.load_data import load_data
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
        """Initialize both teachers (main + pruning expert) and student models"""
        # Setup augmentation config for dual-teacher system (now with meta-path connections!)
        self.augmentation_config = {
            'use_node_masking': getattr(self.args, 'use_node_masking', True),
            'use_meta_path_connections': getattr(self.args, 'use_meta_path_connections', True),  # NEW: Connect all nodes via meta-paths
            'use_autoencoder': getattr(self.args, 'use_autoencoder', True),
            'mask_rate': getattr(self.args, 'mask_rate', 0.15),  # Higher for better robustness
            'remask_rate': getattr(self.args, 'remask_rate', 0.25),
            'connection_strength': getattr(self.args, 'connection_strength', 0.2),  # NEW: Meta-path connection strength
            'num_remasking': getattr(self.args, 'num_remasking', 3),  # More remasking for expert training
            'autoencoder_hidden_dim': self.args.hidden_dim // 2,
            'autoencoder_layers': 3,  # Deeper for better learning
            'reconstruction_weight': getattr(self.args, 'reconstruction_weight', 0.2)  # Higher weight for reconstruction
        }
        
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
        if hasattr(self.args, 'teacher_model_path') and os.path.exists(self.args.teacher_model_path):
            teacher_checkpoint = torch.load(self.args.teacher_model_path, map_location=self.device)
            if 'model_state_dict' in teacher_checkpoint:
                self.teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
            else:
                self.teacher.load_state_dict(teacher_checkpoint)
            print(f"Loaded main teacher from: {self.args.teacher_model_path}")
        else:
            print("Warning: No main teacher model found. Training without knowledge distillation.")
            self.teacher = None
        
        # Load pre-trained middle teacher (pruning expert trained on augmentation data)
        print("Loading pre-trained pruning expert teacher...")
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
            augmentation_config=self.augmentation_config
        ).to(self.device)

        # Load middle teacher checkpoint - handle compression dimension mismatch
        if hasattr(self.args, 'middle_teacher_path') and os.path.exists(self.args.middle_teacher_path):
            middle_checkpoint = torch.load(self.args.middle_teacher_path, map_location=self.device)
            
            # Handle potential dimension mismatch due to compression
            try:
                if 'model_state_dict' in middle_checkpoint:
                    self.middle_teacher.load_state_dict(middle_checkpoint['model_state_dict'])
                else:
                    self.middle_teacher.load_state_dict(middle_checkpoint)
                print(f"Loaded middle teacher from: {self.args.middle_teacher_path}")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Dimension mismatch detected. Recreating middle teacher with correct dimensions...")
                    # Detect actual hidden dimension from weights
                    state_dict = middle_checkpoint.get('model_state_dict', middle_checkpoint)
                    actual_hidden_dim = None
                    for key, tensor in state_dict.items():
                        if 'fc_list.0.weight' in key:
                            actual_hidden_dim = tensor.shape[0]
                            break
                    
                    if actual_hidden_dim:
                        print(f"Recreating middle teacher with hidden_dim={actual_hidden_dim}")
                        self.middle_teacher = MiddleMyHeCo(
                            feats_dim_list=self.feats_dim_list,
                            hidden_dim=actual_hidden_dim,
                            attn_drop=self.args.attn_drop,
                            feat_drop=self.args.feat_drop,
                            P=self.P,
                            sample_rate=self.args.sample_rate,
                            nei_num=self.args.nei_num,
                            tau=self.args.tau,
                            lam=self.args.lam,
                            augmentation_config=self.augmentation_config
                        ).to(self.device)
                        
                        # Load with corrected dimensions
                        self.middle_teacher.load_state_dict(state_dict)
                        print(f"Successfully loaded middle teacher with corrected dimensions")
                    else:
                        raise e
                else:
                    raise e
        else:
            print("Warning: No middle teacher model found. Training without pruning guidance.")
            self.middle_teacher = None

        # Initialize student model with simplified dual-teacher guidance
        use_middle_teacher_guidance = self.middle_teacher is not None
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
            use_middle_teacher_guidance=use_middle_teacher_guidance
        ).to(self.device)

        # Setup Dual-Teacher KD framework 
        # - Main Teacher: Knowledge distillation (learns from original data)
        # - Middle Teacher (Pruning Expert): Pruning guidance (learns from augmented data)
        # Note: MyHeCoKD is now an alias for DualTeacherKD with backward compatibility
        self.kd_framework = MyHeCoKD(
            teacher=self.teacher,           # Main teacher for knowledge distillation
            student=self.student,           # Student to be trained
            middle_teacher=self.middle_teacher,  # Pruning expert for pruning guidance
            pruning_expert=self.middle_teacher   # Alias for clarity
        ).to(self.device)  # Move to GPU!
        
        # Add KD framework parameters to optimizer if it has trainable parameters
        kd_params = list(self.kd_framework.parameters())
        if kd_params:
            # Create new optimizer with both student and KD framework parameters
            all_params = list(self.student.parameters()) + kd_params
            self.optimizer = torch.optim.Adam(
                all_params, 
                lr=self.args.lr, 
                weight_decay=self.args.l2_coef
            )
        
        # Initialize optimizer
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
            
        if self.middle_teacher:
            middle_params = count_parameters(self.middle_teacher)
            print(f"Pruning expert teacher: {middle_params:,} parameters")
        else:
            print("Pruning expert teacher: Not loaded")
            
        student_params = count_parameters(self.student)
        print(f"Student model: {student_params:,} parameters")
        print(f"Student compression ratio: {self.args.student_compression_ratio:.2f}")
        
        # Print dual-teacher mode info
        if self.teacher and self.middle_teacher:
            print("🚀 Dual-Teacher Mode: Main teacher (knowledge) + Pruning expert (efficiency)")
        elif self.teacher:
            print("📚 Knowledge Distillation Mode: Main teacher only")
        elif self.middle_teacher:
            print("✂️ Pruning Guidance Mode: Pruning expert only")
        else:
            print("🎯 Self-Training Mode: No teacher guidance")
        
        # Check guidance status
        print(f"Middle teacher guidance enabled: {self.student.use_middle_teacher_guidance}")
        if self.student.use_middle_teacher_guidance:
            print("✅ Student will use guidance from middle teacher (pruning expert)")
        else:
            print("ℹ️ Student will train without guidance (standard mode)")
        
    def get_contrastive_nodes(self, batch_size=1024):
        """Get random nodes for contrastive learning"""
        total_nodes = self.feats[0].size(0)
        if batch_size >= total_nodes:
            return torch.arange(total_nodes, device=self.device)
        else:
            return torch.randperm(total_nodes, device=self.device)[:batch_size]
    
    def train_epoch(self):
        """Train for one epoch using enhanced dual-teacher framework with proper distillation and pruning"""
        self.student.train()
        
        # Set teachers to eval mode
        if self.teacher:
            self.teacher.eval()
        if self.middle_teacher:
            self.middle_teacher.eval()
            
        self.optimizer.zero_grad()
        
        # Get contrastive nodes for this batch
        contrastive_nodes = self.get_contrastive_nodes(batch_size=1024)
        
        # 1. Get pruning guidance from middle teacher (pruning expert)
        pruning_guidance = None
        pruning_loss = torch.tensor(0.0, device=self.device)
        if self.middle_teacher:
            with torch.no_grad():
                # Get comprehensive pruning guidance
                pruning_guidance = self.middle_teacher.get_pruning_guidance(
                    self.feats, self.mps, self.nei_index
                )
            
            # Calculate expert alignment loss for better pruning
            if hasattr(self.kd_framework, 'calc_expert_alignment_loss'):
                pruning_loss = self.kd_framework.calc_expert_alignment_loss(
                    self.feats, self.mps, self.nei_index, pruning_guidance
                )
        
        # 2. Forward pass - student loss with pruning guidance
        middle_teacher_guidance = None
        if self.middle_teacher and self.student.use_middle_teacher_guidance:
            with torch.no_grad():
                # Get detailed representations from middle teacher
                mp_guidance, sc_guidance = self.middle_teacher.get_representations(
                    self.feats, self.mps, self.nei_index, use_augmentation=True
                )
                middle_teacher_guidance = {
                    'mp_guidance': mp_guidance.detach(),
                    'sc_guidance': sc_guidance.detach(),
                    'pruning_guidance': pruning_guidance
                }
        
        student_loss = self.student(self.feats, self.pos, self.mps, self.nei_index, 
                                   middle_teacher_guidance=middle_teacher_guidance)
        
        # 3. Enhanced knowledge distillation from BOTH teachers
        kd_loss = torch.tensor(0.0, device=self.device)
        middle_kd_loss = torch.tensor(0.0, device=self.device)
        
        # Main teacher knowledge distillation
        if self.teacher:
            # Use the built-in knowledge distillation framework
            if hasattr(self.kd_framework, 'calc_knowledge_distillation_loss'):
                kd_loss = self.kd_framework.calc_knowledge_distillation_loss(
                    self.feats, self.mps, self.nei_index, distill_config={
                        'distill_weight': self.args.stage2_distill_weight,
                        'temperature': getattr(self.args, 'kd_temperature', 4.0),
                        'nodes': contrastive_nodes,
                        'use_kl_div': getattr(self.args, 'use_kl_div', True),
                        'use_info_nce': getattr(self.args, 'use_info_nce', True)
                    }
                )
            else:
                # Fallback to basic distillation
                with torch.no_grad():
                    teacher_mp, teacher_sc = self.teacher.get_representations(self.feats, self.mps, self.nei_index)
                
                student_mp, student_sc = self.student.get_teacher_aligned_representations(
                    self.feats, self.mps, self.nei_index, middle_teacher_guidance
                )
                
                # Enhanced distillation with InfoNCE and KL divergence
                from models.kd_heco import infoNCE, KLDiverge
                
                # InfoNCE contrastive distillation
                info_nce_mp = infoNCE(student_mp, teacher_mp.detach(), 
                                     contrastive_nodes, temperature=4.0)
                info_nce_sc = infoNCE(student_sc, teacher_sc.detach(), 
                                     contrastive_nodes, temperature=4.0)
                
                # KL divergence for soft target distillation
                kl_mp = KLDiverge(teacher_mp.detach(), student_mp, temperature=4.0)
                kl_sc = KLDiverge(teacher_sc.detach(), student_sc, temperature=4.0)
                
                kd_loss = (info_nce_mp + info_nce_sc + kl_mp + kl_sc) * 0.25
        
        # Middle teacher: ONLY for pruning guidance (NO knowledge distillation)
        # The middle teacher should focus purely on providing structural pruning guidance
        # All knowledge distillation comes from the main teacher only
        
        # 4. Subspace contrastive loss for better representation learning
        subspace_loss = torch.tensor(0.0, device=self.device)
        if self.student.use_middle_teacher_guidance and pruning_guidance:
            try:
                from models.kd_heco import subspace_contrastive_loss_hetero
                
                # Get student representations
                student_mp, student_sc = self.student.get_representations(
                    self.feats, self.mps, self.nei_index, middle_teacher_guidance
                )
                
                # Extract masks from pruning guidance if available
                mp_masks = pruning_guidance.get('mp_importance', None)
                sc_masks = pruning_guidance.get('sc_importance', None)
                
                if mp_masks is not None and sc_masks is not None:
                    subspace_loss = subspace_contrastive_loss_hetero(
                        student_mp, student_sc, mp_masks, sc_masks,
                        contrastive_nodes, temperature=1.0, weight=0.1
                    )
            except Exception as e:
                # Skip subspace loss if there are issues
                pass
        
        # 5. Enhanced loss weighting - prioritize the BETTER teacher!
        total_loss = student_loss
        
        # Proper role assignment: Main teacher for knowledge distillation, middle teacher for pruning only
        if self.teacher and self.middle_teacher:
            # Main teacher: Primary knowledge distillation (full weight)
            main_kd_weight = self.args.stage2_distill_weight  # Full distillation weight for main teacher
            middle_pruning_weight = getattr(self.args, 'pruning_weight', 0.3)  # Pruning guidance only
            
            total_loss += main_kd_weight * kd_loss  # Main teacher KD (primary knowledge source)
            # NO middle teacher knowledge distillation - only pruning guidance
            total_loss += middle_pruning_weight * pruning_loss  # Pruning guidance only
            
            if not hasattr(self, '_log_counter'):
                self._log_counter = 0
            if self._log_counter % 20 == 0:  # Log every 20 steps
                print(f"Loss weights - main_kd: {main_kd_weight:.2f}, pruning_only: {middle_pruning_weight:.2f} (no middle_kd)")
            self._log_counter += 1
            
        elif self.teacher:
            # Fallback to main teacher only
            total_loss += self.args.stage2_distill_weight * kd_loss
            
        elif self.middle_teacher:
            # Use only middle teacher (the better one)
            middle_weight = getattr(self.args, 'pruning_weight', 1.0)
            total_loss += middle_weight * pruning_loss
            
        if subspace_loss > 0:
            total_loss += 0.2 * subspace_loss  # Increase subspace learning weight
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), student_loss.item(), kd_loss.item(), 
                pruning_loss.item(), subspace_loss.item())
    

    
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
            'guidance_enabled': self.student.use_middle_teacher_guidance,
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
            # Enhanced training step with dual-teacher guidance, distillation, and pruning
            loss_tuple = self.train_epoch()
            
            # Handle different return formats (no more middle teacher KD)
            if len(loss_tuple) == 5:
                total_loss, student_loss, kd_loss, pruning_loss, subspace_loss = loss_tuple
            else:
                total_loss, student_loss, kd_loss = loss_tuple
                pruning_loss = subspace_loss = 0.0
            
            # Validation step
            val_loss = self.validate()

            # Update progress bar with corrected metrics (main teacher KD + middle teacher pruning)
            postfix_dict = {
                'total': f"{total_loss:.4f}",
                'student': f"{student_loss:.4f}",
                'main_kd': f"{kd_loss:.4f}",  # Main teacher knowledge distillation
                'prune': f"{pruning_loss:.4f}",  # Middle teacher pruning guidance only
                'subspace': f"{subspace_loss:.4f}",
                'val': f"{val_loss:.4f}",
                'best': f"{self.best_loss:.4f}",
                'patience': f"{self.patience_counter}/{self.args.patience}"
            }
            
            # Add guidance info when available
            if self.student.use_middle_teacher_guidance:
                mp_weight, sc_weight = self.student.get_guidance_fusion_weights()
                if mp_weight is not None:
                    postfix_dict.update({
                        'mp_guide': f"{mp_weight:.3f}",
                        'sc_guide': f"{sc_weight:.3f}"
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
        print(f"Student model parameters: {student_params:,}")
        print(f"Student compression ratio: {self.args.student_compression_ratio:.2f}")
        print(f"Middle teacher guidance was {'enabled' if self.student.use_middle_teacher_guidance else 'disabled'}")
        
        if False:  # Remove sparsity stats code block
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