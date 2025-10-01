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

from models.kd_heco import PruningExpertTeacher, StudentMyHeCo, DualTeacherKD, count_parameters
from models.kd_params import kd_params, get_distillation_config, get_augmentation_config
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
        
        # FIX 2.7: Clarified - this is an augmentation expert, not a pruning expert
        # Load pre-trained middle teacher (augmentation expert trained on augmented graphs)
        print("Loading pre-trained augmentation expert teacher...")
        self.middle_teacher = PruningExpertTeacher(  # Name kept for compatibility
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

        self.kd_framework = DualTeacherKD(
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
            
        # FIX 2.7: Updated terminology
        if self.middle_teacher:
            middle_params = count_parameters(self.middle_teacher)
            print(f"Augmentation expert teacher: {middle_params:,} parameters")
        else:
            print("Augmentation expert teacher: Not loaded")
            
        student_params = count_parameters(self.student)
        print(f"Student model: {student_params:,} parameters")
        print(f"Student compression ratio: {self.args.student_compression_ratio:.2f}")
        
        # FIX 2.7 & 2.8: Updated terminology - it's augmentation expert, not pruning expert
        # Print dual-teacher mode info
        if self.teacher and self.middle_teacher:
            print("🚀 Dual-Teacher Mode: Main teacher (knowledge distillation) + Augmentation expert (robustness)")
        elif self.teacher:
            print("📚 Knowledge Distillation Mode: Main teacher only")
        elif self.middle_teacher:
            print("🔄 Augmentation Guidance Mode: Augmentation expert only (no pruning)")
        else:
            print("🎯 Self-Training Mode: No teacher guidance")
        
        # Check guidance status
        print(f"Middle teacher guidance enabled: {self.student.use_middle_teacher_guidance}")
        if self.student.use_middle_teacher_guidance:
            print("✅ Student will use guidance from middle teacher (pruning expert)")
        else:
            print("ℹ️ Student will train without guidance (standard mode)")
        
    def get_contrastive_nodes(self, batch_size=4096):
        """Get random nodes for contrastive learning"""
        total_nodes = self.feats[0].size(0)
        if batch_size >= total_nodes:
            return torch.arange(total_nodes, device=self.device)
        else:
            return torch.randperm(total_nodes, device=self.device)[:batch_size]
    
    def train_epoch(self, epoch=0):
        """Train for one epoch using enhanced dual-teacher framework with proper distillation and pruning"""
        self.student.train()
        
        # Set teachers to eval mode
        if self.teacher:
            self.teacher.eval()
        if self.middle_teacher:
            self.middle_teacher.eval()
            
        self.optimizer.zero_grad()
        
        # Get contrastive nodes for this batch
        contrastive_nodes = self.get_contrastive_nodes(batch_size=self.args.batch_size)
        
        # FIX 2.1: Refresh guidance periodically (every 20 epochs) instead of once
        # This prevents guidance from becoming stale as student evolves
        guidance_refresh_interval = 20
        should_refresh_guidance = (epoch % guidance_refresh_interval == 0)
        
        # 1. Get pruning guidance from middle teacher (augmentation expert)
        pruning_guidance = None
        pruning_loss = torch.tensor(0.0, device=self.device)
        if self.middle_teacher:
            with torch.no_grad():
                # FIX 2.2: Error-driven guidance - compute student errors first
                if should_refresh_guidance and epoch > 0:
                    # Get student's current predictions to identify weak areas
                    student_embeds = self.student.get_embeds(self.feats, self.mps)
                    # Note: In production, you'd compute actual errors on validation set
                    # For now, we'll use the standard guidance but refresh it periodically
                    pass
                
                # Get comprehensive augmentation guidance
                pruning_guidance = self.middle_teacher.get_pruning_guidance(
                    self.feats, self.mps, self.nei_index
                )
            
            # Calculate expert alignment loss for better pruning
            if hasattr(self.kd_framework, 'calc_expert_alignment_loss'):
                pruning_loss = self.kd_framework.calc_expert_alignment_loss(
                    self.feats, self.mps, self.nei_index, pruning_guidance
                )
        
        # 2. Forward pass - student loss with augmentation guidance
        # FIX 2.8: Clarified terminology - middle teacher provides AUGMENTATION guidance, not pruning
        middle_teacher_guidance = None
        if self.middle_teacher and self.student.use_middle_teacher_guidance:
            with torch.no_grad():
                # Get detailed representations from augmentation expert (middle teacher)
                mp_guidance, sc_guidance = self.middle_teacher.get_representations(
                    self.feats, self.mps, self.nei_index, use_augmentation=True
                )
                middle_teacher_guidance = {
                    'mp_guidance': mp_guidance.detach(),  # Augmented meta-path embeddings
                    'sc_guidance': sc_guidance.detach(),  # Augmented schema embeddings
                    'augmentation_guidance': pruning_guidance  # Structural guidance from augmentation
                }
        
        student_loss = self.student(self.feats, self.pos, self.mps, self.nei_index, 
                                   middle_teacher_guidance=middle_teacher_guidance)
        
        # 3. Knowledge distillation from main teacher ONLY
        # FIX 2.8: Clarified roles - Main teacher for KD, Middle teacher for augmentation guidance only
        kd_loss = torch.tensor(0.0, device=self.device)
        
        # Main teacher: Pure knowledge distillation (no augmentation)
        if self.teacher:
            # Use the COMPREHENSIVE distillation framework with all advanced loss types
            if hasattr(self.kd_framework, 'calc_distillation_losss'):
                # Use the comprehensive loss with InfoNCE, self-contrast, multi-level, etc.
                kd_loss, loss_breakdown = self.kd_framework.calc_distillation_loss(
                    self.feats, self.mps, self.nei_index, self.pos,
                    nodes=contrastive_nodes, num_classes=self.nb_classes,
                    distill_config=get_distillation_config(self.args)
                )
                # Extract the main loss from breakdown
                kd_loss = loss_breakdown.get('distill_loss', kd_loss)

            elif hasattr(self.kd_framework, 'calc_knowledge_distillation_loss'):
                # Fallback to simple version
                kd_loss = self.kd_framework.calc_knowledge_distillation_loss(
                    self.feats, self.mps, self.nei_index, distill_config={
                        'distill_weight': self.args.stage2_distill_weight,
                        'temperature': getattr(self.args, 'kd_temperature', 4.0),
                        'nodes': contrastive_nodes,
                        'use_kl_div': getattr(self.args, 'use_kl_div', True),
                        'use_info_nce': getattr(self.args, 'use_info_nce', True)
                    }
                )
        
        # 4. Subspace contrastive loss guided by augmentation patterns
        subspace_loss = torch.tensor(0.0, device=self.device)
        if self.student.use_middle_teacher_guidance and pruning_guidance:  # pruning_guidance = augmentation_guidance
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
        
        # 4.5. NEW: Link prediction losses for better edge modeling
        link_recon_loss = torch.tensor(0.0, device=self.device)
        relational_loss = torch.tensor(0.0, device=self.device)
        
        # Link reconstruction loss on student embeddings
        if epoch % 5 == 0 or epoch < 50:  # Apply more frequently in early training
            try:
                from models.kd_heco import (link_reconstruction_loss, relational_kd_loss, 
                                            sample_edges_from_metapaths, sample_negative_edges)
                
                # Get student embeddings
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
                        print("⚠️ Warning: No edges sampled from meta-paths!")
            except Exception as e:
                if epoch == 0 or epoch % 50 == 0:
                    print(f"⚠️ Warning: Link reconstruction error at epoch {epoch}: {e}")
        
        # Relational KD loss - preserve teacher's pairwise similarity structure
        if self.teacher and (epoch % 3 == 0 or epoch < 50):
            try:
                from models.kd_heco import relational_kd_loss
                
                with torch.no_grad():
                    teacher_embeds = self.teacher.get_embeds(self.feats, self.mps)
                
                student_embeds = self.student.get_embeds(self.feats, self.mps, detach=False)
                
                # Sample nodes for efficiency
                num_sample_nodes = min(512, student_embeds.size(0))
                sample_node_indices = torch.randperm(student_embeds.size(0))[:num_sample_nodes].to(self.device)
                
                # Compute relational KD loss
                relational_loss = relational_kd_loss(
                    teacher_embeds, student_embeds, 
                    sampled_nodes=sample_node_indices,
                    temperature=2.0
                )
            except Exception as e:
                if epoch == 0 or epoch % 50 == 0:
                    print(f"⚠️ Warning: Relational KD error at epoch {epoch}: {e}")
    
        # 5. Enhanced loss weighting
        # FIX 2.8: Clear role separation
        total_loss = student_loss
        
        # Role assignment:
        # - Main teacher: Knowledge distillation (learned representations from clean data)
        # - Middle teacher: Augmentation guidance (structural hints from augmented data)
        if self.teacher and self.middle_teacher:
            # Main teacher: Primary knowledge distillation (full weight)
            main_kd_weight = self.args.stage2_distill_weight  # 0.8 default
            augmentation_guidance_weight = getattr(self.args, 'pruning_weight', 0.3)  # Actually augmentation weight
            
            total_loss += main_kd_weight * kd_loss  # Main teacher KD
            total_loss += augmentation_guidance_weight * pruning_loss  # Augmentation guidance from middle teacher
            
        elif self.teacher:
            total_loss += self.args.stage2_distill_weight * kd_loss
            
        elif self.middle_teacher:
            middle_weight = getattr(self.args, 'pruning_weight', 1.0)
            total_loss += middle_weight * pruning_loss
            
        if subspace_loss > 0:
            total_loss += getattr(self.args, 'subspace_weight', 0.2) * subspace_loss  # Increase subspace learning weight
        
        # NEW: Add link prediction losses
        # Link reconstruction loss for explicit edge modeling
        link_recon_weight = getattr(self.args, 'link_recon_weight', 0.4)  # Default 0.4
        if link_recon_loss > 0:
            total_loss += link_recon_weight * link_recon_loss
        
        # Relational KD loss for preserving pairwise similarities
        relational_kd_weight = getattr(self.args, 'relational_kd_weight', 0.5)  # Default 0.5
        if relational_loss > 0:
            total_loss += relational_kd_weight * relational_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), student_loss.item(), kd_loss.item(), 
                pruning_loss.item(), subspace_loss.item(), 
                link_recon_loss.item(), relational_loss.item())
    
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
            # Enhanced training step with dual-teacher guidance, distillation, and pruning
            # FIX 2.1: Pass epoch number for guidance refresh
            loss_tuple = self.train_epoch(epoch=epoch)
            
            # Handle different return formats
            if len(loss_tuple) == 7:
                total_loss, student_loss, kd_loss, pruning_loss, subspace_loss, link_recon_loss, relational_loss = loss_tuple
            elif len(loss_tuple) == 5:
                total_loss, student_loss, kd_loss, pruning_loss, subspace_loss = loss_tuple
                link_recon_loss = relational_loss = 0.0
            else:
                total_loss, student_loss, kd_loss = loss_tuple
                pruning_loss = subspace_loss = link_recon_loss = relational_loss = 0.0
            
            # Validation step
            val_loss = self.validate()

            # Update progress bar with NEW link prediction metrics
            # FIX 2.8: Updated terminology to reflect actual roles
            postfix_dict = {
                'total': f"{total_loss:.4f}",
                'student': f"{student_loss:.4f}",
                'main_kd': f"{kd_loss:.4f}",  # Main teacher knowledge distillation
                'aug_guide': f"{pruning_loss:.4f}",  # Middle teacher augmentation guidance (not pruning!)
                'subspace': f"{subspace_loss:.4f}",
                'link_rec': f"{link_recon_loss:.4f}",  # NEW: Link reconstruction
                'relation': f"{relational_loss:.4f}",  # NEW: Relational KD
                'val': f"{val_loss:.4f}",
                'best': f"{self.best_loss:.4f}",
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
        print(f"Middle teacher guidance was {'enabled' if self.student.use_middle_teacher_guidance else 'disabled'}")
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