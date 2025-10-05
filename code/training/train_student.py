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
        
        # Load pre-trained middle teacher (augmentation trained on augmented graphs)
        print("Loading pre-trained augmentation teacher...")
        
        # Prepare loss flags for middle teacher
        middle_teacher_loss_flags = {
            'use_middle_divergence_loss': self.args.use_middle_divergence_loss,
            'middle_divergence_weight': self.args.middle_divergence_weight
        }
        
        self.middle_teacher = AugmentationTeacher(  # Name kept for compatibility
            feats_dim_list=self.feats_dim_list,
            hidden_dim=self.args.hidden_dim,
            attn_drop=self.args.attn_drop,
            feat_drop=self.args.feat_drop,
            P=self.P,
            sample_rate=self.args.sample_rate,
            nei_num=self.args.nei_num,
            tau=self.args.tau,
            lam=self.args.lam,
            augmentation_config=self.augmentation_config,
            loss_flags=middle_teacher_loss_flags
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
        
        # Prepare loss flags for student
        student_loss_flags = {
            'use_student_contrast_loss': self.args.use_student_contrast_loss,
            'use_guidance_alignment_loss': self.args.use_guidance_alignment_loss,
            'use_gate_entropy_loss': self.args.use_gate_entropy_loss,
            'guidance_alignment_weight': self.args.guidance_alignment_weight,
            'gate_entropy_weight': self.args.gate_entropy_weight
        }
        
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
            use_augmentation_teacher_guidance=use_augmentation_teacher_guidance,
            loss_flags=student_loss_flags
        ).to(self.device)

        self.kd_framework = DualTeacherKD(
            teacher=self.teacher,           # Main teacher for knowledge distillation
            student=self.student,           # Student to be trained
            augmentation_teacher=self.middle_teacher
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
            print("🚀 Dual-Teacher Mode: Main teacher (knowledge distillation) + Augmentation (robustness)")
        elif self.teacher:
            print("📚 Knowledge Distillation Mode: Main teacher only")
        elif self.middle_teacher:
            print("🔄 Augmentation Guidance Mode: Augmentation only")
        else:
            print("🎯 Self-Training Mode: No teacher guidance")
        
    def get_contrastive_nodes(self, batch_size=4096):
        """Get random nodes for contrastive learning"""
        total_nodes = self.feats[0].size(0)
        if batch_size >= total_nodes:
            return torch.arange(total_nodes, device=self.device)
        else:
            return torch.randperm(total_nodes, device=self.device)[:batch_size]
    
    def train_epoch(self, epoch=0):
        """Train for one epoch using enhanced dual-teacher framework with proper distillation"""
        self.student.train()
        
        # Set teachers to eval mode
        if self.teacher:
            self.teacher.eval()
        if self.middle_teacher:
            self.middle_teacher.eval()
            
        self.optimizer.zero_grad()
        
        # Get contrastive nodes for this batch
        contrastive_nodes = self.get_contrastive_nodes(batch_size=self.args.batch_size)
        
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
        
        # 3. Knowledge distillation from main teacher ONLY
        # Clarified roles - Main teacher for KD, Middle teacher for augmentation guidance only
        kd_loss = torch.tensor(0.0, device=self.device)
        
        # Main teacher: Pure knowledge distillation - CONTROLLED BY FLAG
        if self.teacher and self.args.use_kd_loss:
            if hasattr(self.kd_framework, 'calc_knowledge_distillation_loss'):
                kd_loss = self.kd_framework.calc_knowledge_distillation_loss(
                    self.feats, self.mps, self.nei_index, distill_config=get_distillation_config(self.args),
                )
            else:
                print("No distillation method found in KD framework.")
        
        # 4. Subspace contrastive loss guided by augmentation patterns - CONTROLLED BY FLAG
        subspace_loss = torch.tensor(0.0, device=self.device)
        if self.args.use_subspace_loss and self.student.use_augmentation_teacher_guidance and augmentation_guidance:
            from models.kd_heco import subspace_contrastive_loss_hetero
            
            # Get student representations
            student_mp, student_sc = self.student.get_representations(
                self.feats, self.mps, self.nei_index, augmentation_teacher_guidance
            )
            
            # Extract masks from augmentation guidance if available
            mp_masks = augmentation_guidance.get('mp_importance', None)
            sc_masks = augmentation_guidance.get('sc_importance', None)
            
            if mp_masks is not None and sc_masks is not None:
                subspace_loss = subspace_contrastive_loss_hetero(
                    student_mp, student_sc, mp_masks, sc_masks,
                    contrastive_nodes, temperature=1.0, weight=0.1
                )
        
        # 4.5. Link prediction losses for better edge modeling - CONTROLLED BY FLAGS
        link_recon_loss = torch.tensor(0.0, device=self.device)
        relational_loss = torch.tensor(0.0, device=self.device)
        
        # Get student embeddings once for all advanced losses (avoid redundant computation)
        student_embeds = None
        teacher_embeds = None
        
        # Link reconstruction loss on student embeddings - CONTROLLED BY FLAG
        if self.args.use_link_recon_loss and (epoch % 5 == 0 or epoch < 50):
            try:
                from models.kd_heco import (link_reconstruction_loss, relational_kd_loss, 
                                            sample_edges_from_metapaths, sample_negative_edges)

                # Get student embeddings (reuse if already computed)
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
                        print("⚠️ Warning: No edges sampled from meta-paths!")
            except Exception as e:
                if epoch == 0 or epoch % 50 == 0:
                    print(f"⚠️ Warning: Link reconstruction error at epoch {epoch}: {e}")
        
        # Relational KD loss - preserve teacher's pairwise similarity structure - CONTROLLED BY FLAG
        if self.args.use_relational_kd_loss and self.teacher and (epoch % 3 == 0 or epoch < 50):
            try:
                from models.kd_heco import relational_kd_loss
                
                # Get teacher embeddings (reuse if already computed)
                if teacher_embeds is None:
                    with torch.no_grad():
                        teacher_embeds = self.teacher.get_embeds(self.feats, self.mps)
                
                # Get student embeddings (reuse if already computed)
                if student_embeds is None:
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
    
        total_loss = 0
        total_loss += student_loss
        
        # Role assignment:
        # - Main teacher: Knowledge distillation (learned representations from clean data)
        # - Middle teacher: Augmentation guidance (structural hints from augmented data)
        if self.teacher and self.middle_teacher:
            # Main teacher: Primary knowledge distillation (full weight)
            main_kd_weight = self.args.stage2_distill_weight  # 0.8 default
            augmentation_guidance_weight = self.args.augmentation_weight  # Augmentation guidance weight
            
            total_loss += main_kd_weight * kd_loss  # Main teacher KD
            total_loss += augmentation_guidance_weight * augmentation_alignment_loss  # Augmentation guidance from middle teacher

        elif self.teacher:
            total_loss += self.args.stage2_distill_weight * kd_loss
            
        elif self.middle_teacher:
            total_loss += self.args.augmentation_weight * augmentation_alignment_loss

        if subspace_loss > 0:
            total_loss += self.args.subspace_weight * subspace_loss
        
        # Link reconstruction loss for explicit edge modeling
        if link_recon_loss > 0:
            total_loss += self.args.link_recon_weight * link_recon_loss
        
        # Relational KD loss for preserving pairwise similarities
        if relational_loss > 0:
            total_loss += self.args.relational_kd_weight * relational_loss
        
        # Apply periodically to avoid overhead
        advanced_losses = {}
        
        # Multi-hop link prediction (every 3 epochs)
        if self.args.use_multihop_link_loss and (epoch % 3 == 0 or epoch < 30):
            try:
                from models.kd_heco import multi_hop_link_prediction_loss
                # Ensure student_embeds is computed
                if student_embeds is None:
                    student_embeds = self.student.get_embeds(self.feats, self.mps, detach=False)
                multihop_loss = multi_hop_link_prediction_loss(
                    student_embeds, self.mps, 
                    num_samples=1000, 
                    max_hops=self.args.max_hops,
                    temperature=1.0
                )
                total_loss += self.args.multihop_weight * multihop_loss
                advanced_losses['multihop'] = multihop_loss.item()
            except Exception as e:
                if epoch == 0:
                    print(f"⚠️ Multi-hop loss disabled: {e}")
                advanced_losses['multihop'] = 0.0
        else:
            advanced_losses['multihop'] = 0.0
        
        # Meta-path specific link loss (every 4 epochs)
        if self.args.use_metapath_specific_loss and (epoch % 4 == 0 or epoch < 30):
            try:
                from models.kd_heco import metapath_specific_link_loss
                # Ensure student_embeds is computed
                if student_embeds is None:
                    student_embeds = self.student.get_embeds(self.feats, self.mps, detach=False)
                metapath_loss = metapath_specific_link_loss(
                    student_embeds, self.mps,
                    num_samples_per_path=500,
                    temperature=1.0
                )
                total_loss += self.args.metapath_specific_weight * metapath_loss
                advanced_losses['metapath'] = metapath_loss.item()
            except Exception as e:
                if epoch == 0:
                    print(f"⚠️ Meta-path specific loss disabled: {e}")
                advanced_losses['metapath'] = 0.0
        else:
            advanced_losses['metapath'] = 0.0
        
        # Structural distance preservation (every 2 epochs)
        if self.args.use_structural_distance and (epoch % 2 == 0):
            try:
                from models.kd_heco import structural_distance_preservation_loss
                # Ensure embeddings are computed
                if teacher_embeds is None and self.teacher:
                    with torch.no_grad():
                        teacher_embeds = self.teacher.get_embeds(self.feats, self.mps)
                if student_embeds is None:
                    student_embeds = self.student.get_embeds(self.feats, self.mps, detach=False)
                struct_dist_loss = structural_distance_preservation_loss(
                    teacher_embeds, student_embeds,
                    sampled_nodes=None,
                    temperature=1.5
                )
                total_loss += self.args.structural_distance_weight * struct_dist_loss
                advanced_losses['struct_dist'] = struct_dist_loss.item()
            except Exception as e:
                if epoch == 0:
                    print(f"⚠️ Structural distance loss disabled: {e}")
                advanced_losses['struct_dist'] = 0.0
        else:
            advanced_losses['struct_dist'] = 0.0
        
        # Attention transfer (every 5 epochs)
        if self.args.use_attention_transfer and (epoch % 5 == 0 or epoch < 30):
            try:
                from models.kd_heco import attention_transfer_loss
                # Ensure embeddings are computed
                if teacher_embeds is None and self.teacher:
                    with torch.no_grad():
                        teacher_embeds = self.teacher.get_embeds(self.feats, self.mps)
                if student_embeds is None:
                    student_embeds = self.student.get_embeds(self.feats, self.mps, detach=False)
                att_transfer_loss = attention_transfer_loss(
                    teacher_embeds, student_embeds,
                    power=2
                )
                total_loss += self.args.attention_transfer_weight * att_transfer_loss
                advanced_losses['attention'] = att_transfer_loss.item()
            except Exception as e:
                if epoch == 0:
                    print(f"⚠️ Attention transfer loss disabled: {e}")
                advanced_losses['attention'] = 0.0
        else:
            advanced_losses['attention'] = 0.0
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), student_loss.item(), kd_loss.item(), 
                augmentation_alignment_loss.item(), subspace_loss.item(), 
                link_recon_loss.item(), relational_loss.item(), advanced_losses)
    
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
            
            total_loss, student_loss, kd_loss, augmentation_alignment_loss, subspace_loss, link_recon_loss, relational_loss, advanced_losses = loss_tuple
            
            # Validation step
            val_loss = self.validate()
            
            # Track best model
            is_best = False

            # Update progress bar with NEW link prediction metrics
            # Updated terminology to reflect actual roles
            postfix_dict = {
                'total': f"{total_loss:.4f}",
                'student': f"{student_loss:.4f}",
                'main_kd': f"{kd_loss:.4f}",  # Main teacher knowledge distillation
                'augmentation_align': f"{augmentation_alignment_loss:.4f}",  # Augmentation alignment
                'subspace': f"{subspace_loss:.4f}",
                'link_rec': f"{link_recon_loss:.4f}",  # Link reconstruction
                'relation': f"{relational_loss:.4f}",  # Relational KD
                'val': f"{val_loss:.4f}",
                'best': f"{self.best_loss:.4f}",
                'patience': f"{self.patience_counter}/{self.args.patience}"
            }
            
            # Add advanced losses to progress bar when active
            if advanced_losses:
                if advanced_losses.get('multihop', 0) > 0:
                    postfix_dict['multihop'] = f"{advanced_losses['multihop']:.4f}"
                if advanced_losses.get('metapath', 0) > 0:
                    postfix_dict['metapath'] = f"{advanced_losses['metapath']:.4f}"
                if advanced_losses.get('struct_dist', 0) > 0:
                    postfix_dict['struct'] = f"{advanced_losses['struct_dist']:.4f}"
                if advanced_losses.get('attention', 0) > 0:
                    postfix_dict['attn'] = f"{advanced_losses['attention']:.4f}"
            
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
    
    # Create trainer and start training
    trainer = StudentTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()