import torch
import torch.nn as nn
import torch.nn.functional as F
from models.contrast import Contrast
from models.sc_encoder import mySc_encoder
from training.hetero_augmentations import HeteroAugmentationPipeline
from models.kd_params import get_augmentation_config, get_distillation_config, kd_params

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)

        # Ensure seq_fts is 2D for matrix multiplication
        if seq_fts.dim() == 1:
            seq_fts = seq_fts.unsqueeze(1)
        elif seq_fts.dim() > 2:
            seq_fts = seq_fts.view(-1, seq_fts.size(-1))

        # Handle different sparse tensor formats
        if hasattr(adj, 'is_sparse') and adj.is_sparse:
            # Enhanced sparse tensor safety checks
            if not adj.is_coalesced():
                adj = adj.coalesce()

            # Validate sparse tensor integrity
            if adj._nnz() == 0:
                # Handle empty sparse tensor
                out = torch.zeros(adj.size(0), seq_fts.size(1), device=seq_fts.device, dtype=seq_fts.dtype)
            else:
                # Check dimensions before sparse multiplication
                if adj.dim() != 2:
                    raise ValueError(f"Sparse adjacency matrix must be 2D, got {adj.dim()}D with shape {adj.shape}")
                if seq_fts.dim() != 2:
                    raise ValueError(f"Feature matrix must be 2D, got {seq_fts.dim()}D with shape {seq_fts.shape}")

                # Verify matrix multiplication compatibility
                if adj.size(1) != seq_fts.size(0):
                    raise ValueError(f"Matrix dimensions incompatible: adj {adj.shape} x seq_fts {seq_fts.shape}")

                # Safe sparse matrix multiplication
                try:
                    out = torch.sparse.mm(adj, seq_fts)
                except RuntimeError as e:
                    # Fallback to dense multiplication if sparse fails
                    print(f"Warning: Sparse multiplication failed ({e}), falling back to dense")
                    out = torch.mm(adj.to_dense(), seq_fts)
        else:
            # Dense matrix handling with improved safety
            if adj.dim() == 2 and seq_fts.dim() == 2:
                # Standard case
                if adj.size(1) != seq_fts.size(0):
                    raise ValueError(f"Matrix dimensions incompatible: adj {adj.shape} x seq_fts {seq_fts.shape}")
                out = torch.mm(adj, seq_fts)
            else:
                # Handle dimension mismatches more safely
                if adj.dim() > 2:
                    adj_2d = adj.view(-1, adj.size(-1))
                else:
                    adj_2d = adj

                if seq_fts.dim() > 2:
                    seq_2d = seq_fts.view(-1, seq_fts.size(-1))
                else:
                    seq_2d = seq_fts

                # Final dimension check
                if adj_2d.size(1) != seq_2d.size(0):
                    raise ValueError(f"Matrix dimensions incompatible after reshaping: {adj_2d.shape} x {seq_2d.shape}")

                out = torch.mm(adj_2d, seq_2d)

        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp


class myMp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(myMp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)

    def forward(self, h, mps):
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
        z_mp = self.att(embeds)
        return z_mp


class MyHeCo(nn.Module):
    """Original MyHeCo model"""
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam):
        super(MyHeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.mp = myMp_encoder(P, hidden_dim, attn_drop)
        self.sc = mySc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, feats, pos, mps, nei_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, feats, mps, detach: bool = True):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach() if detach else z_mp

    def get_representations(self, feats, mps, nei_index):
        """Get both meta-path and schema-level representations"""
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        return z_mp, z_sc

    def get_multi_order_representations(self, feats, mps, nei_index):
        """Get multi-level representations for hierarchical distillation"""
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

        representations = []

        # Level 0: Raw feature embeddings
        representations.append(h_all[0])

        # Level 1: First processing layer (before encoder)
        # For simplicity, we use the features after projection as level 1
        representations.append(h_all[0])

        # Level 2: Meta-path encoder output
        z_mp = self.mp(h_all[0], mps)
        representations.append(z_mp)

        # Level 3: Schema-level encoder output
        z_sc = self.sc(h_all, nei_index)
        representations.append(z_sc)

        # Level 4: Combined representation (weighted average)
        combined = (z_mp + z_sc) / 2
        representations.append(combined)

        return representations


class AugmentationTeacher(nn.Module):
    """
    Augmentation Expert Teacher
    
    This middle teacher:
    - Learns on AUGMENTED heterogeneous graphs (masked nodes + meta-path connections)
    - Provides AUGMENTATION GUIDANCE to help student learn robust representations
    """
    def __init__(self, feats_dim_list, hidden_dim, attn_drop, feat_drop, P, sample_rate, nei_num, tau, lam, 
                 augmentation_config=None, loss_flags=None):
        super(AugmentationTeacher, self).__init__()
        self.expert_dim = hidden_dim
        self.P = P
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.tau = tau
        self.lam = lam
        self.feats_dim_list = feats_dim_list
        
        # Loss control flags
        self.loss_flags = loss_flags if loss_flags is not None else {
            'use_middle_divergence_loss': False
        }
        
        # Feature projection layers optimized for augmented data
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.expert_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        
        # Specialized encoders for augmented graph understanding
        self.mp = myMp_encoder(P, self.expert_dim, attn_drop)
        self.sc = mySc_encoder(self.expert_dim, sample_rate, nei_num, attn_drop)
        
        # Augmentation-aware contrast module
        self.contrast = Contrast(self.expert_dim, tau, lam)

        self.augmentation_config = augmentation_config
        self.augmentation_pipeline = HeteroAugmentationPipeline(feats_dim_list, augmentation_config)
        
        # Augmentation guidance networks (provides guidance based on augmented views)
        self.mp_augmentation_guide = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim // 2),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 2, P),  # Guidance for each meta-path
            nn.Sigmoid()
        )
        
        self.sc_augmentation_guide = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim // 2),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 2, nei_num),  # Guidance for each schema connection
            nn.Sigmoid()
        )
        
        # Attention importance predictor
        self.attention_importance = nn.Sequential(
            nn.Linear(self.expert_dim * 2, self.expert_dim),  # Combined mp+sc
            nn.ReLU(),
            nn.Linear(self.expert_dim, self.expert_dim // 4),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Cross-augmentation learning module
        num_heads = max(1, self.expert_dim // 16)  # Ensure divisible by num_heads
        num_heads = min(num_heads, 8)  # Cap at 8 heads
        if self.expert_dim % num_heads != 0:
            num_heads = 1  # Fall back to single head if not divisible
            
        # FIX 2.3: Removed unused 'mask_predictor' - was never used by student
        self.cross_aug_learning = nn.ModuleDict({
            'structure_predictor': nn.Linear(self.expert_dim, self.expert_dim),  # Predict structural importance
            'combined_projector': nn.Linear(self.expert_dim * 2, self.expert_dim),  # Project combined to expert_dim
            'attention_allocator': nn.MultiheadAttention(self.expert_dim, num_heads, dropout=attn_drop, batch_first=True)
        })

    def forward(self, feats, pos, mps, nei_index, return_augmentation_guidance=False):
        """
        Forward pass with augmentation-aware learning and guidance generation
        """
        # Always apply augmentation for this expert - it's trained on augmented data
        aug_feats, aug_info = self.augmentation_pipeline(feats, mps=mps)
        
        # Process original and augmented features
        h_all_orig = []
        h_all_aug = []
        for i in range(len(feats)):
            h_all_orig.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
            h_all_aug.append(F.elu(self.feat_drop(self.fc_list[i](aug_feats[i]))))
        
        # Get embeddings from both original and augmented data
        z_mp_orig = self.mp(h_all_orig[0], mps)
        z_sc_orig = self.sc(h_all_orig, nei_index)
        
        z_mp_aug = self.mp(h_all_aug[0], mps)
        z_sc_aug = self.sc(h_all_aug, nei_index)
        
        # FIXED: Removed contradictory consistency loss
        # We want the expert to learn DIFFERENT representations from augmented data
        # Not the same (that would make augmentation useless)
        # Instead, use contrastive divergence to encourage meaningful differences
        mp_divergence_loss = -F.cosine_similarity(z_mp_orig, z_mp_aug, dim=1).mean() * 0.05
        sc_divergence_loss = -F.cosine_similarity(z_sc_orig, z_sc_aug, dim=1).mean() * 0.05
        
        # Standard contrastive losses for both
        contrast_loss_orig = self.contrast(z_mp_orig, z_sc_orig, pos)
        contrast_loss_aug = self.contrast(z_mp_aug, z_sc_aug, pos)
        
        # Generate pruning guidance based on augmentation patterns
        augmentation_guidance = None
        if return_augmentation_guidance or not self.training:
            augmentation_guidance = self._generate_augmentation_guidance(z_mp_aug, z_sc_aug, aug_info)

        # Total expert loss: learn from both views with diversity bonus
        total_loss = (contrast_loss_orig + contrast_loss_aug) * 0.5 + \
                    mp_divergence_loss + sc_divergence_loss

        if return_augmentation_guidance:
            return total_loss, augmentation_guidance
        return total_loss

    def _generate_augmentation_guidance(self, z_mp, z_sc, aug_info):
        """Generate augmentation guidance based on learned representations"""
        batch_size = z_mp.size(0)
        
        # Meta-path importance guidance
        mp_guidance = self.mp_augmentation_guide(z_mp.mean(0, keepdim=True))  # [1, P]
        
        # Schema-level importance guidance  
        sc_guidance = self.sc_augmentation_guide(z_sc.mean(0, keepdim=True))  # [1, nei_num]
        
        # Combined representation for attention importance
        combined_repr = torch.cat([z_mp, z_sc], dim=1)  # [batch, 2*expert_dim]
        attention_importance = self.attention_importance(combined_repr)  # [batch, 1]
        
        # Predict structural importance based on augmentation effects
        structure_importance = self.cross_aug_learning['structure_predictor'](z_mp)
        structure_importance = F.normalize(structure_importance, p=2, dim=1)
        
        # Attention allocation guidance using multi-head attention
        # Project combined representation to match expert_dim
        combined_projected = self.cross_aug_learning['combined_projector'](combined_repr)
        z_combined_proj = combined_projected.unsqueeze(1)  # [batch, 1, expert_dim]
        
        attn_output, attn_weights = self.cross_aug_learning['attention_allocator'](
            z_combined_proj, z_combined_proj, z_combined_proj
        )
        
        # Helps student learn from augmented views, doesn't prune the model
        augmentation_guidance = {
            'mp_importance': mp_guidance.squeeze(0),  # [P] - Meta-path attention weights
            'sc_importance': sc_guidance.squeeze(0),  # [nei_num] - Schema attention weights
            'attention_importance': attention_importance,  # [batch, 1] - Node-level importance
            'structure_importance': structure_importance,  # [batch, expert_dim] - Structural features
            'attention_weights': attn_weights,  # [batch, 1, 1] - Cross-view attention
            'augmentation_info': aug_info  # Original augmentation metadata
        }
        
        return augmentation_guidance
    
    def get_embeds(self, feats, mps, detach: bool = True, use_augmentation: bool = True):
        """Get embeddings with optional augmentation"""
        if use_augmentation and self.training:
            aug_feats, _ = self.augmentation_pipeline(feats, mps=mps)
            z_mp = F.elu(self.fc_list[0](aug_feats[0]))
        else:
            z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach() if detach else z_mp
    
    def get_representations(self, feats, mps, nei_index, use_augmentation: bool = True):
        """Get both meta-path and schema-level representations with optional augmentation"""
        if use_augmentation and self.training:
            aug_feats, _ = self.augmentation_pipeline(feats, mps=mps)
            processed_feats = aug_feats
        else:
            processed_feats = feats
            
        h_all = []
        for i in range(len(processed_feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](processed_feats[i]))))
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        return z_mp, z_sc
    
    def get_augmentation_guidance(self, feats, mps, nei_index):
        """
        Get augmentation-based guidance for student model
        """
        self.eval()  # Set to eval mode for stable guidance
        with torch.no_grad():
            # Create dummy pos tensor on the same device as features
            dummy_pos = torch.zeros(1, device=feats[0].device)
            _, augmentation_guidance = self.forward(feats, dummy_pos, mps, nei_index, return_augmentation_guidance=True)
        return augmentation_guidance

class StudentMyHeCo(nn.Module):
    """Compressed student version of MyHeCo with optional augmentation teacher guidance"""
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, compression_ratio=0.5, use_augmentation_teacher_guidance=False,
                 loss_flags=None):
        super(StudentMyHeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.student_dim = int(hidden_dim * compression_ratio)
        self.P = P  # Number of meta-paths
        self.nei_num = nei_num  # Number of neighbor types for schema-level encoder
        self.use_augmentation_teacher_guidance = use_augmentation_teacher_guidance
        
        # Loss control flags
        self.loss_flags = loss_flags if loss_flags is not None else {
            'use_student_contrast_loss': True,
            'use_guidance_alignment_loss': True,
            'use_gate_entropy_loss': True
        }

        # Compressed feature projection layers
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.student_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        # Compressed encoders
        self.mp = myMp_encoder(P, self.student_dim, attn_drop)
        self.sc = mySc_encoder(self.student_dim, sample_rate, nei_num, attn_drop)
        self.contrast = Contrast(self.student_dim, tau, lam)

        # Projection layer to match teacher dimension for distillation
        self.teacher_projection = nn.Linear(self.student_dim, hidden_dim)

        # If using augmentation teacher guidance, initialize guidance integration layers
        if self.use_augmentation_teacher_guidance:
            self._init_guidance_integration()

    def _init_guidance_integration(self):
        """Initialize augmentation teacher guidance integration layers"""
        # Meta-path guidance integration
        self.mp_guidance_gate = nn.Sequential(
            nn.Linear(self.student_dim * 2, self.student_dim),
            nn.Sigmoid()
        )
        
        # Schema guidance integration
        self.sc_guidance_gate = nn.Sequential(
            nn.Linear(self.student_dim * 2, self.student_dim),
            nn.Sigmoid()
        )
        
        # Guidance fusion weights (learnable balance between student and middle teacher)
        self.mp_fusion_weight = nn.Parameter(torch.tensor(0.3))  # Start with 30% middle teacher influence
        self.sc_fusion_weight = nn.Parameter(torch.tensor(0.3))  # Start with 30% middle teacher influence

    def forward(self, feats, pos, mps, nei_index, augmentation_teacher_guidance=None):
        """
        Forward pass with optional augmentation teacher guidance
        
        Two modes:
        1. No guidance: Standard student forward pass
        2. Augmentation teacher guidance: Use augmentation teacher's guidance for better learning
        """
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        # Mode 1: Use augmentation teacher guidance if available and enabled
        if self.use_augmentation_teacher_guidance and augmentation_teacher_guidance is not None:
            z_mp = self._forward_with_guidance(h_all[0], mps, augmentation_teacher_guidance, 'mp')
            z_sc = self._forward_with_guidance(h_all, nei_index, augmentation_teacher_guidance, 'sc')
        else:
            # Mode 2: Standard student forward pass
            z_mp = self.mp(h_all[0], mps)
            z_sc = self.sc(h_all, nei_index)
            
        # Base contrastive loss (CONTROLLED BY FLAG)
        total_loss = torch.tensor(0.0, device=z_mp.device)
        
        if self.loss_flags.get('use_student_contrast_loss', True):
            contrast_loss = self.contrast(z_mp, z_sc, pos)
            total_loss += contrast_loss
        
        # Add guidance alignment loss if augmentation teacher guidance is used (CONTROLLED BY FLAG)
        if self.use_augmentation_teacher_guidance and augmentation_teacher_guidance is not None and self.training:
            if self.loss_flags.get('use_guidance_alignment_loss', False):
                guidance_loss = self._compute_guidance_alignment_loss(z_mp, z_sc, augmentation_teacher_guidance)
                guidance_weight = self.loss_flags.get('guidance_alignment_weight', 0.2)
                total_loss += guidance_loss * guidance_weight
            
            # Add entropy regularization to prevent guidance gate saturation (CONTROLLED BY FLAG)
            if self.loss_flags.get('use_gate_entropy_loss', False):
                gate_entropy_loss = self._compute_gate_entropy_regularization()
                gate_weight = self.loss_flags.get('gate_entropy_weight', 0.05)
                total_loss += gate_entropy_loss * gate_weight
            
        return total_loss

    def _forward_with_guidance(self, h_input, adj_input, augmentation_teacher_guidance, module_type):
        """
        Forward pass with augmentation teacher guidance integration
        
        Args:
            h_input: Input features (h for mp, h_all for sc)  
            adj_input: Adjacency input (mps for mp, nei_index for sc)
            augmentation_teacher_guidance: Guidance from augmentation teacher
            module_type: 'mp' for meta-path, 'sc' for schema
        """
        if module_type == 'mp':
            # Standard student forward pass
            student_output = self.mp(h_input, adj_input)
            
            # Get augmentation teacher guidance for meta-path
            if 'mp_guidance' in augmentation_teacher_guidance:
                teacher_guidance = augmentation_teacher_guidance['mp_guidance']
                
                # Ensure dimensions match
                if teacher_guidance.size(-1) != self.student_dim:
                    # Project teacher guidance to student dimension
                    guidance_proj = nn.Linear(teacher_guidance.size(-1), self.student_dim, device=teacher_guidance.device)
                    teacher_guidance = guidance_proj(teacher_guidance)
                
                # Fuse student output with teacher guidance
                fused_features = torch.cat([student_output, teacher_guidance], dim=-1)
                guidance_gate = self.mp_guidance_gate(fused_features)
                
                # Weighted fusion
                fusion_weight = torch.sigmoid(self.mp_fusion_weight)
                result = (1 - fusion_weight) * student_output + fusion_weight * teacher_guidance * guidance_gate
            else:
                result = student_output
                
        elif module_type == 'sc':
            # Standard student forward pass
            student_output = self.sc(h_input, adj_input)
            
            # Get augmentation teacher guidance for schema
            if 'sc_guidance' in augmentation_teacher_guidance:
                teacher_guidance = augmentation_teacher_guidance['sc_guidance']
                
                # Ensure dimensions match
                if teacher_guidance.size(-1) != self.student_dim:
                    # Project teacher guidance to student dimension
                    guidance_proj = nn.Linear(teacher_guidance.size(-1), self.student_dim, device=teacher_guidance.device)
                    teacher_guidance = guidance_proj(teacher_guidance)
                
                # Fuse student output with teacher guidance
                fused_features = torch.cat([student_output, teacher_guidance], dim=-1)
                guidance_gate = self.sc_guidance_gate(fused_features)
                
                # Weighted fusion
                fusion_weight = torch.sigmoid(self.sc_fusion_weight)
                result = (1 - fusion_weight) * student_output + fusion_weight * teacher_guidance * guidance_gate
            else:
                result = student_output
        else:
            raise ValueError(f"Unknown module_type: {module_type}")
            
        return result
    
    def _compute_guidance_alignment_loss(self, z_mp, z_sc, augmentation_teacher_guidance):
        """
        Simple alignment loss to learn from augmentation teacher guidance
        """
        total_loss = torch.tensor(0.0, device=z_mp.device)
        
        # Align meta-path representations
        if 'mp_guidance' in augmentation_teacher_guidance:
            mp_target = augmentation_teacher_guidance['mp_guidance']
            if mp_target.size(-1) == z_mp.size(-1) and mp_target.size(0) == z_mp.size(0):
                mp_loss = F.mse_loss(z_mp, mp_target.detach())
                total_loss += mp_loss * 0.5
        
        # Align schema representations
        if 'sc_guidance' in augmentation_teacher_guidance:
            sc_target = augmentation_teacher_guidance['sc_guidance']
            if sc_target.size(-1) == z_sc.size(-1) and sc_target.size(0) == z_sc.size(0):
                sc_loss = F.mse_loss(z_sc, sc_target.detach())
                total_loss += sc_loss * 0.5
        
        return total_loss
    
    def _compute_gate_entropy_regularization(self):
        """
        Prevent guidance gate saturation by encouraging entropy
        Gates should stay in middle range (0.3-0.7), not saturate at 0 or 1
        """
        if not hasattr(self, 'mp_fusion_weight') or not hasattr(self, 'sc_fusion_weight'):
            return torch.tensor(0.0)
        
        # Get fusion weights (should be in 0.3-0.7 range for healthy learning)
        mp_weight = torch.sigmoid(self.mp_fusion_weight)
        sc_weight = torch.sigmoid(self.sc_fusion_weight)
        
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        # Maximum at p=0.5, minimum at p=0 or p=1
        def binary_entropy(p):
            p = torch.clamp(p, 1e-7, 1-1e-7)  # Prevent log(0)
            return -(p * torch.log(p) + (1-p) * torch.log(1-p))
        
        mp_entropy = binary_entropy(mp_weight)
        sc_entropy = binary_entropy(sc_weight)
        
        # Loss is negative entropy (we want to maximize entropy = prevent saturation)
        entropy_loss = -(mp_entropy + sc_entropy) / 2
        
        return entropy_loss

    def get_embeds(self, feats, mps, detach: bool = True, augmentation_teacher_guidance=None):
        """Get embeddings with optional augmentation teacher guidance"""
        z_mp = F.elu(self.fc_list[0](feats[0]))
        
        if self.use_augmentation_teacher_guidance and augmentation_teacher_guidance is not None:
            z_mp = self._forward_with_guidance(z_mp, mps, augmentation_teacher_guidance, 'mp')
        else:
            z_mp = self.mp(z_mp, mps)
        
        return z_mp.detach() if detach else z_mp
    
    def get_representations(self, feats, mps, nei_index, augmentation_teacher_guidance=None):
        """Get both meta-path and schema-level representations"""
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        if self.use_augmentation_teacher_guidance and augmentation_teacher_guidance is not None:
            z_mp = self._forward_with_guidance(h_all[0], mps, augmentation_teacher_guidance, 'mp')
            z_sc = self._forward_with_guidance(h_all, nei_index, augmentation_teacher_guidance, 'sc')
        else:
            z_mp = self.mp(h_all[0], mps)
            z_sc = self.sc(h_all, nei_index)
            
        return z_mp, z_sc
    
    def get_teacher_aligned_representations(self, feats, mps, nei_index, augmentation_teacher_guidance=None):
        """Get representations projected to teacher dimension for knowledge distillation"""
        z_mp, z_sc = self.get_representations(feats, mps, nei_index, augmentation_teacher_guidance)
        z_mp_aligned = self.teacher_projection(z_mp)
        z_sc_aligned = self.teacher_projection(z_sc)
        return z_mp_aligned, z_sc_aligned

    def get_guidance_fusion_weights(self):
        """Get current fusion weights for analysis"""
        if not self.use_augmentation_teacher_guidance:
            return None, None
        
        return torch.sigmoid(self.mp_fusion_weight).item(), torch.sigmoid(self.sc_fusion_weight).item()

# SimCLR
def infoNCE(embeds1, embeds2, nodes, temperature):
    """
    InfoNCE (Noise Contrastive Estimation)
    
    OPTIMIZED: Normalize once and reuse, use logsumexp for numerical stability
    """
    # Normalize embeddings to unit sphere (do once, reuse for all operations)
    embeds1_norm = F.normalize(embeds1 + 1e-8, p=2, dim=-1)
    embeds2_norm = F.normalize(embeds2 + 1e-8, p=2, dim=-1)
    
    # Pick embeddings for selected nodes
    pckEmbeds1 = embeds1_norm[nodes]  # [batch_size, embed_dim]
    pckEmbeds2 = embeds2_norm[nodes]  # [batch_size, embed_dim]
    
    # Positive pairs: same nodes in different embedding spaces
    pos_sim = torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temperature  # [batch_size]
    
    # Negative pairs: each node in embeds1 vs all nodes in embeds2
    # Compute all similarities at once
    all_sim = (pckEmbeds1 @ embeds2_norm.T) / temperature  # [batch_size, num_nodes]
    
    # Use logsumexp for numerical stability: -log(exp(pos) / sum(exp(all))) = logsumexp(all) - pos
    loss = torch.logsumexp(all_sim, dim=-1) - pos_sim
    
    return loss.mean()


def KLDiverge(teacher_logits, student_logits, temperature):
    """KL divergence loss for soft target distillation"""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def self_contrast_loss(mp_embeds, sc_embeds, unique_nodes, temperature=1.0, weight=1.0):
    """
    Self-contrast loss adapted for heterogeneous graphs
    Enhances negative sampling by contrasting within embeddings
    
    OPTIMIZED: Pre-compute similarity matrices and use logsumexp for numerical stability
    """
    # Split nodes once
    mid = len(unique_nodes) // 2 if len(unique_nodes) > 1 else len(unique_nodes)
    unique_mp_nodes = unique_nodes[:mid]
    unique_sc_nodes = unique_nodes[mid:] if mid < len(unique_nodes) else unique_nodes
    
    # Pre-compute picked embeddings (avoid redundant indexing)
    mp_picked = mp_embeds[unique_mp_nodes]  # [B1, D]
    sc_picked = sc_embeds[unique_sc_nodes]  # [B2, D]
    
    # Compute similarity matrices once (reuse for loss computation)
    # Using matrix multiplication instead of separate function calls
    mp_sc_sim = (mp_picked @ sc_embeds.T) / temperature  # [B1, N]
    sc_mp_sim = (sc_picked @ mp_embeds.T) / temperature  # [B2, N]
    mp_mp_sim = (mp_picked @ mp_embeds.T) / temperature  # [B1, N]
    sc_sc_sim = (sc_picked @ sc_embeds.T) / temperature  # [B2, N]
    
    # Vectorized loss computation using logsumexp (numerically stable)
    # torch.logsumexp(x) = log(sum(exp(x))) but more stable
    loss = torch.logsumexp(mp_sc_sim, dim=-1).mean()
    loss += torch.logsumexp(sc_mp_sim, dim=-1).mean()
    loss += torch.logsumexp(mp_mp_sim, dim=-1).mean()
    loss += torch.logsumexp(sc_sc_sim, dim=-1).mean()
    
    return loss * weight


def subspace_contrastive_loss_hetero(mp_embeds, sc_embeds, mp_masks, sc_masks, 
                                   unique_nodes, temperature=1.0, weight=1.0, 
                                   augmentation_run=0, use_loosening=True):
    """
    Subspace contrastive learning adapted for heterogeneous graphs
    Uses both meta-path and schema-level embeddings with mask-based similarity
    Tighten constraints as model shrinks (reversed logic)
    
    OPTIMIZED: Pre-compute loosen_factor and reduce conditional checks
    """
    if mp_masks is None or sc_masks is None:
        # Fallback to standard contrastive learning
        return torch.tensor(0.0, device=mp_embeds.device)
    
    # Tightening factors for different augmentation stages (pre-computed lookup table)
    # As model gets smaller (higher augmentation_run), we TIGHTEN constraints (negative loosening)
    # Smaller models need stricter guidance, not more relaxed targets
    tighten_factors = [0.0, -0.02, -0.05, -0.08, -0.12, -0.16, -0.2, -0.25, -0.3, -0.35, -0.4]
    loosen_factor = tighten_factors[min(augmentation_run, len(tighten_factors)-1)] if use_loosening else 0.0
    
    # Select nodes for contrastive learning (do this first to reduce tensor operations)
    num_selected = min(512, len(unique_nodes))
    selected_nodes = unique_nodes[:num_selected]
    
    # Apply masks and select in one operation (avoid intermediate full-size tensors)
    mp_masked_selected = (mp_embeds * mp_masks)[selected_nodes] if mp_masks.dim() == mp_embeds.dim() else mp_embeds[selected_nodes]
    sc_masked_selected = (sc_embeds * sc_masks)[selected_nodes] if sc_masks.dim() == sc_embeds.dim() else sc_embeds[selected_nodes]
    
    # Compute similarities (temperature division is fused with matmul)
    temp_inv = 1.0 / temperature
    mp_sim_matrix = (mp_masked_selected @ mp_masked_selected.T) * temp_inv
    sc_sim_matrix = (sc_masked_selected @ sc_masked_selected.T) * temp_inv
    
    # Create targets based on mask similarities (if masks available)
    if hasattr(mp_masks, 'shape') and mp_masks.dim() >= 2:
        mp_mask_selected = mp_masks[selected_nodes]
        mp_mask_sim = mp_mask_selected @ mp_mask_selected.T
        threshold = mp_mask_sim.mean() - loosen_factor
        mp_targets = (mp_mask_sim >= threshold).float()
    else:
        # Identity matrix as fallback (pre-allocate on correct device)
        mp_targets = torch.eye(num_selected, device=mp_embeds.device, dtype=mp_embeds.dtype)
    
    if hasattr(sc_masks, 'shape') and sc_masks.dim() >= 2:
        sc_mask_selected = sc_masks[selected_nodes]
        sc_mask_sim = sc_mask_selected @ sc_mask_selected.T
        threshold = sc_mask_sim.mean() - loosen_factor
        sc_targets = (sc_mask_sim >= threshold).float()
    else:
        # Identity matrix as fallback (pre-allocate on correct device)
        sc_targets = torch.eye(num_selected, device=sc_embeds.device, dtype=sc_embeds.dtype)
    
    # Compute contrastive losses (use argmax on targets)
    mp_loss = F.cross_entropy(mp_sim_matrix, mp_targets.argmax(dim=1))
    sc_loss = F.cross_entropy(sc_sim_matrix, sc_targets.argmax(dim=1))
    
    # Combined loss (single multiplication instead of two operations)
    total_loss = (mp_loss + sc_loss) * weight
    return total_loss


def link_reconstruction_loss(embeddings, pos_edges, neg_edges, temperature=1.0):
    """
    Link reconstruction loss for better edge modeling in link prediction
    
    Args:
        embeddings: Node embeddings [num_nodes, embed_dim]
        pos_edges: Positive edge indices [num_pos_edges, 2]
        neg_edges: Negative edge indices [num_neg_edges, 2]
        temperature: Temperature for scaling scores
        
    Returns:
        Link reconstruction loss
    """
    if pos_edges is None or len(pos_edges) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Normalize embeddings for better similarity computation
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Positive edges - should have high similarity
    src_pos, dst_pos = pos_edges[:, 0], pos_edges[:, 1]
    pos_scores = (embeddings[src_pos] * embeddings[dst_pos]).sum(dim=-1) / temperature
    pos_loss = -F.logsigmoid(pos_scores).mean()
    
    # Negative edges - should have low similarity
    if neg_edges is not None and len(neg_edges) > 0:
        src_neg, dst_neg = neg_edges[:, 0], neg_edges[:, 1]
        neg_scores = (embeddings[src_neg] * embeddings[dst_neg]).sum(dim=-1) / temperature
        neg_loss = -F.logsigmoid(-neg_scores).mean()
    else:
        neg_loss = torch.tensor(0.0, device=embeddings.device)
    
    return pos_loss + neg_loss


def relational_kd_loss(teacher_embeds, student_embeds, sampled_nodes=None, temperature=2.0):
    """
    Relational Knowledge Distillation - Preserves pairwise similarity structure
    
    This loss ensures the student learns the same node-node relationships as the teacher,
    which is crucial for link prediction tasks.
    
    OPTIMIZED: Fuse normalization with similarity computation, pre-compute temperature scaling
    
    Args:
        teacher_embeds: Teacher embeddings [num_nodes, teacher_dim]
        student_embeds: Student embeddings [num_nodes, student_dim]
        sampled_nodes: Nodes to sample for efficiency (optional)
        temperature: Temperature for softening distributions
        
    Returns:
        Relational KD loss (KL divergence on similarity distributions)
    """
    # Sample nodes for computational efficiency
    if sampled_nodes is None:
        num_nodes = min(teacher_embeds.size(0), 512)  # Limit to 512 for efficiency
        sampled_nodes = torch.randperm(teacher_embeds.size(0), device=teacher_embeds.device)[:num_nodes]
    
    if len(sampled_nodes) < 2:
        return torch.tensor(0.0, device=teacher_embeds.device)
    
    # Extract and normalize sampled embeddings in one step
    teacher_samp_norm = F.normalize(teacher_embeds[sampled_nodes], p=2, dim=-1)
    student_samp_norm = F.normalize(student_embeds[sampled_nodes], p=2, dim=-1)
    
    # Pre-compute temperature scaling factor (inverse for efficiency)
    temp_inv = 1.0 / temperature
    
    # Compute similarity matrices with fused temperature scaling
    teacher_sim = torch.mm(teacher_samp_norm, teacher_samp_norm.t()) * temp_inv
    student_sim = torch.mm(student_samp_norm, student_samp_norm.t()) * temp_inv
    
    # Convert to probability distributions and compute KL divergence
    # Use F.kl_div with log_target=False (teacher_dist not in log space)
    teacher_dist = F.softmax(teacher_sim, dim=-1)
    student_log_dist = F.log_softmax(student_sim, dim=-1)
    
    # KL divergence loss: KL(teacher || student)
    relational_loss = F.kl_div(student_log_dist, teacher_dist, reduction='batchmean')
    
    return relational_loss


def sample_edges_from_metapaths(mps, num_samples=1000):
    """
    Sample positive edges from meta-path adjacency matrices
    
    Args:
        mps: List of meta-path adjacency matrices (sparse)
        num_samples: Number of edges to sample per meta-path
        
    Returns:
        pos_edges: Sampled positive edges [num_edges, 2]
    """
    all_edges = []
    
    for mp in mps:
        if mp is None:
            continue
            
        # Convert sparse tensor to COO format
        if hasattr(mp, 'coalesce'):
            mp = mp.coalesce()
            indices = mp.indices().t()  # [num_edges, 2]
        else:
            # Dense tensor
            indices = torch.nonzero(mp, as_tuple=False)
        
        if len(indices) == 0:
            continue
        
        # Sample edges
        num_to_sample = min(num_samples, len(indices))
        sampled_indices = torch.randperm(len(indices))[:num_to_sample]
        sampled_edges = indices[sampled_indices]
        all_edges.append(sampled_edges)
    
    if len(all_edges) == 0:
        return None
    
    return torch.cat(all_edges, dim=0)


def sample_negative_edges(num_nodes, num_samples, existing_edges=None):
    """
    Sample negative edges (non-existing edges)
    
    Args:
        num_nodes: Total number of nodes
        num_samples: Number of negative edges to sample
        existing_edges: Set of existing edges to avoid (optional)
        
    Returns:
        neg_edges: Negative edges [num_samples, 2]
    """
    neg_edges = []
    
    # Create set of existing edges for quick lookup
    if existing_edges is not None:
        existing_set = set(map(tuple, existing_edges.cpu().numpy()))
    else:
        existing_set = set()
    
    # Sample with replacement (some may be duplicates, but that's ok for negative sampling)
    while len(neg_edges) < num_samples:
        src = torch.randint(0, num_nodes, (num_samples - len(neg_edges),))
        dst = torch.randint(0, num_nodes, (num_samples - len(neg_edges),))
        
        for s, d in zip(src, dst):
            if s != d and (s.item(), d.item()) not in existing_set:
                neg_edges.append([s.item(), d.item()])
                if len(neg_edges) >= num_samples:
                    break
    
    return torch.tensor(neg_edges, dtype=torch.long)


def multi_hop_link_prediction_loss(embeddings, mps, num_samples=1000, max_hops=3, temperature=1.0):
    """
    Multi-hop link prediction loss
    
    Captures relationships at different distances:
    - 1-hop: Direct connections
    - 2-hop: Second-degree connections  
    - 3-hop: Third-degree connections
    
    This helps the student model understand both local and global graph structure,
    significantly improving link prediction performance.
    """
    total_loss = 0.0
    num_valid_hops = 0
    
    for hop in range(1, min(max_hops + 1, 4)):  # Limit to 3 hops for efficiency
        hop_edges = []
        
        for mp in mps:
            if not isinstance(mp, torch.Tensor):
                continue
            
            try:
                # Get k-hop adjacency
                mp_dense = mp.to_dense() if mp.is_sparse else mp
                mp_k_hop = mp_dense.clone()
                
                # Compute A^k (k-hop adjacency)
                for _ in range(hop - 1):
                    mp_k_hop = torch.mm(mp_k_hop, mp_dense)
                    # Binarize to avoid overflow
                    mp_k_hop = (mp_k_hop > 0).float()
                
                # Sample edges from k-hop adjacency
                if mp_k_hop.sum() > 0:
                    nonzero = mp_k_hop.nonzero(as_tuple=False)
                    if len(nonzero) > 0:
                        num_sample = min(num_samples // (len(mps) * max_hops), len(nonzero))
                        if num_sample > 0:
                            indices = torch.randperm(len(nonzero))[:num_sample]
                            hop_edges.append(nonzero[indices])
            except:
                continue
        
        if hop_edges:
            hop_edges = torch.cat(hop_edges, dim=0)[:num_samples]
            
            # Sample negative edges
            neg_edges = sample_negative_edges(embeddings.size(0), len(hop_edges), hop_edges)
            neg_edges = neg_edges.to(embeddings.device)
            
            # Compute loss for this hop level (weight by inverse distance)
            hop_loss = link_reconstruction_loss(embeddings, hop_edges, neg_edges, temperature)
            hop_weight = 1.0 / hop  # Closer hops weighted more
            total_loss += hop_weight * hop_loss
            num_valid_hops += 1
    
    return total_loss / max(num_valid_hops, 1)


def metapath_specific_link_loss(embeddings, mps, num_samples_per_path=500, temperature=1.0):
    """
    Meta-path specific link prediction
    
    Different meta-paths capture different semantic relationships: (examples on ACM)
    - PAP (Paper-Author-Paper): Co-authorship patterns
    - PSP (Paper-Subject-Paper): Topical similarity
    
    Learning path-specific patterns improves link prediction accuracy.
    """
    total_loss = 0.0
    num_valid_paths = 0
    
    for mp in mps:
        if not isinstance(mp, torch.Tensor):
            continue
        
        try:
            # Sample edges from this specific meta-path
            pos_edges = sample_edges_from_metapaths([mp], num_samples_per_path)
            
            if pos_edges is None or len(pos_edges) == 0:
                continue
            
            pos_edges = pos_edges.to(embeddings.device)
            
            # Sample negative edges
            neg_edges = sample_negative_edges(embeddings.size(0), len(pos_edges), pos_edges)
            neg_edges = neg_edges.to(embeddings.device)
            
            # Path-specific loss
            path_loss = link_reconstruction_loss(embeddings, pos_edges, neg_edges, temperature)
            total_loss += path_loss
            num_valid_paths += 1
        except:
            continue
    
    return total_loss / max(num_valid_paths, 1)


def structural_distance_preservation_loss(teacher_embeds, student_embeds, sampled_nodes=None, temperature=1.5):
    """
    Structural distance preservation
    
    Preserves not just similarity, but also dissimilarity structure.
    Nodes that are far apart in teacher space should also be far in student space.
    
    This helps maintain the global topology of the embedding space.
    """
    if sampled_nodes is None:
        num_nodes = min(teacher_embeds.size(0), 256)
        sampled_nodes = torch.randperm(teacher_embeds.size(0))[:num_nodes]
    
    if len(sampled_nodes) < 2:
        return torch.tensor(0.0, device=teacher_embeds.device)
    
    # Sample embeddings
    teacher_samp = teacher_embeds[sampled_nodes]
    student_samp = student_embeds[sampled_nodes]
    
    # Normalize
    teacher_samp = F.normalize(teacher_samp, p=2, dim=-1)
    student_samp = F.normalize(student_samp, p=2, dim=-1)
    
    # Compute distance matrices (1 - cosine similarity = distance)
    teacher_dist = 1.0 - torch.mm(teacher_samp, teacher_samp.t())
    student_dist = 1.0 - torch.mm(student_samp, student_samp.t())
    
    # MSE on distance matrices
    dist_loss = F.mse_loss(student_dist / temperature, teacher_dist / temperature)
    
    return dist_loss


def attention_transfer_loss(teacher_embeds, student_embeds, power=2):
    """
    Attention Transfer
    
    Transfer attention maps from teacher to student.
    Attention maps highlight important features/relationships.
    
    Based on "Paying More Attention to Attention" (ICLR 2017)
    """
    # Compute attention maps (normalized L2 norm across feature dimension)
    def attention_map(x, p=2):
        return F.normalize(x.pow(p).mean(1).view(x.size(0), -1), p=2, dim=1)
    
    teacher_att = attention_map(teacher_embeds, power)
    student_att = attention_map(student_embeds, power)
    
    # MSE on attention maps
    return F.mse_loss(student_att, teacher_att)

class DualTeacherKD(nn.Module):
    """
    Dual-Teacher Knowledge Distillation Framework
    
    Two specialized teachers work together:
    1. Teacher: Provides knowledge distillation to student
    2. Augmentation Teacher: Provides augmentation guidance based on augmented graph learning
    """
    def __init__(self, teacher=None, student=None, augmentation_teacher=None):
        super(DualTeacherKD, self).__init__()
        self.teacher = teacher  # Main teacher for knowledge distillation
        self.student = student  # Student model
        self.augmentation_teacher = augmentation_teacher
        
        # Initialize prediction heads for knowledge alignment
        if self.student is not None and self.teacher is not None:
            student_dim = getattr(self.student, 'student_dim', 64)
            teacher_dim = getattr(self.teacher, 'hidden_dim', 128)
            
            # Knowledge alignment head
            self.knowledge_alignment = nn.Sequential(
                nn.Linear(student_dim, teacher_dim // 2),
                nn.ReLU(),
                nn.Linear(teacher_dim // 2, teacher_dim),
                nn.LayerNorm(teacher_dim)
            )
            
        if self.student is not None and self.augmentation_expert is not None:
            student_dim = getattr(self.student, 'student_dim', 64)
            expert_dim = getattr(self.augmentation_expert, 'hidden_dim', 128)
            
            # Augmentation guidance alignment head
            self.augmentation_alignment = nn.Sequential(
                nn.Linear(student_dim, expert_dim // 2),
                nn.ReLU(),
                nn.Linear(expert_dim // 2, expert_dim),
                nn.LayerNorm(expert_dim)
            )
    
    def forward(self):
        pass
    
    def calc_knowledge_distillation_loss(self, feats, mps, nei_index, distill_config=None):
        """Calculate knowledge distillation loss between teacher and student"""
        if self.teacher is None or self.student is None:
            return torch.tensor(0.0, device=feats[0].device)
        
        # Get representations from both models
        with torch.no_grad():
            teacher_mp, teacher_sc = self.teacher.get_representations(feats, mps, nei_index)
            
        student_mp, student_sc = self.student.get_representations(feats, mps, nei_index)
        
        # Align dimensions if necessary
        if hasattr(self, 'knowledge_alignment'):
            student_mp_aligned = self.knowledge_alignment(student_mp)
            student_sc_aligned = self.knowledge_alignment(student_sc)
        else:
            # Simple projection if dimensions don't match
            if student_mp.size(-1) != teacher_mp.size(-1):
                student_mp_aligned = F.linear(student_mp, 
                    torch.randn(teacher_mp.size(-1), student_mp.size(-1), device=student_mp.device))
                student_sc_aligned = F.linear(student_sc,
                    torch.randn(teacher_sc.size(-1), student_sc.size(-1), device=student_sc.device))
            else:
                student_mp_aligned = student_mp
                student_sc_aligned = student_sc
        
        # Temperature for soft targets
        temperature = distill_config.get('temperature', 3.0) if distill_config else 3.0
        
        # KL divergence loss for soft targets
        mp_kd_loss = KLDiverge(teacher_mp, student_mp_aligned, temperature)
        sc_kd_loss = KLDiverge(teacher_sc, student_sc_aligned, temperature)
        
        return (mp_kd_loss + sc_kd_loss) * 0.5

    def calc_augmentation_alignment_loss(self, feats, mps, nei_index, augmentation_guidance):
        """Calculate alignment loss between student and augmentation expert guidance"""
        if self.augmentation_expert is None or self.student is None:
            return torch.tensor(0.0, device=feats[0].device)
        
        # Get student representations
        student_mp, student_sc = self.student.get_representations(feats, mps, nei_index)
        
        # Get expert representations (without augmentation for alignment)
        expert_mp, expert_sc = self.augmentation_expert.get_representations(feats, mps, nei_index, use_augmentation=False)
        
        # Align dimensions
        if hasattr(self, 'augmentation_alignment'):
            student_mp_aligned = self.augmentation_alignment(student_mp)
            student_sc_aligned = self.augmentation_alignment(student_sc)
        else:
            student_mp_aligned = student_mp
            student_sc_aligned = student_sc
        
        # Structure consistency loss
        mp_consistency = F.mse_loss(F.normalize(student_mp_aligned, p=2, dim=1), 
                                   F.normalize(expert_mp, p=2, dim=1))
        sc_consistency = F.mse_loss(F.normalize(student_sc_aligned, p=2, dim=1), 
                                   F.normalize(expert_sc, p=2, dim=1))
        
        # augmentation guidance alignment (computed in student model)
        total_loss = (mp_consistency + sc_consistency) * 0.5
        
        return total_loss
    
    def _detect_teacher_conflict(self, feats, mps, nei_index, augmentation_guidance):
        """
        Detect when main teacher and augmentation teacher give conflicting guidance
        Returns conflict penalty in [0, 1] where 1 = high conflict
        """
        if self.teacher is None or self.augmentation_expert is None:
            return 0.0
        
        with torch.no_grad():
            # Get representations from both teachers
            teacher_mp, teacher_sc = self.teacher.get_representations(feats, mps, nei_index)
            expert_mp, expert_sc = self.augmentation_expert.get_representations(feats, mps, nei_index, use_augmentation=False)
            
            # Normalize for fair comparison
            teacher_mp_norm = F.normalize(teacher_mp, p=2, dim=1)
            teacher_sc_norm = F.normalize(teacher_sc, p=2, dim=1)
            expert_mp_norm = F.normalize(expert_mp, p=2, dim=1)
            expert_sc_norm = F.normalize(expert_sc, p=2, dim=1)
            
            # Compute cosine similarity between teachers (high = agreement, low = conflict)
            mp_similarity = F.cosine_similarity(teacher_mp_norm, expert_mp_norm, dim=1).mean()
            sc_similarity = F.cosine_similarity(teacher_sc_norm, expert_sc_norm, dim=1).mean()
            
            # Average similarity
            avg_similarity = (mp_similarity + sc_similarity) / 2
            
            # Convert similarity to conflict penalty: 
            # similarity 1.0 → conflict 0.0 (perfect agreement)
            # similarity 0.0 → conflict 0.5 (orthogonal = some conflict)
            # similarity -1.0 → conflict 1.0 (opposite = maximum conflict)
            conflict_penalty = (1.0 - avg_similarity) / 2.0
            
            return conflict_penalty.item()


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_compression_ratio(teacher, student):
    """Calculate the compression ratio between teacher and student"""
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    return student_params / teacher_params