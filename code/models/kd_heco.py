import torch
import torch.nn as nn
import torch.nn.functional as F
from models.contrast import Contrast
from models.sc_encoder import mySc_encoder
from training.hetero_augmentations import HeteroAugmentationPipeline

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
        
        # Handle sparse vs dense adjacency matrices
        if hasattr(adj, 'is_sparse') and adj.is_sparse:
            # Sparse path
            if not adj.is_coalesced():
                adj = adj.coalesce()
            
            if adj._nnz() == 0:
                # Empty sparse matrix
                out = torch.zeros(adj.size(0), seq_fts.size(1), device=seq_fts.device, dtype=seq_fts.dtype)
            else:
                # Dimension check
                if adj.size(1) != seq_fts.size(0):
                    raise ValueError(f"Matrix dimensions incompatible: adj {adj.shape} x seq_fts {seq_fts.shape}")
                
                try:
                    out = torch.sparse.mm(adj, seq_fts)
                except RuntimeError as e:
                    print(f"Warning: Sparse mm failed ({e}), falling back to dense")
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
    """Original MyHeCo model

    Accepts optional kwargs for compatibility with training scripts that may pass
    meta-path encoder configuration (mp_encoder_type, mp_low_rank_dim, use_path_gate,
    operator_type, poly_order). These are currently ignored in this implementation.
    """
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, **kwargs):
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
    Augmentation Teacher
    
    This middle teacher:
    - Learns on AUGMENTED heterogeneous graphs
    - Provides AUGMENTATION GUIDANCE to help student learn robust representations
    """
    def __init__(self, feats_dim_list, hidden_dim, attn_drop, feat_drop, P, sample_rate, nei_num, tau, lam, 
                 augmentation_config=None):
        super(AugmentationTeacher, self).__init__()
        self.expert_dim = hidden_dim
        self.P = P
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.tau = tau
        self.lam = lam
        self.feats_dim_list = feats_dim_list
        
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
            
        self.cross_aug_learning = nn.ModuleDict({
            'structure_predictor': nn.Linear(self.expert_dim, self.expert_dim),  # Predict structural importance
            'combined_projector': nn.Linear(self.expert_dim * 2, self.expert_dim),  # Project combined to expert_dim
            'attention_allocator': nn.MultiheadAttention(self.expert_dim, num_heads, dropout=attn_drop, batch_first=True)
        })

    def forward(self, feats, pos, mps, nei_index, return_augmentation_guidance=False):
        """
        Forward pass with augmentation-aware learning and guidance generation
        """
        # Always apply augmentation
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
        
        mp_divergence_loss = -F.cosine_similarity(z_mp_orig, z_mp_aug, dim=1).mean() * 0.05
        sc_divergence_loss = -F.cosine_similarity(z_sc_orig, z_sc_aug, dim=1).mean() * 0.05
        
        # Standard contrastive losses for both
        contrast_loss_orig = self.contrast(z_mp_orig, z_sc_orig, pos)
        contrast_loss_aug = self.contrast(z_mp_aug, z_sc_aug, pos)
        
        # Generate pruning guidance based on augmentation patterns
        augmentation_guidance = None
        if return_augmentation_guidance or not self.training:
            augmentation_guidance = self._generate_augmentation_guidance(z_mp_aug, z_sc_aug, aug_info)

        # Total loss: learn from both views with diversity bonus
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
        if use_augmentation:
            aug_feats, _ = self.augmentation_pipeline(feats, mps=mps)
            z_mp = F.elu(self.fc_list[0](aug_feats[0]))
        else:
            z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach() if detach else z_mp
    
    def get_representations(self, feats, mps, nei_index, use_augmentation: bool = True):
        """Get both meta-path and schema-level representations with optional augmentation"""
        if use_augmentation:
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
                 nei_num, tau, lam, compression_ratio=0.5, use_augmentation_teacher_guidance=False):
        super(StudentMyHeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.student_dim = int(hidden_dim * compression_ratio)
        self.P = P  # Number of meta-paths
        self.nei_num = nei_num  # Number of neighbor types for schema-level encoder
        self.use_augmentation_teacher_guidance = use_augmentation_teacher_guidance

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

        # Persistent teacher->student projector layers to avoid creating layers inside forward
        # Teacher guidance embeddings typically come from hidden_dim space
        self.mp_teacher_to_student = nn.Linear(self.hidden_dim, self.student_dim)
        self.sc_teacher_to_student = nn.Linear(self.hidden_dim, self.student_dim)

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
            
        contrast_loss = self.contrast(z_mp, z_sc, pos)
        return contrast_loss

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
                    # Project teacher guidance to student dimension using persistent layer
                    teacher_guidance = self.mp_teacher_to_student(teacher_guidance)
                
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
                    # Project teacher guidance to student dimension using persistent layer
                    teacher_guidance = self.sc_teacher_to_student(teacher_guidance)
                
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


def KLDiverge(teacher_logits, student_logits, temperature):
    """KL divergence loss for soft target distillation"""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


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

class DualTeacherKD(nn.Module):
    """
    Knowledge Distillation Framework
    
    Teacher-Student knowledge distillation.
    """
    def __init__(self, teacher=None, student=None, augmentation_teacher=None):
        super(DualTeacherKD, self).__init__()
        self.teacher = teacher  # Main teacher for knowledge distillation
        self.student = student  # Student model
        self.augmentation_teacher = augmentation_teacher  # Optional augmentation expert
        
        # Initialize prediction heads for knowledge alignment
        if self.student is not None and self.teacher is not None:
            student_dim = getattr(self.student, 'student_dim', 64)
            teacher_dim = getattr(self.teacher, 'hidden_dim', 128)
            
            # Knowledge alignment head
            # CRITICAL: No LayerNorm! It normalizes outputs and destroys magnitude needed for link prediction
            # Use simple projection with nonlinearity to align student→teacher dimensions
            self.knowledge_alignment = nn.Sequential(
                nn.Linear(student_dim, teacher_dim // 2),
                nn.ReLU(),
                nn.Linear(teacher_dim // 2, teacher_dim)
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
        
        # Align dimensions using pre-initialized alignment head
        # CRITICAL: knowledge_alignment must exist (initialized in __init__)
        if not hasattr(self, 'knowledge_alignment'):
            raise RuntimeError(
                "knowledge_alignment head not initialized. "
                "Ensure both teacher and student are provided to DualTeacherKD.__init__()"
            )
        
        student_mp_aligned = self.knowledge_alignment(student_mp)
        student_sc_aligned = self.knowledge_alignment(student_sc)
        
        # Temperature for soft targets
        temperature = distill_config.get('kd_temperature', 3.0) if distill_config else 3.0
        
        # Normalization destroys the embedding scale which is crucial for dot product similarities
        # The student needs to learn both direction AND magnitude from the teacher
        mp_kd_loss = F.mse_loss(student_mp_aligned, teacher_mp) * (temperature ** 2)
        sc_kd_loss = F.mse_loss(student_sc_aligned, teacher_sc) * (temperature ** 2)
        
        return (mp_kd_loss + sc_kd_loss) * 0.5
    
    def calc_augmentation_alignment_loss(self, feats, mps, nei_index, augmentation_guidance=None):
        """
        Calculate augmentation alignment loss - guides student to learn from middle teacher's augmented views
        
        Args:
            feats: Node features
            mps: Meta-path matrices
            nei_index: Neighbor indices
            augmentation_guidance: Guidance dict from middle teacher
            
        Returns:
            Alignment loss to make student benefit from augmented teacher's knowledge
        """
        if self.augmentation_teacher is None or self.student is None:
            return torch.tensor(0.0, device=feats[0].device)
        
        # Get student representations (with guidance if enabled)
        student_mp, student_sc = self.student.get_representations(feats, mps, nei_index)
        
        # Get middle teacher representations (with augmentation)
        with torch.no_grad():
            middle_mp, middle_sc = self.augmentation_teacher.get_representations(
                feats, mps, nei_index, use_augmentation=True
            )
        
        # Align student to middle teacher's AUGMENTED view
        # Use same alignment head as main teacher for consistency
        if not hasattr(self, 'knowledge_alignment'):
            # Fallback: direct MSE without alignment
            print("Warning: knowledge_alignment head not found. Using direct MSE for alignment.")
            mp_align_loss = F.mse_loss(student_mp, middle_mp)
            sc_align_loss = F.mse_loss(student_sc, middle_sc)
        else:
            # Proper alignment through projection head
            # CRITICAL FIX: Remove normalization to preserve magnitude for link prediction
            student_mp_aligned = self.knowledge_alignment(student_mp)
            student_sc_aligned = self.knowledge_alignment(student_sc)
            
            mp_align_loss = F.mse_loss(student_mp_aligned, middle_mp)
            sc_align_loss = F.mse_loss(student_sc_aligned, middle_sc)
        
        # Combine with lower weight
        alignment_loss = (mp_align_loss + sc_align_loss) * 0.5
        
        return alignment_loss


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_compression_ratio(teacher, student):
    """Calculate the compression ratio between teacher and student"""
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    return student_params / teacher_params