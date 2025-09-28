import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrast import Contrast
from .sc_encoder import mySc_encoder
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

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()

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


class MiddleMyHeCo(nn.Module):
    """Middle teacher with compressed architecture and augmentation for hierarchical distillation"""
    def __init__(self, feats_dim_list, hidden_dim, attn_drop, feat_drop, P, sample_rate, nei_num, tau, lam, 
                 compression_ratio=0.7, augmentation_config=None):
        super(MiddleMyHeCo, self).__init__()
        # Compress hidden dimension for middle teacher
        self.compressed_dim = int(hidden_dim * compression_ratio)
        self.original_hidden_dim = hidden_dim
        self.P = P
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.tau = tau
        self.lam = lam
        self.feats_dim_list = feats_dim_list
        
        # Compressed feature projection layers
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.compressed_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        
        # Compressed encoders
        self.mp = myMp_encoder(P, self.compressed_dim, attn_drop)
        self.sc = mySc_encoder(self.compressed_dim, sample_rate, nei_num, attn_drop)
        
        # Standard contrast module
        self.contrast = Contrast(self.compressed_dim, tau, lam)
        
        # Augmentation pipeline
        if augmentation_config is None:
            augmentation_config = {
                'use_node_masking': True,
                'use_edge_augmentation': True,
                'use_autoencoder': True,
                'mask_rate': 0.1,
                'remask_rate': 0.3,
                'edge_drop_rate': 0.1,
                'autoencoder_hidden_dim': self.compressed_dim,
                'autoencoder_layers': 2,
                'reconstruction_weight': 0.1
            }
        self.augmentation_pipeline = HeteroAugmentationPipeline(feats_dim_list, augmentation_config)
        
        # Alignment layers for distillation
        self.teacher_align = nn.Linear(self.compressed_dim, hidden_dim)  # Align with teacher
        self.student_align = nn.Linear(self.compressed_dim, hidden_dim // 2)  # Align with student

    def forward(self, feats, pos, mps, nei_index, use_augmentation=True):
        # Apply augmentation during training
        total_reconstruction_loss = torch.tensor(0.0, device=feats[0].device)
        
        if self.training and use_augmentation:
            aug_feats, aug_mps, aug_info = self.augmentation_pipeline(feats, mps)
            # Add reconstruction loss if available
            if 'total_reconstruction_loss' in aug_info:
                total_reconstruction_loss = aug_info['total_reconstruction_loss'] * 0.1  # weight
        else:
            aug_feats, aug_mps = feats, mps
        
        # Process features
        h_all = []
        for i in range(len(aug_feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](aug_feats[i]))))
        
        # Get meta-path and schema-level embeddings
        z_mp = self.mp(h_all[0], aug_mps)
        z_sc = self.sc(h_all, nei_index)
        
        # Standard contrast loss
        contrast_loss = self.contrast(z_mp, z_sc, pos)
        
        # Total loss includes reconstruction loss
        total_loss = contrast_loss + total_reconstruction_loss
        return total_loss

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()
    
    def get_representations(self, feats, mps, nei_index):
        """Get both meta-path and schema-level representations"""
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        return z_mp, z_sc
    
    def get_teacher_aligned_representations(self, feats, mps, nei_index):
        """Get representations aligned with teacher dimension for stage 1 distillation"""
        z_mp, z_sc = self.get_representations(feats, mps, nei_index)
        z_mp_aligned = self.teacher_align(z_mp)
        z_sc_aligned = self.teacher_align(z_sc)
        return z_mp_aligned, z_sc_aligned
    
    def get_student_aligned_representations(self, feats, mps, nei_index):
        """Get representations aligned with student dimension for stage 2 distillation"""
        z_mp, z_sc = self.get_representations(feats, mps, nei_index)
        z_mp_aligned = self.student_align(z_mp)
        z_sc_aligned = self.student_align(z_sc)
        return z_mp_aligned, z_sc_aligned


class StudentMyHeCo(nn.Module):
    """Compressed student version of MyHeCo with progressive pruning capabilities"""
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, compression_ratio=0.5, enable_pruning=True):
        super(StudentMyHeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.student_dim = int(hidden_dim * compression_ratio)
        self.P = P  # Number of meta-paths
        self.enable_pruning = enable_pruning

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

        # Initialize attention pruning masks
        if self.enable_pruning:
            self._init_attention_masks()

    def _init_attention_masks(self):
        """Initialize attention pruning masks"""
        # Meta-path attention masks
        self.mp_att_mask_train = nn.Parameter(torch.ones(1, self.student_dim), requires_grad=True)
        self.mp_att_mask_fixed = nn.Parameter(torch.ones(1, self.student_dim), requires_grad=False)
        
        # Schema-level attention masks
        self.sc_att_mask_train = nn.Parameter(torch.ones(1, self.student_dim), requires_grad=True)
        self.sc_att_mask_fixed = nn.Parameter(torch.ones(1, self.student_dim), requires_grad=False)
        
        # Meta-path level pruning masks
        self.mp_mask_train = nn.ParameterList([
            nn.Parameter(torch.ones(1), requires_grad=True) for _ in range(self.P)
        ])
        self.mp_mask_fixed = nn.ParameterList([
            nn.Parameter(torch.ones(1), requires_grad=False) for _ in range(self.P)
        ])
        
        # Embedding dimension pruning masks
        self.emb_mask_train = nn.Parameter(torch.ones(self.student_dim), requires_grad=True)
        self.emb_mask_fixed = nn.Parameter(torch.ones(self.student_dim), requires_grad=False)

    def forward(self, feats, pos, mps, nei_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        # Apply attention masks during forward pass
        if self.enable_pruning:
            z_mp = self._forward_with_attention_masks(h_all[0], mps)
            z_sc = self._forward_sc_with_masks(h_all, nei_index)
        else:
            z_mp = self.mp(h_all[0], mps)
            z_sc = self.sc(h_all, nei_index)
            
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def _forward_with_attention_masks(self, h, mps):
        """Forward pass with attention masks for meta-path encoder"""
        if not self.enable_pruning:
            return self.mp(h, mps)
        
        # Apply embedding mask to input
        h_masked = h * self.emb_mask_train * self.emb_mask_fixed
        
        # Apply meta-path level masks
        mps_masked = []
        for i, mp in enumerate(mps):
            if i < len(self.mp_mask_train):
                mask_val = self.mp_mask_train[i] * self.mp_mask_fixed[i]
                if hasattr(mp, 'is_sparse') and mp.is_sparse:
                    mps_masked.append(mp * mask_val.item())
                else:
                    mps_masked.append(mp * mask_val)
            else:
                mps_masked.append(mp)
        
        # Get embeddings from meta-path encoder
        embeds = []
        for i in range(self.P):
            if i < len(mps_masked):
                embeds.append(self.mp.node_level[i](h_masked, mps_masked[i]))
        
        # Apply attention mask to attention mechanism
        z_mp = self._attention_with_mask(embeds, self.mp_att_mask_train * self.mp_att_mask_fixed)
        return z_mp

    def _forward_sc_with_masks(self, h_all, nei_index):
        """Forward pass with attention masks for schema-level encoder"""
        if not self.enable_pruning:
            return self.sc(h_all, nei_index)
        
        # Apply embedding mask to all features
        h_masked = []
        for h in h_all:
            h_masked.append(h * self.emb_mask_train * self.emb_mask_fixed)
        
        # Get schema-level representation with masked attention
        z_sc = self.sc(h_masked, nei_index)
        return z_sc

    def _attention_with_mask(self, embeds, att_mask):
        """Apply masked attention mechanism"""
        beta = []
        
        # Apply mask to attention weights
        masked_att = self.mp.att.att * att_mask
        attn_curr = self.mp.att.attn_drop(masked_att)
        
        for embed in embeds:
            sp = self.mp.att.tanh(self.mp.att.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.mp.att.softmax(beta)
        
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        
        return z_mp

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        if self.enable_pruning:
            z_mp = self._forward_with_attention_masks(z_mp, mps)
        else:
            z_mp = self.mp(z_mp, mps)
        return z_mp.detach()
    
    def get_representations(self, feats, mps, nei_index):
        """Get both meta-path and schema-level representations"""
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        if self.enable_pruning:
            z_mp = self._forward_with_attention_masks(h_all[0], mps)
            z_sc = self._forward_sc_with_masks(h_all, nei_index)
        else:
            z_mp = self.mp(h_all[0], mps)
            z_sc = self.sc(h_all, nei_index)
            
        return z_mp, z_sc
    
    def get_teacher_aligned_representations(self, feats, mps, nei_index):
        """Get representations projected to teacher dimension"""
        z_mp, z_sc = self.get_representations(feats, mps, nei_index)
        z_mp_aligned = self.teacher_projection(z_mp)
        z_sc_aligned = self.teacher_projection(z_sc)
        return z_mp_aligned, z_sc_aligned

    def get_masks(self):
        """Get current pruning masks for subspace contrastive learning"""
        if not self.enable_pruning:
            dummy_mask = torch.ones(self.student_dim, device=next(self.parameters()).device)
            return dummy_mask, dummy_mask
        
        # Combined embedding masks
        emb_mask = self.emb_mask_train * self.emb_mask_fixed
        return emb_mask, emb_mask

    def apply_progressive_pruning(self, pruning_ratios):
        """Apply progressive pruning based on magnitude"""
        if not self.enable_pruning:
            return

        with torch.no_grad():
            # Prune attention weights
            att_ratio = pruning_ratios.get('attention', 0.1)
            if att_ratio > 0 and att_ratio < 1.0:
                # Meta-path attention pruning
                mp_att_importance = torch.abs(self.mp_att_mask_train * self.mp_att_mask_fixed)
                if mp_att_importance.sum() > 0:
                    mp_threshold = torch.quantile(mp_att_importance.flatten(), att_ratio)
                    self.mp_att_mask_fixed.data = (mp_att_importance >= mp_threshold).float()
                
                # Schema-level attention pruning
                sc_att_importance = torch.abs(self.sc_att_mask_train * self.sc_att_mask_fixed)
                if sc_att_importance.sum() > 0:
                    sc_threshold = torch.quantile(sc_att_importance.flatten(), att_ratio)
                    self.sc_att_mask_fixed.data = (sc_att_importance >= sc_threshold).float()
            
            # Prune embedding dimensions
            emb_ratio = pruning_ratios.get('embedding', 0.1)
            if emb_ratio > 0 and emb_ratio < 1.0:  # Validate ratio range
                try:
                    combined_mask = self.emb_mask_train * self.emb_mask_fixed
                    importance = torch.abs(combined_mask)

                    # Ensure we have non-zero importance values
                    if importance.sum() > 0:
                        threshold = torch.quantile(importance, emb_ratio)
                        self.emb_mask_fixed.data = (importance >= threshold).float()
                    else:
                        print("Warning: All embedding importance values are zero, skipping pruning")
                except Exception as e:
                    print(f"Warning: Embedding pruning failed: {e}")

            # Prune meta-path connections
            mp_ratio = pruning_ratios.get('metapath', 0.05)
            if mp_ratio > 0 and mp_ratio < 1.0 and len(self.mp_mask_train) > 0:
                try:
                    for i in range(len(self.mp_mask_train)):
                        if i >= len(self.mp_mask_fixed):
                            break

                        combined_mask = self.mp_mask_train[i] * self.mp_mask_fixed[i]
                        importance = torch.abs(combined_mask)

                        # For single values, use simple thresholding
                        if importance.numel() == 1:
                            if importance.item() < mp_ratio:
                                self.mp_mask_fixed[i].data.fill_(0.0)
                        else:
                            # Handle multi-dimensional masks
                            threshold = torch.quantile(importance, mp_ratio)
                            self.mp_mask_fixed[i].data = (importance >= threshold).float()
                except Exception as e:
                    print(f"Warning: Meta-path pruning failed: {e}")

    def get_attention_weights(self):
        """Get current attention weights for analysis"""
        if not self.enable_pruning:
            return None, None
        
        mp_att_weights = self.mp_att_mask_train * self.mp_att_mask_fixed
        sc_att_weights = self.sc_att_mask_train * self.sc_att_mask_fixed
        
        return mp_att_weights.detach(), sc_att_weights.detach()

    def get_sparsity_stats(self):
        """Get current sparsity statistics"""
        if not self.enable_pruning:
            return {
                'embedding_sparsity': 1.0, 
                'metapath_sparsity': 1.0,
                'mp_attention_sparsity': 1.0,
                'sc_attention_sparsity': 1.0
            }

        # Embedding sparsity
        emb_mask = self.emb_mask_train * self.emb_mask_fixed
        emb_sparsity = (emb_mask != 0).float().mean().item()

        # Meta-path sparsity
        mp_sparsity = 0.0
        for i in range(len(self.mp_mask_train)):
            mask = self.mp_mask_train[i] * self.mp_mask_fixed[i]
            mp_sparsity += (mask != 0).float().mean().item()
        mp_sparsity /= len(self.mp_mask_train) if len(self.mp_mask_train) > 0 else 1
        
        # Attention sparsity
        mp_att_sparsity = (self.mp_att_mask_train * self.mp_att_mask_fixed != 0).float().mean().item()
        sc_att_sparsity = (self.sc_att_mask_train * self.sc_att_mask_fixed != 0).float().mean().item()

        return {
            'embedding_sparsity': emb_sparsity,
            'metapath_sparsity': mp_sparsity,
            'mp_attention_sparsity': mp_att_sparsity,
            'sc_attention_sparsity': sc_att_sparsity
        }

    def reset_trainable_masks(self):
        """Reset trainable masks to ones for next training iteration"""
        if not self.enable_pruning:
            return
            
        with torch.no_grad():
            self.mp_att_mask_train.data.fill_(1.0)
            self.sc_att_mask_train.data.fill_(1.0)
            self.emb_mask_train.data.fill_(1.0)
            for mask in self.mp_mask_train:
                mask.data.fill_(1.0)

# SimCLR
def infoNCE(anchor, positive, nodes, temperature):
    """InfoNCE loss for knowledge distillation"""
    anchor = anchor[nodes]
    positive = positive[nodes]
    
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    
    # Compute similarities
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
    
    # Compute negative similarities (all other samples)
    neg_sim = torch.matmul(anchor, positive.t()) / temperature
    
    # InfoNCE loss
    exp_pos = torch.exp(pos_sim)
    exp_neg = torch.sum(torch.exp(neg_sim), dim=-1)
    
    loss = -torch.log(exp_pos / (exp_pos + exp_neg)).mean()
    return loss


def KLDiverge(teacher_logits, student_logits, temperature):
    """KL divergence loss for soft target distillation"""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def self_contrast_loss(mp_embeds, sc_embeds, unique_nodes, temperature=1.0, weight=1.0):
    """
    Self-contrast loss adapted for heterogeneous graphs
    Enhances negative sampling by contrasting within embeddings
    """
    def point_neg_predict(embeds1, embeds2, nodes, temp):
        """Compute negative predictions for contrastive learning"""
        picked_embeds = embeds1[nodes]
        preds = picked_embeds @ embeds2.T
        return torch.exp(preds / temp).sum(-1)
    
    loss = 0
    unique_mp_nodes = unique_nodes[:len(unique_nodes)//2] if len(unique_nodes) > 1 else unique_nodes
    unique_sc_nodes = unique_nodes[len(unique_nodes)//2:] if len(unique_nodes) > 1 else unique_nodes
    
    # Meta-path vs Schema-level contrast
    loss += torch.log(point_neg_predict(mp_embeds, sc_embeds, unique_mp_nodes, temperature) + 1e-5).mean()
    loss += torch.log(point_neg_predict(sc_embeds, mp_embeds, unique_sc_nodes, temperature) + 1e-5).mean()
    
    # Self-contrast within same representation space
    loss += torch.log(point_neg_predict(mp_embeds, mp_embeds, unique_mp_nodes, temperature) + 1e-5).mean()
    loss += torch.log(point_neg_predict(sc_embeds, sc_embeds, unique_sc_nodes, temperature) + 1e-5).mean()
    
    return loss * weight


def subspace_contrastive_loss_hetero(mp_embeds, sc_embeds, mp_masks, sc_masks, 
                                   unique_nodes, temperature=1.0, weight=1.0, 
                                   pruning_run=0, use_loosening=True):
    """
    Subspace contrastive learning adapted for heterogeneous graphs
    Uses both meta-path and schema-level embeddings with mask-based similarity
    """
    if mp_masks is None or sc_masks is None:
        # Fallback to standard contrastive learning
        return torch.tensor(0.0, device=mp_embeds.device)
    
    # Loosening factors for different pruning stages
    loosen_factors = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    loosen_factor = loosen_factors[min(pruning_run, len(loosen_factors)-1)] if use_loosening else 0.0
    
    # Apply masks to embeddings
    mp_masked = mp_embeds * mp_masks if mp_masks.dim() == mp_embeds.dim() else mp_embeds
    sc_masked = sc_embeds * sc_masks if sc_masks.dim() == sc_embeds.dim() else sc_embeds
    
    # Select nodes for contrastive learning
    selected_nodes = unique_nodes[:min(512, len(unique_nodes))]  # Limit for efficiency
    mp_selected = mp_masked[selected_nodes]
    sc_selected = sc_masked[selected_nodes]
    
    # Compute similarities
    mp_sim_matrix = mp_selected @ mp_selected.T / temperature
    sc_sim_matrix = sc_selected @ sc_selected.T / temperature
    
    # Create targets based on mask similarities (if masks available)
    if hasattr(mp_masks, 'shape') and mp_masks.dim() >= 2:
        mp_mask_selected = mp_masks[selected_nodes]
        mp_mask_sim = mp_mask_selected @ mp_mask_selected.T
        mp_targets = (mp_mask_sim >= (mp_mask_sim.mean() - loosen_factor)).float()
    else:
        # Identity matrix as fallback
        mp_targets = torch.eye(len(selected_nodes), device=mp_embeds.device)
    
    if hasattr(sc_masks, 'shape') and sc_masks.dim() >= 2:
        sc_mask_selected = sc_masks[selected_nodes]
        sc_mask_sim = sc_mask_selected @ sc_mask_selected.T
        sc_targets = (sc_mask_sim >= (sc_mask_sim.mean() - loosen_factor)).float()
    else:
        sc_targets = torch.eye(len(selected_nodes), device=sc_embeds.device)
    
    # Compute contrastive losses
    mp_loss = F.cross_entropy(mp_sim_matrix, mp_targets.argmax(dim=1))
    sc_loss = F.cross_entropy(sc_sim_matrix, sc_targets.argmax(dim=1))
    
    total_loss = (mp_loss + sc_loss) * weight
    return total_loss

class MyHeCoKD(nn.Module):
    """Knowledge Distillation framework for heterogeneous graph learning with hierarchical support"""
    def __init__(self, teacher=None, student=None, middle_teacher=None):
        super(MyHeCoKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.middle_teacher = middle_teacher
        
        # Determine distillation mode
        if self.middle_teacher is not None:
            if self.teacher is not None:
                self.mode = "teacher_to_middle"  # Stage 1
            else:
                self.mode = "middle_to_student"  # Stage 2
        else:
            self.mode = "direct"  # Direct distillation
        
        print(f"KD Mode: {self.mode}")
    
    def forward(self):
        pass
    
    def get_teacher_student_pair(self):
        """Get appropriate teacher-student pair based on current mode"""
        if self.mode == "teacher_to_middle":
            return self.teacher, self.middle_teacher
        elif self.mode == "middle_to_student":
            return self.middle_teacher, self.student  
        else:  # direct
            return self.teacher, self.student
    
    def calc_distillation_loss(self, feats, mps, nei_index, pos,
                              nodes=None, distill_config=None):
        """
        Calculate knowledge distillation loss with enhanced LightGNN techniques
        
        Args:
            feats: Node features
            mps: Meta-paths
            nei_index: Neighbor indices
            pos: Positive pairs
            nodes: Nodes for contrastive learning
            distill_config: Distillation configuration dict
        """
        # Get appropriate teacher-student pair
        teacher, student = self.get_teacher_student_pair()
        
        if distill_config is None:
            distill_config = {
                'use_embedding_kd': True,
                'use_prediction_kd': True,  # Now properly implemented
                'use_heterogeneous_kd': True,
                'use_self_contrast': True,
                'use_subspace_contrast': True,
                'use_multi_level_kd': True,  # Multi-level distillation
                'embedding_temp': 4.0,
                'prediction_temp': 3.0,  # Slightly lower for prediction-level
                'self_contrast_temp': 1.0,
                'embedding_weight': 0.4,  # Reduced to balance with prediction
                'prediction_weight': 0.6,  # Higher weight for prediction-level KD
                'heterogeneous_weight': 0.3,
                'self_contrast_weight': 0.2,
                'subspace_weight': 0.3,
                'multi_level_weight': 0.4,
                'pruning_run': 0
            }
        
        # Student forward pass
        student_loss = student(feats, pos, mps, nei_index)
        
        # Get teacher representations (detached) - use single detached copy to prevent corruption
        with torch.no_grad():
            # Create single detached copy for memory efficiency
            mps_detached = []
            for mp in mps:
                if hasattr(mp, 'is_sparse') and mp.is_sparse:
                    if not mp.is_coalesced():
                        mp = mp.coalesce()
                    mps_detached.append(mp.detach())
                else:
                    mps_detached.append(mp.detach())

            # Determine which alignment strategy to use based on model types
            teacher_type = type(teacher).__name__
            student_type = type(student).__name__
            
            # For student training (Middle Teacher -> Student): Teacher aligns down to student
            if hasattr(teacher, 'get_student_aligned_representations') and student_type == 'StudentMyHeCo':
                print("Middle Teacher to Student alignment")
                teacher_mp, teacher_sc = teacher.get_student_aligned_representations(feats, mps_detached, nei_index)
                student_mp, student_sc = student.get_representations(feats, mps_detached, nei_index)
            # For middle teacher training (Teacher -> Middle Teacher): Student aligns up to teacher  
            elif hasattr(student, 'get_teacher_aligned_representations'):
                print("Teacher to Middle Teacher alignment")
                teacher_mp, teacher_sc = teacher.get_representations(feats, mps_detached, nei_index)
                student_mp, student_sc = student.get_teacher_aligned_representations(feats, mps_detached, nei_index)
            # Fallback: no alignment
            else:
                print("Fuck up some where")
                teacher_mp, teacher_sc = teacher.get_representations(feats, mps_detached, nei_index)
                student_mp, student_sc = student.get_representations(feats, mps_detached, nei_index)
        
        total_distill_loss = 0
        losses = {'main_loss': student_loss}
        
        # Embedding-level knowledge distillation
        if distill_config['use_embedding_kd'] and nodes is not None:
            mp_embed_loss = infoNCE(teacher_mp, student_mp, nodes, distill_config['embedding_temp'])
            sc_embed_loss = infoNCE(teacher_sc, student_sc, nodes, distill_config['embedding_temp'])
            embed_distill_loss = (mp_embed_loss + sc_embed_loss) / 2
            total_distill_loss += distill_config['embedding_weight'] * embed_distill_loss
            losses['embedding_distill'] = embed_distill_loss
        
        # Self-contrast loss
        if distill_config['use_self_contrast'] and nodes is not None:
            unique_nodes = torch.unique(nodes)
            self_contrast = self_contrast_loss(
                student_mp, student_sc, unique_nodes, 
                temperature=distill_config['self_contrast_temp'],
                weight=distill_config['self_contrast_weight']
            )
            total_distill_loss += self_contrast
            losses['self_contrast'] = self_contrast
        
        # Subspace contrastive loss with real masks
        if distill_config['use_subspace_contrast'] and nodes is not None:
            # Get actual masks from student model if available
            if hasattr(student, 'get_masks'):
                mp_masks, sc_masks = student.get_masks()
            else:
                # Fallback to dummy masks
                mp_masks = torch.ones_like(student_mp)
                sc_masks = torch.ones_like(student_sc)

            subspace_loss = subspace_contrastive_loss_hetero(
                student_mp, student_sc, mp_masks, sc_masks,
                torch.unique(nodes),
                temperature=distill_config.get('subspace_temp', 1.0),
                weight=distill_config['subspace_weight'],
                pruning_run=distill_config.get('pruning_run', 0),
                use_loosening=True  # Enable adaptive loosening
            )
            total_distill_loss += subspace_loss
            losses['subspace_contrast'] = subspace_loss
        
        # Multi-level distillation from teacher layers (NEW)
        if distill_config['use_multi_level_kd'] and nodes is not None and hasattr(teacher, 'get_multi_order_representations'):
            with torch.no_grad():
                # Reuse the same detached mps for memory efficiency
                teacher_multi_representations = teacher.get_multi_order_representations(feats, mps_detached, nei_index)

            # Use high-order representations (layers 2+) for distillation
            if len(teacher_multi_representations) > 2:
                # Combine high-order representations
                high_order_teacher = sum(teacher_multi_representations[2:]) / len(teacher_multi_representations[2:])

                # Student combined representation
                student_combined = (student_mp + student_sc) / 2

                # Multi-level distillation loss
                multi_level_loss = infoNCE(high_order_teacher, student_combined, nodes, distill_config['embedding_temp'])
                total_distill_loss += distill_config['multi_level_weight'] * multi_level_loss
                losses['multi_level_distill'] = multi_level_loss

        # Heterogeneous graph specific distillation
        if distill_config['use_heterogeneous_kd']:
            mp_mse_loss = F.mse_loss(student_mp, teacher_mp)
            sc_mse_loss = F.mse_loss(student_sc, teacher_sc)
            hetero_distill_loss = mp_mse_loss + sc_mse_loss
            total_distill_loss += distill_config['heterogeneous_weight'] * hetero_distill_loss
            losses['heterogeneous_distill'] = hetero_distill_loss

        # Prediction-level knowledge distillation (for downstream tasks)
        if distill_config['use_prediction_kd']:
            prediction_losses = []

            # 1. Soft Target Distillation for Node Classification
            # Generate predictions using a classifier on top of embeddings
            if hasattr(teacher, 'get_embeds') and hasattr(student, 'get_embeds'):
                with torch.no_grad():
                    teacher_embeds = teacher.get_embeds(feats, mps_detached)
                student_embeds = student.get_embeds(feats, mps_detached)

                # Create separate classifiers for teacher and student (different embedding dimensions)
                teacher_embed_dim = teacher_embeds.shape[1]
                student_embed_dim = student_embeds.shape[1]

                # Teacher classifier
                if not hasattr(self, 'teacher_prediction_classifier'):
                    self.teacher_prediction_classifier = nn.Sequential(
                        nn.Linear(teacher_embed_dim, teacher_embed_dim // 2),
                        nn.ReLU(),
                        nn.Linear(teacher_embed_dim // 2, 2)  # Adjust num_classes as needed
                    ).to(teacher_embeds.device)

                # Student classifier
                if not hasattr(self, 'student_prediction_classifier'):
                    self.student_prediction_classifier = nn.Sequential(
                        nn.Linear(student_embed_dim, student_embed_dim // 2),
                        nn.ReLU(),
                        nn.Linear(student_embed_dim // 2, 2)  # Adjust num_classes as needed
                    ).to(student_embeds.device)

                # Get teacher predictions (soft targets)
                with torch.no_grad():
                    teacher_logits = self.teacher_prediction_classifier(teacher_embeds)
                    teacher_soft_targets = F.softmax(teacher_logits / distill_config['prediction_temp'], dim=-1)

                # Get student predictions
                student_logits = self.student_prediction_classifier(student_embeds)
                student_log_probs = F.log_softmax(student_logits / distill_config['prediction_temp'], dim=-1)

                # KL Divergence loss for soft target distillation
                kl_loss = F.kl_div(student_log_probs, teacher_soft_targets, reduction='batchmean')
                kl_loss *= (distill_config['prediction_temp'] ** 2)  # Temperature scaling
                prediction_losses.append(kl_loss)

            # 2. Contrastive Prediction Distillation
            # Match prediction similarities between teacher and student
            if nodes is not None and len(nodes) > 1:
                # Sample node pairs for contrastive prediction
                num_pairs = min(512, len(nodes))  # Limit for efficiency
                sampled_nodes = nodes[:num_pairs] if len(nodes) >= num_pairs else nodes

                # Get teacher and student embeddings for sampled nodes
                with torch.no_grad():
                    teacher_sampled = teacher_embeds[sampled_nodes]
                student_sampled = student_embeds[sampled_nodes]

                # Compute pairwise prediction similarities
                teacher_pred_sim = torch.mm(teacher_sampled, teacher_sampled.t())
                student_pred_sim = torch.mm(student_sampled, student_sampled.t())

                # MSE loss on prediction similarity matrices
                pred_sim_loss = F.mse_loss(student_pred_sim, teacher_pred_sim)
                prediction_losses.append(pred_sim_loss)

            # 3. Link Prediction Knowledge Distillation
            # If pos (positive pairs) are available, do link prediction KD
            if pos is not None and hasattr(pos, 'indices'):
                try:
                    # Get edge indices from pos tensor
                    if hasattr(pos, 'coalesce'):
                        pos_coalesced = pos.coalesce()
                        edge_indices = pos_coalesced.indices()  # [2, num_edges]
                    else:
                        edge_indices = torch.nonzero(pos, as_tuple=False).t()

                    if edge_indices.shape[1] > 0:
                        # Sample edges for link prediction KD
                        num_edges = min(256, edge_indices.shape[1])
                        sampled_edge_indices = torch.randperm(edge_indices.shape[1])[:num_edges]
                        sampled_edges = edge_indices[:, sampled_edge_indices]  # [2, num_sampled]

                        # Get node embeddings
                        with torch.no_grad():
                            teacher_embeds_all = teacher.get_embeds(feats, mps_detached)
                        student_embeds_all = student.get_embeds(feats, mps_detached)

                        # Compute edge predictions (dot product)
                        teacher_src = teacher_embeds_all[sampled_edges[0]]
                        teacher_dst = teacher_embeds_all[sampled_edges[1]]
                        teacher_edge_preds = (teacher_src * teacher_dst).sum(dim=-1)

                        student_src = student_embeds_all[sampled_edges[0]]
                        student_dst = student_embeds_all[sampled_edges[1]]
                        student_edge_preds = (student_src * student_dst).sum(dim=-1)

                        # Soft target distillation for link predictions
                        teacher_edge_soft = torch.sigmoid(teacher_edge_preds / distill_config['prediction_temp'])
                        student_edge_logits = student_edge_preds / distill_config['prediction_temp']

                        # Binary cross entropy loss for link prediction KD
                        link_pred_loss = F.binary_cross_entropy_with_logits(
                            student_edge_logits, teacher_edge_soft
                        )
                        prediction_losses.append(link_pred_loss)

                except Exception as e:
                    # If link prediction KD fails, continue without it
                    print(f"Warning: Link prediction KD failed: {e}")

            # Combine all prediction losses
            if prediction_losses:
                total_pred_loss = sum(prediction_losses) / len(prediction_losses)
                total_distill_loss += distill_config['prediction_weight'] * total_pred_loss
                losses['prediction_distill'] = total_pred_loss
            else:
                losses['prediction_distill'] = torch.tensor(0.0, device=teacher_mp.device)
        
        total_loss = student_loss + total_distill_loss
        losses['total_loss'] = total_loss
        losses['distill_loss'] = total_distill_loss
        
        return total_loss, losses


def create_teacher_student_models(hidden_dim, feats_dim_list, feat_drop, attn_drop, 
                                 P, sample_rate, nei_num, tau, lam, compression_ratio=0.5):
    """
    Create teacher and student models
    
    Returns:
        teacher: Full-size teacher model
        student: Compressed student model
        kd_model: Knowledge distillation framework
    """
    teacher = MyHeCo(hidden_dim, feats_dim_list, feat_drop, attn_drop, 
                     P, sample_rate, nei_num, tau, lam)
    
    student = StudentMyHeCo(hidden_dim, feats_dim_list, feat_drop, attn_drop, 
                           P, sample_rate, nei_num, tau, lam, compression_ratio)
    
    kd_model = MyHeCoKD(teacher=teacher, student=student)
    
    return teacher, student, kd_model


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_compression_ratio(teacher, student):
    """Calculate the compression ratio between teacher and student"""
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    return student_params / teacher_params
