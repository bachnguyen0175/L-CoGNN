import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.contrast import Contrast
from models.sc_encoder import mySc_encoder
from training.hetero_augmentations import HeteroAugmentationPipeline
from models.kd_params import *

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
                'use_autoencoder': True,
                'mask_rate': 0.1,
                'remask_rate': 0.2,
                'edge_drop_rate': 0.05,
                'num_remasking': 2,
                'autoencoder_hidden_dim': self.original_hidden_dim // 2,  # Half of main hidden dim
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
            # Augmentation pipeline only works on features now (no edge dropping)
            aug_feats, aug_info = self.augmentation_pipeline(feats)
            aug_mps = mps  # Use original meta-paths since no edge augmentation
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
        self.nei_num = nei_num  # Number of neighbor types for schema-level encoder
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
            self._init_attention_pruning()

    def _init_attention_pruning(self):
        """Initialize attention-focused pruning parameters"""
        # Meta-path attention pruning parameters
        self.mp_attention_mask = nn.Parameter(torch.ones(1, self.student_dim), requires_grad=True)
        self.mp_path_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1), requires_grad=True) for _ in range(self.P)
        ])
        
        # Schema attention pruning parameters  
        self.sc_attention_mask = nn.Parameter(torch.ones(1, self.student_dim), requires_grad=True)
        self.sc_type_weights = nn.Parameter(torch.ones(self.nei_num), requires_grad=True)
        
        # Attention pruning thresholds
        self.mp_attention_threshold = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.sc_attention_threshold = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        
        # Temperature for soft attention pruning
        self.attention_temperature = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        
        # Sparsity control with structure preservation
        self.register_buffer('attention_sparsity_weight', torch.tensor(0.005))
        self.register_buffer('target_attention_sparsity', torch.tensor(0.6))  # Target 60% attention sparsity
        self.register_buffer('structure_preservation_weight', torch.tensor(0.1))  # Preserve important structure

    def forward(self, feats, pos, mps, nei_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        # Apply attention pruning during forward pass
        if self.enable_pruning:
            z_mp = self._forward_with_metapath_attention_pruning(h_all[0], mps)
            z_sc = self._forward_with_schema_attention_pruning(h_all, nei_index)
        else:
            z_mp = self.mp(h_all[0], mps)
            z_sc = self.sc(h_all, nei_index)
            
        # Base contrastive loss
        contrast_loss = self.contrast(z_mp, z_sc, pos)
        
        # Add attention pruning loss
        if self.enable_pruning and self.training:
            attention_loss = self.get_attention_pruning_loss()
            total_loss = contrast_loss + attention_loss
        else:
            total_loss = contrast_loss
            
        return total_loss

    def _forward_with_metapath_attention_pruning(self, h, mps):
        """Forward pass with meta-path attention pruning"""
        if not self.enable_pruning:
            return self.mp(h, mps)
        
        # Get meta-path attention masks
        mp_att_mask, mp_path_masks = self.get_metapath_attention_masks()
        
        # Apply path-level pruning - weight each meta-path
        pruned_embeds = []
        for i in range(self.P):
            if i < len(mp_path_masks):
                # Apply path weight to the meta-path
                path_weight = mp_path_masks[i]
                embed = self.mp.node_level[i](h, mps[i])
                pruned_embeds.append(embed * path_weight)
            else:
                pruned_embeds.append(self.mp.node_level[i](h, mps[i]))
        
        # Apply attention-level pruning to the attention mechanism
        z_mp = self._pruned_metapath_attention(pruned_embeds, mp_att_mask)
        return z_mp

    def _forward_with_schema_attention_pruning(self, h_all, nei_index):
        """Forward pass with schema attention pruning"""
        if not self.enable_pruning:
            return self.sc(h_all, nei_index)
        
        # Get schema attention masks
        sc_att_mask, sc_type_masks = self.get_schema_attention_masks()
        
        # Apply intra-attention (within each node type) with pruning
        pruned_embeds = []
        for i in range(self.sc.nei_num):
            # Sample neighbors as usual
            sele_nei = []
            sample_num = self.sc.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).to(h_all[0].device)
            
            # Apply type-level pruning weight
            type_weight = sc_type_masks[i] if i < len(sc_type_masks) else 1.0
            one_type_emb = F.elu(self.sc.intra[i](sele_nei, h_all[i + 1], h_all[0]))
            pruned_embeds.append(one_type_emb * type_weight)
        
        # Apply inter-attention (between node types) with attention pruning
        z_sc = self._pruned_schema_attention(pruned_embeds, sc_att_mask)
        return z_sc

    def _pruned_metapath_attention(self, embeds, att_mask):
        """Meta-path attention mechanism with attention pruning"""
        beta = []
        
        # Apply attention mask to the attention parameter
        masked_att = self.mp.att.att * att_mask
        attn_curr = self.mp.att.attn_drop(masked_att)
        
        # Compute attention weights for each meta-path embedding
        for embed in embeds:
            sp = self.mp.att.tanh(self.mp.att.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.mp.att.softmax(beta)
        
        # Weighted combination of embeddings
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        
        return z_mp

    def _pruned_schema_attention(self, embeds, att_mask):
        """Schema-level attention mechanism with attention pruning"""
        beta = []
        
        # Apply attention mask to the inter-attention parameter
        masked_att = self.sc.inter.att * att_mask
        attn_curr = self.sc.inter.attn_drop(masked_att)
        
        # Compute attention weights for each node type embedding
        for embed in embeds:
            sp = self.sc.inter.tanh(self.sc.inter.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.sc.inter.softmax(beta)
        
        # Weighted combination of embeddings
        z_sc = 0
        for i in range(len(embeds)):
            z_sc += embeds[i] * beta[i]
        
        return z_sc

    def get_embeds(self, feats, mps, detach: bool = True):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        if self.enable_pruning:
            z_mp = self._forward_with_metapath_attention_pruning(z_mp, mps)
        else:
            z_mp = self.mp(z_mp, mps)
        return z_mp.detach() if detach else z_mp
    
    def get_representations(self, feats, mps, nei_index):
        """Get both meta-path and schema-level representations"""
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        if self.enable_pruning:
            z_mp = self._forward_with_metapath_attention_pruning(h_all[0], mps)
            z_sc = self._forward_with_schema_attention_pruning(h_all, nei_index)
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
        """Get current attention masks for subspace contrastive learning"""
        if not self.enable_pruning:
            dummy_mask = torch.ones(self.student_dim, device=next(self.parameters()).device)
            return dummy_mask, dummy_mask
        
        # Get attention masks - use flattened attention masks as embedding masks
        mp_att_mask, _ = self.get_metapath_attention_masks()
        sc_att_mask, _ = self.get_schema_attention_masks()
        
        # Expand to match embedding dimension if needed
        if mp_att_mask.size(-1) != self.student_dim:
            mp_att_mask = mp_att_mask.expand(-1, self.student_dim)
        if sc_att_mask.size(-1) != self.student_dim:
            sc_att_mask = sc_att_mask.expand(-1, self.student_dim)
            
        return mp_att_mask.squeeze(0), sc_att_mask.squeeze(0)

    def get_metapath_attention_masks(self):
        """Generate meta-path attention pruning masks"""
        temp = torch.clamp(self.attention_temperature, min=0.5, max=10.0)
        
        # Meta-path attention mask (for attention mechanism itself)
        mp_att_mask = torch.sigmoid((self.mp_attention_mask - self.mp_attention_threshold) / temp)
        
        # Individual meta-path weights
        mp_path_masks = []
        for i in range(len(self.mp_path_weights)):
            path_mask = torch.sigmoid((self.mp_path_weights[i] - 0.5) / temp)
            mp_path_masks.append(path_mask)
        
        return mp_att_mask, mp_path_masks

    def get_schema_attention_masks(self):
        """Generate schema attention pruning masks"""
        temp = torch.clamp(self.attention_temperature, min=0.5, max=10.0)
        
        # Schema attention mask (for inter-attention mechanism)
        sc_att_mask = torch.sigmoid((self.sc_attention_mask - self.sc_attention_threshold) / temp)
        
        # Individual node type weights
        sc_type_masks = torch.sigmoid((self.sc_type_weights - 0.5) / temp)
        
        return sc_att_mask, sc_type_masks

    def get_attention_pruning_loss(self):
        """Calculate attention-focused pruning loss"""
        if not self.enable_pruning:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Get attention masks
        mp_att_mask, mp_path_masks = self.get_metapath_attention_masks()
        sc_att_mask, sc_type_masks = self.get_schema_attention_masks()
        
        # Sparsity regularization - encourage attention pruning
        attention_sparsity = 0.0
        
        # Meta-path attention sparsity
        attention_sparsity += torch.mean(mp_att_mask)
        for path_mask in mp_path_masks:
            attention_sparsity += torch.mean(path_mask)
        
        # Schema attention sparsity
        attention_sparsity += torch.mean(sc_att_mask)
        attention_sparsity += torch.mean(sc_type_masks)
        
        # Normalize by number of components
        num_components = 2 + len(mp_path_masks) + len(sc_type_masks)
        avg_attention_sparsity = attention_sparsity / num_components
        
        # Target sparsity loss
        target_loss = torch.abs(avg_attention_sparsity - self.target_attention_sparsity)
        
        # Structure preservation: prevent pruning of highly important attention components
        structure_loss = 0.0
        
        # Preserve top-k% most important meta-path weights
        top_k_ratio = 0.3  # Keep top 30% meta-paths
        if len(mp_path_masks) > 1:
            mp_values = torch.stack([mask.squeeze() for mask in mp_path_masks])
            top_k = max(1, int(len(mp_path_masks) * top_k_ratio))
            top_indices = torch.topk(mp_values, top_k, largest=True)[1]
            
            # Penalty for pruning top important paths
            for idx in top_indices:
                structure_loss += torch.max(torch.tensor(0.5, device=mp_values.device) - mp_path_masks[idx], 
                                          torch.tensor(0.0, device=mp_values.device))
        
        # Preserve schema type diversity
        if len(sc_type_masks) > 1:
            # Encourage at least 50% of types to remain active
            min_active_types = max(1, len(sc_type_masks) // 2)
            active_types = int((sc_type_masks > 0.5).sum().item())
            if active_types < min_active_types:
                structure_loss += (min_active_types - active_types) * 0.1
        
        total_attention_loss = (attention_sparsity + target_loss) * self.attention_sparsity_weight + \
                              structure_loss * self.structure_preservation_weight
        return total_attention_loss

    def apply_progressive_attention_pruning(self, pruning_ratios):
        """Apply progressive attention pruning with advanced scheduling"""
        if not self.enable_pruning:
            return
        
        # Update target attention sparsity based on training progress
        max_sparsity = pruning_ratios.get('max_attention_sparsity', 0.6)
        min_sparsity = pruning_ratios.get('min_attention_sparsity', 0.2)
        
        # Gradually increase target sparsity during training
        current_epoch = pruning_ratios.get('current_epoch', 0)
        max_epochs = pruning_ratios.get('max_epochs', 100)
        
        # Use cosine annealing for smoother pruning progression
        progress = min(current_epoch / max_epochs, 1.0)
        cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))  # Smooth S-curve
        new_target = min_sparsity + (max_sparsity - min_sparsity) * cosine_progress
        
        self.target_attention_sparsity.data.fill_(new_target)
        
        # Dynamically adjust sparsity weight based on training phase
        base_weight = pruning_ratios.get('attention_sparsity_weight', 0.005)
        
        # Reduce sparsity weight as training progresses to avoid over-pruning
        if progress > 0.8:  # In final 20% of pruning
            adjusted_weight = base_weight * 0.5  # Reduce weight for stability
        elif progress > 0.6:  # In middle-late phase
            adjusted_weight = base_weight * 0.8
        else:  # Early-middle phase
            adjusted_weight = base_weight
            
        self.attention_sparsity_weight.data.fill_(adjusted_weight)
        
        # Optional: Adjust temperature for harder pruning as training progresses
        if 'adjust_temperature' in pruning_ratios and pruning_ratios['adjust_temperature']:
            # Start with soft pruning (high temp), move to hard pruning (low temp)
            initial_temp = 2.0
            final_temp = 0.5
            new_temp = initial_temp - (initial_temp - final_temp) * progress
            self.attention_temperature.data.fill_(max(new_temp, 0.1))  # Minimum temperature

    def get_attention_weights(self):
        """Get current attention weights for analysis"""
        if not self.enable_pruning:
            return None, None, None, None
        
        mp_att_mask, mp_path_masks = self.get_metapath_attention_masks()
        sc_att_mask, sc_type_masks = self.get_schema_attention_masks()
        return mp_att_mask.detach(), mp_path_masks, sc_att_mask.detach(), sc_type_masks.detach()

    def get_pruning_schedule_info(self, current_epoch, total_epochs=200):
        """Get recommended pruning schedule information"""
        schedule_info = {
            'current_phase': '',
            'should_prune': False,
            'pruning_frequency': 0,
            'recommended_params': {}
        }
        
        # Determine current training phase
        if current_epoch < 20:
            schedule_info['current_phase'] = 'Warm-up (No Pruning)'
            schedule_info['should_prune'] = False
            schedule_info['recommendation'] = 'Let model stabilize, focus on knowledge distillation'
            
        elif current_epoch < 50:
            schedule_info['current_phase'] = 'Early Pruning'
            schedule_info['should_prune'] = (current_epoch % 5 == 0)
            schedule_info['pruning_frequency'] = 5
            schedule_info['recommended_params'] = {
                'min_attention_sparsity': 0.1,
                'max_attention_sparsity': 0.6,
                'attention_sparsity_weight': 0.001,
                'adjust_temperature': True
            }
            schedule_info['recommendation'] = 'Gentle pruning start, every 5 epochs'
            
        elif current_epoch < 100:
            schedule_info['current_phase'] = 'Aggressive Pruning'
            schedule_info['should_prune'] = (current_epoch % 3 == 0)
            schedule_info['pruning_frequency'] = 3
            schedule_info['recommended_params'] = {
                'min_attention_sparsity': 0.1,
                'max_attention_sparsity': 0.7,
                'attention_sparsity_weight': 0.005,
                'adjust_temperature': True
            }
            schedule_info['recommendation'] = 'Main pruning phase, every 3 epochs'
            
        else:
            schedule_info['current_phase'] = 'Fine-tuning'
            schedule_info['should_prune'] = (current_epoch % 10 == 0)
            schedule_info['pruning_frequency'] = 10
            schedule_info['recommended_params'] = {
                'min_attention_sparsity': 0.1,
                'max_attention_sparsity': 0.8,
                'attention_sparsity_weight': 0.002,
                'adjust_temperature': False
            }
            schedule_info['recommendation'] = 'Stabilize pruned model, every 10 epochs'
            
        return schedule_info

    def get_sparsity_stats(self):
        """Get current attention pruning statistics"""
        if not self.enable_pruning:
            return {
                'metapath_attention_sparsity': 1.0, 
                'schema_attention_sparsity': 1.0,
                'metapath_path_sparsity': 1.0,
                'schema_type_sparsity': 1.0,
                'attention_stats': {}
            }

        # Get attention masks
        mp_att_mask, mp_path_masks = self.get_metapath_attention_masks()
        sc_att_mask, sc_type_masks = self.get_schema_attention_masks()
        
        # Calculate attention sparsity (values close to 1 = kept, close to 0 = pruned)
        mp_att_sparsity = torch.mean(mp_att_mask).item()
        sc_att_sparsity = torch.mean(sc_att_mask).item()
        
        # Meta-path path-level sparsity
        mp_path_sparsity = 0.0
        for path_mask in mp_path_masks:
            mp_path_sparsity += torch.mean(path_mask).item()
        mp_path_sparsity /= len(mp_path_masks) if len(mp_path_masks) > 0 else 1
        
        # Schema type-level sparsity
        sc_type_sparsity = torch.mean(sc_type_masks).item()
        
        # Attention statistics
        attention_stats = {
            'mp_attention_mean': torch.mean(self.mp_attention_mask).item(),
            'sc_attention_mean': torch.mean(self.sc_attention_mask).item(),
            'mp_attention_threshold': self.mp_attention_threshold.item(),
            'sc_attention_threshold': self.sc_attention_threshold.item(),
            'attention_temperature': self.attention_temperature.item(),
            'target_attention_sparsity': self.target_attention_sparsity.item()
        }

        return {
            'metapath_attention_sparsity': mp_att_sparsity,
            'schema_attention_sparsity': sc_att_sparsity,
            'metapath_path_sparsity': mp_path_sparsity,
            'schema_type_sparsity': sc_type_sparsity,
            'attention_stats': attention_stats
        }

    def reset_attention_parameters(self):
        """Reset attention pruning parameters for fresh training (optional)"""
        if not self.enable_pruning:
            return
            
        with torch.no_grad():
            # Reset attention masks to encourage learning from scratch
            self.mp_attention_mask.data.fill_(1.0)
            self.sc_attention_mask.data.fill_(1.0)
            
            # Reset path and type weights
            for path_weight in self.mp_path_weights:
                path_weight.data.fill_(1.0)
            self.sc_type_weights.data.fill_(1.0)
            
            # Reset attention thresholds
            self.mp_attention_threshold.data.fill_(0.3)
            self.sc_attention_threshold.data.fill_(0.3)
            
            # Reset temperature
            self.attention_temperature.data.fill_(2.0)

# SimCLR
def infoNCE(embeds1, embeds2, nodes, temperature):
    """
    Tính InfoNCE (Noise Contrastive Estimation)
    """
    # Normalize embeddings to unit sphere
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2) 
    
    # Pick embeddings for selected nodes
    pckEmbeds1 = embeds1[nodes]  # [batch_size, embed_dim]
    pckEmbeds2 = embeds2[nodes]  # [batch_size, embed_dim]
    
    # Positive pairs: same nodes in different embedding spaces
    nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temperature)  # [batch_size]
    
    # Negative pairs: each node in embeds1 vs all nodes in embeds2  
    deno = torch.exp(pckEmbeds1 @ embeds2.T / temperature).sum(-1) + 1e-8  # [batch_size]
    
    return (-torch.log(nume / deno)).mean()


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
            distill_config = get_distillation_config(kd_params())
        
        # Build detached copy of mps for teacher to prevent autograd graph bloat
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

        # Compute teacher representations (detached) and student with gradients
        if hasattr(teacher, 'get_student_aligned_representations') and student_type == 'StudentMyHeCo':
            teacher_mp, teacher_sc = teacher.get_student_aligned_representations(feats, mps_detached, nei_index)
            student_mp, student_sc = student.get_representations(feats, mps_detached, nei_index)
        elif hasattr(student, 'get_teacher_aligned_representations'):
            teacher_mp, teacher_sc = teacher.get_representations(feats, mps_detached, nei_index)
            student_mp, student_sc = student.get_teacher_aligned_representations(feats, mps_detached, nei_index)
        else:
            teacher_mp, teacher_sc = teacher.get_representations(feats, mps_detached, nei_index)
            student_mp, student_sc = student.get_representations(feats, mps_detached, nei_index)
        
        total_distill_loss = 0
        losses = {}
        
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
        
        total_loss = total_distill_loss
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