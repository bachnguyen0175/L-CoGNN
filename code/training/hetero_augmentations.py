#!/usr/bin/env python3
"""
Heterogeneous Graph Augmentation Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


class MetaPathConnector(nn.Module):
    """
    Meta-path based augmentation that connects nodes via existing meta-paths
    This creates virtual connections only within the original graph structure
    """
    def __init__(self, feats_dim_list: List[int], connection_strength: float = 0.1, num_metapaths: int = None, 
                 low_rank_dim: int = 64):
        super(MetaPathConnector, self).__init__()
        self.connection_strength = connection_strength
        self.feats_dim_list = feats_dim_list
        self.num_metapaths = num_metapaths  # Number of meta-paths (e.g., 2 for ACM: PAP, PSP)
        self.low_rank_dim = low_rank_dim
        
        # Learnable meta-path embeddings for each node type
        self.meta_path_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, dim) * 0.1) for dim in feats_dim_list
        ])
        
        # Use LOW-RANK projection to reduce parameters dramatically
        # Instead of dim x dim projection, use dim -> low_rank_dim -> dim
        # This reduces params from O(d²) to O(2*d*k) where k << d
        self.connection_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, min(low_rank_dim, dim), bias=False),  # Bottleneck
                nn.Linear(min(low_rank_dim, dim), dim, bias=False)   # Expand
            ) for dim in feats_dim_list
        ])
        
        # Input: propagated features (after projection, so dimension is dim not low_rank_dim)
        # Each meta-path can have different importance for augmentation
        if num_metapaths is not None and num_metapaths > 1:
            self.metapath_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim // 4),  # Input is full dim (after low-rank back-projection)
                    nn.Tanh(),
                    nn.Linear(dim // 4, 1, bias=False)
                ) for dim in feats_dim_list
            ])
        else:
            self.metapath_attention = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters for low-rank projections and attention"""
        # Initialize meta-path embeddings
        for embedding in self.meta_path_embeddings:
            nn.init.xavier_normal_(embedding, gain=0.1)
        
        # Initialize LOW-RANK projections (Sequential of two Linear layers)
        for projection_seq in self.connection_projections:
            assert isinstance(projection_seq, nn.Sequential), \
                "Projection must be Sequential (low-rank). Check initialization."
            for layer in projection_seq:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.0)
        
        # Initialize meta-path attention
        if self.metapath_attention is not None:
            for attn_module in self.metapath_attention:
                for layer in attn_module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight, gain=1.414)
    
    def forward(self, feats: List[torch.Tensor], mps: List[torch.Tensor] = None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Apply meta-path connections respecting original graph structure
        """
        # Validate inputs
        assert mps is not None, "Meta-paths (mps) are REQUIRED for structure-aware augmentation"
        assert isinstance(mps, list) or isinstance(mps, tuple), "mps must be a list/tuple of meta-path matrices"
        
        connected_feats = []
        connection_info = {
            'meta_path_connections': [],
            'connection_matrices': [],
            'structure_preserved': True  # Always true in structure-aware mode
        }
        
        # Apply meta-path connections to PRIMARY node type only (typically feat_idx=0)
        # In heterogeneous graphs (e.g., ACM), meta-paths are for target nodes (papers)
        # Other node types (authors, subjects) don't need meta-path augmentation
        for i, feat in enumerate(feats):
            if i == 0:  # Primary node type (e.g., papers in ACM)
                # Apply structure-aware meta-path connections
                connected_feat = self._apply_structure_aware_connections(
                    feat, i, mps
                )
                connected_feats.append(connected_feat)
                connection_info['meta_path_connections'].append(True)
            else:
                # Auxiliary node types: apply simple low-rank transformation for consistency
                # No meta-path propagation needed for these types
                if i < len(self.connection_projections):
                    transformed_feat = self.connection_projections[i](feat)
                    # Apply gating for consistency
                    gate = torch.sigmoid(self.meta_path_embeddings[i])
                    augmented_feat = feat + self.connection_strength * (transformed_feat * gate)
                    connected_feats.append(augmented_feat)
                else:
                    # No transformation available, use original
                    connected_feats.append(feat)
                connection_info['meta_path_connections'].append(False)
        
        return connected_feats, connection_info
    
    def _apply_structure_aware_connections(self, feat: torch.Tensor, feat_idx: int, 
                                         mps: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply meta-path connections that respect the original graph structure
        """
        # Validate inputs
        assert feat_idx < len(self.meta_path_embeddings), \
            f"feat_idx {feat_idx} out of range for meta_path_embeddings (size {len(self.meta_path_embeddings)})"
        
        # Project features using LOW-RANK projection
        projected_feat = self.connection_projections[feat_idx](feat)
        
        # REQUIREMENT: Meta-paths MUST be provided for augmentation to work
        assert mps is not None, "Meta-paths (mps) are required for structure-aware augmentation."
        
        # If mps is provided as single-element list wrapping multiple paths, handle it
        if len(mps) == 1 and isinstance(mps[0], list):
            meta_path_matrix = mps[0]  # Unwrap
        else:
            meta_path_matrix = mps  # Use directly
    
        # Check if mps[feat_idx] is a single matrix or list of matrices
        if isinstance(meta_path_matrix, list):
            # Multiple meta-paths: aggregate with attention (à la HAN)
            metapath_outputs = []
            for mp_idx, mp_matrix in enumerate(meta_path_matrix):
                # Validate dimension compatibility - NO SKIP, must fix dimensions
                assert mp_matrix.size(1) == feat.size(0), \
                    f"Meta-path {mp_idx} dimension mismatch: matrix.size(1)={mp_matrix.size(1)} != feat.size(0)={feat.size(0)}"
                
                # Propagate through this meta-path
                if mp_matrix.is_sparse:
                    mp_propagated = torch.sparse.mm(mp_matrix.float(), projected_feat.float())
                    mp_propagated = mp_propagated.to(feat.dtype)
                else:
                    mp_propagated = torch.mm(mp_matrix.float(), projected_feat.float()).to(feat.dtype)
                
                metapath_outputs.append(mp_propagated)
            
            # Must have at least one meta-path output
            assert len(metapath_outputs) > 0, "No valid meta-path outputs generated. Check meta-path matrices."
            
            if len(metapath_outputs) == 1:
                # Single meta-path, use directly
                propagated = metapath_outputs[0]
                gate = torch.sigmoid(self.meta_path_embeddings[feat_idx])
                meta_signal = self.connection_strength * (propagated * gate)
            else:
                # Multiple meta-paths
                assert self.metapath_attention is not None, \
                    "Meta-path attention required for multiple meta-paths. Initialize with num_metapaths > 1."
                assert feat_idx < len(self.metapath_attention), \
                    f"feat_idx {feat_idx} out of range for metapath_attention"
                
                # Stack all meta-path outputs: [num_metapaths, num_nodes, feat_dim]
                stacked_outputs = torch.stack(metapath_outputs, dim=0)
                
                # Compute attention scores for each meta-path
                # Use mean pooling to get graph-level representation
                mp_reprs = stacked_outputs.mean(dim=1)  # [num_metapaths, feat_dim]
                
                # Compute attention logits
                attn_logits = []
                for mp_repr in mp_reprs:
                    logit = self.metapath_attention[feat_idx](mp_repr.unsqueeze(0))
                    attn_logits.append(logit)
                
                attn_logits = torch.cat(attn_logits, dim=0)  # [num_metapaths, 1]
                attn_weights = F.softmax(attn_logits, dim=0)  # [num_metapaths, 1]
                
                # Weighted sum of meta-path outputs
                propagated = (stacked_outputs * attn_weights.unsqueeze(1)).sum(dim=0)
                
                # Apply gating
                gate = torch.sigmoid(self.meta_path_embeddings[feat_idx])
                meta_signal = self.connection_strength * (propagated * gate)
        else:
            # Single meta-path matrix (tensor, not list)
            # Validate dimension compatibility
            assert meta_path_matrix.size(1) == feat.size(0), \
                f"Meta-path dimension mismatch: matrix.size(1)={meta_path_matrix.size(1)} != feat.size(0)={feat.size(0)}"
            
            # Apply meta-path propagation using existing structure
            if meta_path_matrix.is_sparse:
                # Sparse matrix multiplication - ensure consistent dtypes for mixed precision compatibility
                meta_path_matrix = meta_path_matrix.float()  # Convert to float32 for sparse ops
                projected_feat_float = projected_feat.float()  # Ensure projected_feat is also float32
                propagated = torch.sparse.mm(meta_path_matrix, projected_feat_float)
                propagated = propagated.to(feat.dtype)  # Convert back to original dtype
            else:
                propagated = torch.mm(meta_path_matrix.float(), projected_feat.float()).to(feat.dtype)
            
            # Apply connection strength to propagation, use meta-path embedding as gate
            gate = torch.sigmoid(self.meta_path_embeddings[feat_idx])  # [1, feat_dim]
            meta_signal = self.connection_strength * (propagated * gate)
        
        # alpha ∈ [0.1, 0.2] controls the strength of initial residual connection
        # This preserves more original features while adding meta-path information
        alpha = 0.15  # Can be made learnable if needed
        connected_feat = (1 + alpha) * feat + (1 - alpha) * meta_signal
        
        return connected_feat
    
    def get_connection_strength(self) -> float:
        """Get current connection strength"""
        return self.connection_strength
    
    def set_connection_strength(self, strength: float):
        """Set connection strength"""
        self.connection_strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1


class HeteroAugmentationPipeline(nn.Module):
    """
    Heterogeneous graph augmentation pipeline with:
    - Structure-Aware Meta-Path Connections (respects original graph topology)
    - Low-Rank Projections for parameter efficiency
    - Semantic-level Meta-Path Attention
    """
    def __init__(self, feats_dim_list: List[int], augmentation_config: Dict[str, Any] = None):
        super(HeteroAugmentationPipeline, self).__init__()
        
        # Default configuration - Meta-path connections ALWAYS enabled
        default_config = {
            'use_meta_path_connections': True,
            'connection_strength': 0.1,
            'low_rank_dim': 64,
            'num_metapaths': 2
        }
        
        # Merge default_config with augmentation_config (augmentation_config overrides defaults)
        self.config = {**default_config, **(augmentation_config or {})}
        
        # Validate configuration
        assert self.config.get('use_meta_path_connections', True), \
            "Meta-path connections must be enabled."
        assert self.config.get('num_metapaths') is not None and self.config.get('num_metapaths') > 0, \
            "num_metapaths must be specified and > 0 for proper meta-path augmentation"
        
        # Initialize meta-path connector with REQUIRED parameters
        self.meta_path_connector = MetaPathConnector(
            feats_dim_list,
            self.config['connection_strength'],
            num_metapaths=self.config['num_metapaths'],
            low_rank_dim=self.config.get('low_rank_dim', 64)
        )
    
    def forward(self, feats: List[torch.Tensor], mps=None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Apply augmentation pipeline: Structure-aware meta-path connections with low-rank projections
        """
        assert mps is not None, \
            "Meta-paths (mps) are REQUIRED for structure-aware augmentation"
        assert isinstance(mps, list), \
            f"mps must be a list of meta-path matrices, got {type(mps)}"
        
        # Apply meta-path connections with low-rank projections
        aug_feats, connection_info = self.meta_path_connector(feats, mps)
        aug_info = {'connection_info': connection_info}
        
        return aug_feats, aug_info
    
    def get_multiple_augmentations(self, feats: List[torch.Tensor], mps=None,
                                  num_augmentations: int = 3) -> List[Tuple]:
        """
        Generate multiple different augmentations
        """
        augmentations = []
        for _ in range(num_augmentations):
            aug_feats, aug_info = self.forward(feats, mps)
            augmentations.append((aug_feats, aug_info))
        
        return augmentations
    
    def get_reconstruction_loss(self, aug_info: Dict) -> torch.Tensor:
        """
        Extract reconstruction loss from augmentation info for backward compatibility
        Note: Autoencoder removed, this always returns 0.0
        """
        return torch.tensor(0.0)
    
    def set_meta_path_connection_strength(self, strength: float):
        """
        Set the meta-path connection strength
        
        Args:
            strength: Connection strength between 0.0 and 1.0
        """
        if hasattr(self, 'meta_path_connector'):
            self.meta_path_connector.set_connection_strength(strength)
            self.config['connection_strength'] = strength
    
    def get_meta_path_connection_strength(self) -> float:
        """
        Get the current meta-path connection strength
        
        Returns:
            Current connection strength
        """
        if hasattr(self, 'meta_path_connector'):
            return self.meta_path_connector.get_connection_strength()
        return 0.0
    
    def disable_meta_path_connections(self):
        """Disable meta-path connections"""
        self.config['use_meta_path_connections'] = False
    
    def enable_meta_path_connections(self):
        """Enable meta-path connections"""
        self.config['use_meta_path_connections'] = True