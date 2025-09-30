#!/usr/bin/env python3
"""
Heterogeneous Graph Augmentation Module - Compatible with HeCo Architecture

Supported augmentations:
1. Node Feature Masking
2. Remasking Strategy
3. Structure-Aware Meta-Path Connections (respects original graph topology)
4. Encoder-Decoder with Masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from dgl.nn.pytorch import GATConv, GraphConv, GINConv
import numpy as np


class HeteroNodeMasker(nn.Module):
    """
    Node feature masking augmentation adapted for HeCo architecture
    """
    def __init__(self, feats_dim_list: List[int], mask_rate: float = 0.1, 
                 remask_rate: float = 0.3, num_remasking: int = 2):
        super(HeteroNodeMasker, self).__init__()
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remasking = num_remasking
        
        # Learnable mask tokens for each feature type
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim)) for dim in feats_dim_list
        ])
        
        # Initialize mask tokens
        for token in self.mask_tokens:
            nn.init.xavier_normal_(token, gain=1.414)
    
    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict]:
        """
        Apply node feature masking
        """
        masked_feats = []
        mask_info = {'masked_nodes': [], 'keep_nodes': []}
        
        for i, feat in enumerate(feats):
            if i < len(self.mask_tokens):
                masked_feat, mask_nodes, keep_nodes = self._mask_features(
                    feat, self.mask_tokens[i], self.mask_rate
                )
                masked_feats.append(masked_feat)
                mask_info['masked_nodes'].append(mask_nodes)
                mask_info['keep_nodes'].append(keep_nodes)
            else:
                # If no mask token for this type, return original features
                masked_feats.append(feat)
                mask_info['masked_nodes'].append(torch.tensor([]))
                mask_info['keep_nodes'].append(torch.arange(feat.size(0)))
        
        return masked_feats, mask_info
    
    def _mask_features(self, features: torch.Tensor, mask_token: torch.Tensor, 
                      mask_rate: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to node features"""
        num_nodes = features.size(0)
        perm = torch.randperm(num_nodes, device=features.device)
        
        # Random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        # Create masked features
        masked_features = features.clone()
        masked_features[mask_nodes] = 0.0
        masked_features[mask_nodes] += mask_token
        
        return masked_features, mask_nodes, keep_nodes
    
    def remask_features(self, features: torch.Tensor, feat_idx: int) -> torch.Tensor:
        """Apply remasking during training"""
        if feat_idx >= len(self.mask_tokens):
            return features
        
        num_nodes = features.size(0)
        perm = torch.randperm(num_nodes, device=features.device)
        num_remask_nodes = int(self.remask_rate * num_nodes)
        remask_nodes = perm[:num_remask_nodes]
        
        remasked_features = features.clone()
        remasked_features[remask_nodes] = 0.0
        remasked_features[remask_nodes] += self.mask_tokens[feat_idx]
        
        return remasked_features


class MetaPathConnector(nn.Module):
    """
    Meta-path based augmentation that connects nodes via existing meta-paths
    This creates virtual connections only within the original graph structure
    """
    def __init__(self, feats_dim_list: List[int], connection_strength: float = 0.1):
        super(MetaPathConnector, self).__init__()
        self.connection_strength = connection_strength
        self.feats_dim_list = feats_dim_list
        
        # Learnable meta-path embeddings for each node type
        self.meta_path_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, dim) * 0.1) for dim in feats_dim_list
        ])
        
        # Projection layers for meta-path connections
        self.connection_projections = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for dim in feats_dim_list
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        for embedding in self.meta_path_embeddings:
            nn.init.xavier_normal_(embedding, gain=0.1)
        
        for projection in self.connection_projections:
            nn.init.xavier_normal_(projection.weight, gain=1.0)
    
    def forward(self, feats: List[torch.Tensor], mps: List[torch.Tensor] = None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Apply meta-path connections respecting original graph structure
        
        Args:
            feats: List of node features for each type
            mps: List of meta-path adjacency matrices (optional, for structure-aware connections)
            
        Returns:
            connected_feats: Features with meta-path connections
            connection_info: Information about the connections made
        """
        connected_feats = []
        connection_info = {
            'meta_path_connections': [],
            'connection_matrices': [],
            'structure_preserved': mps is not None
        }
        
        # Apply meta-path connections to each node type
        for i, feat in enumerate(feats):
            if i < len(self.meta_path_embeddings):
                # Get structure-aware meta-path connections
                connected_feat = self._apply_structure_aware_connections(
                    feat, i, mps
                )
                connected_feats.append(connected_feat)
                
                # Store connection information
                connection_info['meta_path_connections'].append(True)
            else:
                # No meta-path embedding for this type, use original features
                connected_feats.append(feat)
                connection_info['meta_path_connections'].append(False)
        
        return connected_feats, connection_info
    
    def _apply_structure_aware_connections(self, feat: torch.Tensor, feat_idx: int, 
                                         mps: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply meta-path connections that respect the original graph structure
        
        Args:
            feat: Node features for this type
            feat_idx: Index of the feature type
            mps: Meta-path adjacency matrices
        """
        if feat_idx >= len(self.meta_path_embeddings):
            return feat
        
        # Project features
        projected_feat = self.connection_projections[feat_idx](feat)
        
        if mps is not None and feat_idx < len(mps) and mps[feat_idx] is not None:
            # Use structure-aware connections based on meta-paths
            meta_path_matrix = mps[feat_idx]
            
            # Check dimension compatibility
            if meta_path_matrix.size(1) != feat.size(0):
                # Dimension mismatch, fall back to local connections
                meta_signal = self._apply_local_meta_connections(feat, feat_idx)
            else:
                # Apply meta-path propagation using existing structure
                if meta_path_matrix.is_sparse:
                    # Sparse matrix multiplication - ensure consistent dtypes for mixed precision compatibility
                    meta_path_matrix = meta_path_matrix.float()  # Convert to float32 for sparse ops
                    projected_feat = projected_feat.float()      # Ensure projected_feat is also float32
                    propagated = torch.sparse.mm(meta_path_matrix, projected_feat)
                    propagated = propagated.to(feat.dtype)       # Convert back to original dtype
                else:
                    propagated = torch.mm(meta_path_matrix.float(), projected_feat.float()).to(feat.dtype)
                
                # Add meta-path embedding to propagated features
                meta_signal = self.connection_strength * (
                    propagated + self.meta_path_embeddings[feat_idx]
                )
        else:
            # Fallback: local neighborhood connections only
            meta_signal = self._apply_local_meta_connections(feat, feat_idx)
        
        # Add meta-path connections to original features
        connected_feat = feat + meta_signal
        
        return connected_feat
    
    def _apply_local_meta_connections(self, feat: torch.Tensor, feat_idx: int) -> torch.Tensor:
        """
        Apply local meta-path connections when no structure information is available
        Uses a more conservative approach that doesn't connect all nodes globally
        """
        # Project features 
        projected_feat = self.connection_projections[feat_idx](feat)
        
        # Create local connectivity pattern (e.g., k-nearest neighbors in feature space)
        num_nodes = feat.size(0)
        
        # Compute pairwise similarities to find local neighbors
        similarities = torch.mm(F.normalize(projected_feat, p=2, dim=1), 
                               F.normalize(projected_feat, p=2, dim=1).t())
        
        # Keep only top-k connections per node to maintain sparsity
        k = min(10, num_nodes // 10)  # At most 10 neighbors, or 10% of nodes
        topk_values, topk_indices = torch.topk(similarities, k=k+1, dim=1)  # +1 to exclude self
        
        # Create sparse connection matrix
        row_indices = torch.arange(num_nodes, device=feat.device).unsqueeze(1).expand(-1, k+1)
        col_indices = topk_indices
        
        # Exclude self-connections
        mask = col_indices != row_indices
        valid_rows = row_indices[mask]
        valid_cols = col_indices[mask]
        valid_values = topk_values[mask]
        
        # Normalize connection weights - handle variable length per node
        if len(valid_values) > 0:
            # Group by row and normalize within each row's connections
            unique_rows = torch.unique(valid_rows)
            normalized_values = torch.zeros_like(valid_values)
            
            for row in unique_rows:
                row_mask = valid_rows == row
                row_values = valid_values[row_mask]
                if len(row_values) > 0:
                    normalized_values[row_mask] = F.softmax(row_values, dim=0)
            
            valid_values = normalized_values
        else:
            # No valid connections, return empty
            valid_values = torch.tensor([], device=feat.device)
        
        # Create sparse adjacency matrix for local connections
        local_adj = torch.sparse_coo_tensor(
            torch.stack([valid_rows, valid_cols]), 
            valid_values,
            (num_nodes, num_nodes),
            device=feat.device
        ).coalesce()
        
        # Apply local meta-path propagation
        local_propagated = torch.sparse.mm(local_adj, projected_feat)
        
        # Add meta-path embedding
        meta_signal = self.connection_strength * (
            local_propagated + self.meta_path_embeddings[feat_idx]
        )
        
        return meta_signal
    
    def get_connection_strength(self) -> float:
        """Get current connection strength"""
        return self.connection_strength
    
    def set_connection_strength(self, strength: float):
        """Set connection strength"""
        self.connection_strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1


class Autoencoder(nn.Module):
    """
    Autoencoder exactly like original code with DGL GNN layers
    """
    def __init__(self, in_dim, hidden_dim, encoder, decoder, feat_drop, attn_drop, enc_num_layer,
                 dec_num_layer, num_heads, mask_rate, remask_rate, num_remasking):
        super(Autoencoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.dropout = feat_drop
        # encoder
        for i in range(enc_num_layer):
            if encoder == 'GAT' and num_heads == 1:
                self.encoder.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
            elif encoder == 'GAT' and num_heads != 1:
                if i == 0:
                    self.encoder.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
                elif i == 1:
                    self.encoder.append(GATConv(in_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
                elif i == 2:
                    self.encoder.append(
                        GATConv(in_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
            elif encoder == 'GCN':
                self.encoder.append(GraphConv(in_dim, hidden_dim, weight=False, bias=False,
                                              activation=nn.Identity(), allow_zero_in_degree=True))
            elif encoder == 'GIN':
                liner = torch.nn.Linear(in_dim, hidden_dim)
                self.encoder.append(GINConv(liner, 'sum', activation=nn.Identity()))
        # decoder
        for i in range(dec_num_layer):
            if decoder == 'GAT' and num_heads == 1:
                self.decoder.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
            elif decoder == 'GAT' and num_heads != 1:
                if i == 0:
                    self.decoder.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
                elif i == 1:
                    self.decoder.append(
                        GATConv(in_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
                elif i == 2:
                    self.decoder.append(
                        GATConv(in_dim * num_heads, hidden_dim, num_heads, feat_drop, attn_drop, activation=F.elu))
            elif decoder == 'GCN':
                self.decoder.append(GraphConv(in_dim, hidden_dim, weight=False, bias=False,
                                              activation=nn.Identity(), allow_zero_in_degree=True))
            elif decoder == 'GIN':
                liner = torch.nn.Linear(in_dim, hidden_dim)
                self.decoder.append(GINConv(liner, 'min', activation=nn.Identity()))
        # random_mask
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remasking = num_remasking

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.encoder_to_decoder = nn.Linear(in_dim * num_heads, in_dim, bias=False)
        self.decoder_to_contrastive = nn.Linear(in_dim * num_heads, in_dim, bias=False)

        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def forward(self, g, x, drop_g1=None, drop_g2=None):
        # mask
        pre_use_g, mask_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self.mask_rate)
        use_g = drop_g1 if drop_g1 is not None else g
        # multi-layer encoder
        Encode = []
        for i, layer in enumerate(self.encoder):
            if i == 0:
                enc_rep = layer(use_g, mask_x).flatten(1)
            else:
                enc_rep = layer(use_g, enc_rep).flatten(1)
            # enc_rep = F.dropout(enc_rep, self.dropout, training=self.training)
            Encode.append(enc_rep)
        Es = torch.stack(Encode, dim=1)  # (N, M, D * K)
        Es = torch.mean(Es, dim=1)
        # encode_to_decode
        origin_rep = self.encoder_to_decoder(Es)
        # decode
        Decode = []
        loss_rec_all = 0
        for i in range(self.num_remasking):
            # remask
            rep = origin_rep.clone()
            rep, remask_nodes, rekeep_nodes = self.random_remask(pre_use_g, rep, self.remask_rate)
            # multi-layer decoder
            for i, layer in enumerate(self.decoder):
                if i == 0:
                    recon = layer(pre_use_g, rep).flatten(1)
                else:
                    recon = layer(pre_use_g, recon).flatten(1)
                # recon = F.dropout(recon, self.dropout, training=self.training)
            Decode.append(recon)
            Ds = torch.stack(Decode, dim=1)  # (N, M, D * K)
            Ds = torch.mean(Ds, dim=1)

        return self.decoder_to_contrastive(Ds)

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, g, rep, remask_rate=0.5):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes



class HeteroAugmentationPipeline(nn.Module):
    """
    Heterogeneous graph augmentation pipeline with:
    - Node Feature Masking
    - Structure-Aware Meta-Path Connections (respects original graph topology)
    - Remasking Strategy  
    - Encoder-Decoder with Masking (DGL-based)
    """
    def __init__(self, feats_dim_list: List[int], augmentation_config: Dict[str, Any] = None):
        super(HeteroAugmentationPipeline, self).__init__()
        
        # Default configuration - Node masking + Meta-path connections + Encoder-Decoder with masking
        default_config = {
            'use_node_masking': True,
            'use_meta_path_connector': True,  # New: Meta-path connections
            'use_autoencoder': True,  # Enabled - DGL Autoencoder with masking
            'mask_rate': 0.1,
            'remask_rate': 0.3,
            'num_remasking': 2,
            'connection_strength': 0.1,
            'autoencoder_hidden_dim': 64,
            'encoder': 'GAT',
            'decoder': 'GAT', 
            'feat_drop': 0.1,
            'attn_drop': 0.1,
            'enc_num_layer': 2,
            'dec_num_layer': 2,
            'num_heads': 1,
            'reconstruction_weight': 0.1
        }
        
        self.config = {**default_config, **(augmentation_config or {})}
        
        # Initialize augmentation modules
        if self.config['use_node_masking']:
            self.node_masker = HeteroNodeMasker(
                feats_dim_list, 
                self.config['mask_rate'],
                self.config['remask_rate'],
                self.config['num_remasking']
            )
        
        # Initialize meta-path connector
        if self.config['use_meta_path_connector']:
            self.meta_path_connector = MetaPathConnector(
                feats_dim_list,
                self.config['connection_strength']
            )
        
        # Initialize autoencoders for each feature type
        if self.config['use_autoencoder']:
            self.autoencoders = nn.ModuleList([
                Autoencoder(
                    in_dim=feat_dim,
                    hidden_dim=self.config['autoencoder_hidden_dim'],
                    encoder=self.config['encoder'],
                    decoder=self.config['decoder'],
                    feat_drop=self.config['feat_drop'],
                    attn_drop=self.config['attn_drop'],
                    enc_num_layer=self.config['enc_num_layer'],
                    dec_num_layer=self.config['dec_num_layer'],
                    num_heads=self.config['num_heads'],
                    mask_rate=self.config['mask_rate'],
                    remask_rate=self.config['remask_rate'],
                    num_remasking=self.config['num_remasking']
                ) for feat_dim in feats_dim_list
            ])
    
    def forward(self, feats: List[torch.Tensor], gs=None, mps=None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Apply augmentation pipeline: Node masking + Meta-path connections + Encoder-Decoder with masking
        Args:
            feats: List of node features
            gs: Optional list of DGL graphs for autoencoder (if None, autoencoder will be skipped)
            mps: Optional list of meta-path adjacency matrices for structure-aware connections
        """
        aug_feats = feats
        aug_info = {}
        
        # Apply node masking augmentation
        if self.config['use_node_masking'] and hasattr(self, 'node_masker'):
            aug_feats, mask_info = self.node_masker(aug_feats)
            aug_info['mask_info'] = mask_info
        
        # Apply meta-path connections (structure-aware)
        if self.config['use_meta_path_connector'] and hasattr(self, 'meta_path_connector'):
            aug_feats, connection_info = self.meta_path_connector(aug_feats, mps)
            aug_info['connection_info'] = connection_info
        
        # Apply autoencoder reconstruction with masking (requires DGL graphs)
        if self.config['use_autoencoder'] and hasattr(self, 'autoencoders') and gs is not None:
            reconstructed_feats = []
            
            try:
                for i, (feat, autoencoder) in enumerate(zip(aug_feats, self.autoencoders)):
                    # Use corresponding graph for this feature type
                    g = gs[min(i, len(gs)-1)]  # Use last graph if not enough graphs
                    
                    # Get reconstruction using DGL autoencoder with masking
                    reconstructed = autoencoder(g, feat)
                    reconstructed_feats.append(reconstructed)
                
                # Use reconstructed features as augmented features
                aug_feats = reconstructed_feats
                aug_info['autoencoder_applied'] = True
                
            except Exception as e:
                # If autoencoder fails, use original features
                aug_info['autoencoder_skipped'] = str(e)
        elif self.config['use_autoencoder'] and gs is None:
            aug_info['autoencoder_skipped'] = 'No DGL graphs provided'
        
        return aug_feats, aug_info
    
    def get_multiple_augmentations(self, feats: List[torch.Tensor], gs=None, mps=None,
                                  num_augmentations: int = 3) -> List[Tuple]:
        """
        Generate multiple different augmentations
        """
        augmentations = []
        for _ in range(num_augmentations):
            aug_feats, aug_info = self.forward(feats, gs, mps)
            augmentations.append((aug_feats, aug_info))
        
        return augmentations
    
    def get_reconstruction_loss(self, aug_info: Dict) -> torch.Tensor:
        """
        Extract reconstruction loss from augmentation info for backward compatibility
        """
        if 'reconstruction_losses' in aug_info:
            return sum(aug_info['reconstruction_losses']) if aug_info['reconstruction_losses'] else torch.tensor(0.0)
        elif 'total_reconstruction_loss' in aug_info:
            return aug_info['total_reconstruction_loss']
        else:
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
        self.config['use_meta_path_connector'] = False
    
    def enable_meta_path_connections(self):
        """Enable meta-path connections"""
        self.config['use_meta_path_connector'] = True
    
    def get_pruning_targets(self, feats: List[torch.Tensor], aug_info: Dict) -> Dict:
        """
        Generate pruning targets based on augmentation patterns for expert teacher training
        
        Args:
            feats: Original node features
            aug_info: Augmentation information from forward pass
            
        Returns:
            Dict containing pruning targets and importance scores
        """
        pruning_targets = {}
        
        # Node-level pruning targets based on masking patterns
        if 'masked_nodes' in aug_info and len(aug_info['masked_nodes']) > 0:
            for i, masked_nodes in enumerate(aug_info['masked_nodes']):
                if len(masked_nodes) > 0:
                    # Nodes that were masked are less important for this feature type
                    num_nodes = feats[i].size(0)
                    node_importance = torch.ones(num_nodes, device=feats[i].device)
                    node_importance[masked_nodes] *= 0.3  # Reduce importance of masked nodes
                    pruning_targets[f'node_importance_{i}'] = node_importance
        
        # Structure-level pruning targets based on meta-path connections
        if 'meta_path_connections' in aug_info:
            connection_matrices = aug_info.get('connection_matrices', [])
            structure_importance = []
            
            for conn_matrix in connection_matrices:
                if conn_matrix is not None:
                    # Calculate structural centrality as importance measure
                    if hasattr(conn_matrix, 'sum'):
                        centrality = conn_matrix.sum(dim=1)  # Sum of connections
                        centrality = centrality / (centrality.max() + 1e-8)  # Normalize
                        structure_importance.append(centrality)
            
            if structure_importance:
                pruning_targets['structure_importance'] = torch.stack(structure_importance, dim=0)
        
        # Attention-level pruning targets
        if hasattr(self, 'meta_path_connector'):
            # Meta-path level importance
            mp_strength = self.get_meta_path_connection_strength()
            pruning_targets['meta_path_importance'] = torch.tensor(mp_strength)
            
        # Feature-level importance based on reconstruction difficulty
        if 'reconstruction_losses' in aug_info and aug_info['reconstruction_losses']:
            feature_difficulty = []
            for loss in aug_info['reconstruction_losses']:
                if isinstance(loss, torch.Tensor):
                    feature_difficulty.append(loss.detach())
                    
            if feature_difficulty:
                # Features with higher reconstruction loss are more important
                feature_importance = torch.stack(feature_difficulty)
                feature_importance = F.softmax(feature_importance, dim=0)
                pruning_targets['feature_importance'] = feature_importance
        
        return pruning_targets
    
    def create_expert_training_batch(self, feats: List[torch.Tensor], mps=None, 
                                   batch_size: int = None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Create a specialized training batch for pruning expert with multiple augmentation views
        
        Args:
            feats: Original node features
            mps: Meta-path adjacency matrices
            batch_size: Optional batch size for sampling
            
        Returns:
            Tuple of (multi_view_features, comprehensive_aug_info)
        """
        if batch_size is not None and batch_size < feats[0].size(0):
            # Sample nodes for efficiency
            indices = torch.randperm(feats[0].size(0), device=feats[0].device)[:batch_size]
            sampled_feats = [feat[indices] for feat in feats]
        else:
            sampled_feats = feats
            indices = None
        
        # Generate multiple augmentation views with different strengths
        augmentation_views = []
        comprehensive_info = {
            'view_count': 3,
            'augmentation_strengths': [0.1, 0.2, 0.3],
            'view_infos': [],
            'consensus_targets': {},
            'diversity_measures': []
        }
        
        # Create three different augmentation views
        for i, strength in enumerate([0.1, 0.2, 0.3]):
            # Temporarily adjust augmentation strength
            original_mask_rate = self.config.get('mask_rate', 0.1)
            original_connection_strength = self.get_meta_path_connection_strength()
            
            # Update strengths
            if hasattr(self, 'node_masker'):
                self.node_masker.mask_rate = strength
            self.set_meta_path_connection_strength(strength)
            
            # Generate augmented view
            aug_feats_view, aug_info_view = self.forward(sampled_feats, gs=None, mps=mps)
            
            # Get pruning targets for this view
            pruning_targets_view = self.get_pruning_targets(sampled_feats, aug_info_view)
            aug_info_view['pruning_targets'] = pruning_targets_view
            
            augmentation_views.append(aug_feats_view)
            comprehensive_info['view_infos'].append(aug_info_view)
            
            # Restore original settings
            if hasattr(self, 'node_masker'):
                self.node_masker.mask_rate = original_mask_rate
            self.set_meta_path_connection_strength(original_connection_strength)
        
        # Create consensus pruning targets across views
        comprehensive_info['consensus_targets'] = self._create_consensus_targets(
            comprehensive_info['view_infos']
        )
        
        # Calculate view diversity measures
        comprehensive_info['diversity_measures'] = self._calculate_view_diversity(
            augmentation_views
        )
        
        return augmentation_views, comprehensive_info
    
    def _create_consensus_targets(self, view_infos: List[Dict]) -> Dict:
        """Create consensus pruning targets from multiple augmentation views"""
        consensus = {}
        
        # Aggregate node importance across views
        node_importances = []
        for view_info in view_infos:
            if 'pruning_targets' in view_info:
                for key, value in view_info['pruning_targets'].items():
                    if 'node_importance' in key:
                        node_importances.append(value)
        
        if node_importances:
            consensus['avg_node_importance'] = torch.stack(node_importances, dim=0).mean(dim=0)
            consensus['std_node_importance'] = torch.stack(node_importances, dim=0).std(dim=0)
        
        # Aggregate feature importance
        feature_importances = []
        for view_info in view_infos:
            if 'pruning_targets' in view_info and 'feature_importance' in view_info['pruning_targets']:
                feature_importances.append(view_info['pruning_targets']['feature_importance'])
        
        if feature_importances:
            consensus['avg_feature_importance'] = torch.stack(feature_importances, dim=0).mean(dim=0)
            consensus['feature_agreement'] = torch.stack(feature_importances, dim=0).std(dim=0)
        
        return consensus
    
    def _calculate_view_diversity(self, augmentation_views: List[List[torch.Tensor]]) -> List[float]:
        """Calculate diversity measures between different augmentation views"""
        diversities = []
        
        for i in range(len(augmentation_views)):
            for j in range(i + 1, len(augmentation_views)):
                view1_feats = augmentation_views[i]
                view2_feats = augmentation_views[j]
                
                # Calculate cosine similarity between views
                view_similarity = 0.0
                for feat1, feat2 in zip(view1_feats, view2_feats):
                    if feat1.size() == feat2.size():
                        feat1_flat = feat1.view(-1)
                        feat2_flat = feat2.view(-1)
                        similarity = F.cosine_similarity(feat1_flat, feat2_flat, dim=0)
                        view_similarity += similarity.item()
                
                view_similarity /= len(view1_feats)
                diversity = 1.0 - abs(view_similarity)  # Higher diversity = lower similarity
                diversities.append(diversity)
        
        return diversities