#!/usr/bin/env python3
"""
Simplified Heterogeneous Graph Augmentation Module
Only includes: Node Feature Masking, Remasking Strategy, Edge Augmentation, Encoder-Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


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


class MetaPathAugmenter(nn.Module):
    """
    Meta-path edge augmentation for HeCo (simplified, no DGL dependency)
    """
    def __init__(self, drop_rate: float = 0.1):
        super(MetaPathAugmenter, self).__init__()
        self.drop_rate = drop_rate
    
    def forward(self, mps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply edge dropping to meta-paths
        """
        if self.training and self.drop_rate > 0:
            augmented_mps = []
            for mp in mps:
                if mp.is_sparse:
                    # Handle sparse tensors
                    mp = mp.coalesce()  # Ensure tensor is coalesced
                    indices = mp.indices()
                    values = mp.values()
                    
                    # Create dropout mask
                    edge_mask = torch.rand(values.size(0), device=mp.device) > self.drop_rate
                    
                    # Apply mask
                    filtered_indices = indices[:, edge_mask]
                    filtered_values = values[edge_mask]
                    
                    # Create new sparse tensor
                    augmented_mp = torch.sparse_coo_tensor(
                        filtered_indices, filtered_values, mp.size(), device=mp.device
                    ).coalesce()
                    augmented_mps.append(augmented_mp)
                else:
                    # Handle dense tensors
                    edge_mask = torch.rand_like(mp) > self.drop_rate
                    augmented_mp = mp * edge_mask.float()
                    augmented_mps.append(augmented_mp)
            
            return augmented_mps
        else:
            return mps


class SimpleAutoEncoder(nn.Module):
    """
    Simple Autoencoder for feature reconstruction (no DGL dependency)
    Implements encoder-decoder architecture with masking
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super(SimpleAutoEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder layers
        encoder_layers = []
        curr_dim = in_dim
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        curr_dim = hidden_dim
        for i in range(num_layers - 1):
            decoder_layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
        # Final layer to original dimension
        decoder_layers.append(nn.Linear(hidden_dim, in_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.414)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder-decoder
        """
        # Encode
        encoded = self.encoder(x)
        # Decode
        reconstructed = self.decoder(encoded)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoded representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from representation"""
        return self.decoder(z)


def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor, 
                       mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute reconstruction loss between original and reconstructed features
    """
    if mask is not None:
        # Only compute loss on masked positions
        loss = F.mse_loss(reconstructed[mask], original[mask])
    else:
        # Compute loss on all positions
        loss = F.mse_loss(reconstructed, original)
    return loss


class HeteroAugmentationPipeline(nn.Module):
    """
    Simplified augmentation pipeline with only core methods:
    - Node Feature Masking
    - Remasking Strategy  
    - Edge Augmentation
    - Encoder-Decoder
    """
    def __init__(self, feats_dim_list: List[int], augmentation_config: Dict[str, Any] = None):
        super(HeteroAugmentationPipeline, self).__init__()
        
        # Default configuration
        default_config = {
            'use_node_masking': True,
            'use_edge_augmentation': True,
            'use_autoencoder': True,
            'mask_rate': 0.1,
            'remask_rate': 0.3,
            'num_remasking': 2,
            'edge_drop_rate': 0.1,
            'autoencoder_hidden_dim': 64,
            'autoencoder_layers': 2,
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
        
        if self.config['use_edge_augmentation']:
            self.edge_augmenter = MetaPathAugmenter(
                self.config['edge_drop_rate']
            )
        
        # Initialize autoencoders for each feature type
        if self.config['use_autoencoder']:
            self.autoencoders = nn.ModuleList([
                SimpleAutoEncoder(
                    in_dim=feat_dim,
                    hidden_dim=self.config['autoencoder_hidden_dim'],
                    num_layers=self.config['autoencoder_layers']
                ) for feat_dim in feats_dim_list
            ])
    
    def forward(self, feats: List[torch.Tensor], mps: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        """
        Apply simplified augmentation pipeline
        """
        aug_feats = feats
        aug_mps = mps
        aug_info = {}
        
        # Apply node masking augmentation
        if self.config['use_node_masking'] and hasattr(self, 'node_masker'):
            aug_feats, mask_info = self.node_masker(aug_feats)
            aug_info['mask_info'] = mask_info
        
        # Apply edge augmentation
        if self.config['use_edge_augmentation'] and hasattr(self, 'edge_augmenter'):
            aug_mps = self.edge_augmenter(aug_mps)
            aug_info['edge_augmented'] = True
        
        # Apply autoencoder reconstruction
        if self.config['use_autoencoder'] and hasattr(self, 'autoencoders'):
            reconstructed_feats = []
            reconstruction_losses = []
            
            for i, (feat, autoencoder) in enumerate(zip(aug_feats, self.autoencoders)):
                # Get reconstruction
                reconstructed = autoencoder(feat)
                reconstructed_feats.append(reconstructed)
                
                # Compute reconstruction loss
                if 'mask_info' in aug_info and i < len(aug_info['mask_info']['masked_nodes']):
                    mask_nodes = aug_info['mask_info']['masked_nodes'][i]
                    if len(mask_nodes) > 0:
                        recon_loss = reconstruction_loss(feat, reconstructed, mask_nodes)
                        reconstruction_losses.append(recon_loss)
                else:
                    recon_loss = reconstruction_loss(feat, reconstructed)
                    reconstruction_losses.append(recon_loss)
            
            # Use reconstructed features as augmented features
            aug_feats = reconstructed_feats
            aug_info['reconstruction_losses'] = reconstruction_losses
            aug_info['total_reconstruction_loss'] = sum(reconstruction_losses) if reconstruction_losses else torch.tensor(0.0)
        
        return aug_feats, aug_mps, aug_info
    
    def get_multiple_augmentations(self, feats: List[torch.Tensor], mps: List[torch.Tensor], 
                                  num_augmentations: int = 2) -> List[Tuple]:
        """
        Generate multiple different augmentations
        """
        augmentations = []
        for _ in range(num_augmentations):
            aug_feats, aug_mps, aug_info = self.forward(feats, mps)
            augmentations.append((aug_feats, aug_mps, aug_info))
        
        return augmentations