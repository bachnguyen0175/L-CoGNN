#!/usr/bin/env python3
"""
Heterogeneous Graph Augmentation Module - Compatible with HeCo Architecture

Supported augmentations:
1. Node Feature Masking
2. Remasking Strategy
3. Encoder-Decoder with Masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from dgl.nn.pytorch import GATConv, GraphConv, GINConv


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
    - Remasking Strategy  
    - Encoder-Decoder with Masking (DGL-based)
    """
    def __init__(self, feats_dim_list: List[int], augmentation_config: Dict[str, Any] = None):
        super(HeteroAugmentationPipeline, self).__init__()
        
        # Default configuration - Node masking + Encoder-Decoder with masking
        default_config = {
            'use_node_masking': True,
            'use_autoencoder': True,  # Enabled - DGL Autoencoder with masking
            'mask_rate': 0.1,
            'remask_rate': 0.3,
            'num_remasking': 2,
            # Autoencoder config (enabled for DGL-based masking)
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
    
    def forward(self, feats: List[torch.Tensor], gs=None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Apply augmentation pipeline: Node masking + Encoder-Decoder with masking
        Args:
            feats: List of node features
            gs: Optional list of DGL graphs for autoencoder (if None, autoencoder will be skipped)
        """
        aug_feats = feats
        aug_info = {}
        
        # Apply node masking augmentation
        if self.config['use_node_masking'] and hasattr(self, 'node_masker'):
            aug_feats, mask_info = self.node_masker(aug_feats)
            aug_info['mask_info'] = mask_info
        
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
    
    def get_multiple_augmentations(self, feats: List[torch.Tensor], gs=None, 
                                  num_augmentations: int = 2) -> List[Tuple]:
        """
        Generate multiple different augmentations
        """
        augmentations = []
        for _ in range(num_augmentations):
            aug_feats, aug_info = self.forward(feats, gs)
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