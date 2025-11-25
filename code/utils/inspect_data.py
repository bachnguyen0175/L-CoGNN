#!/usr/bin/env python3
"""
Data Inspection Tool for L-CoGNN
=================================

This script provides comprehensive analysis of the heterogeneous graph datasets
used in L-CoGNN, including structure, types, shapes, and sample data.

Usage:
    python inspect_data.py --dataset acm --ratio 20
    python inspect_data.py --dataset acm --ratio custom --detailed
    python inspect_data.py --dataset dblp --ratio 60 --show_samples
"""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import sys
from collections import Counter
import torch as th

# Ensure the utils package is importable when the script is executed directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

for _path in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.load_data import load_data


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def inspect_array(name, data, show_samples=False, sample_size=5):
    """Inspect a numpy array or tensor"""
    print(f"\n📦 {name}:")
    print(f"   Type: {type(data)}")
    
    if isinstance(data, (np.ndarray, th.Tensor)):
        if isinstance(data, th.Tensor):
            data_np = data.numpy() if not data.is_sparse else data.to_dense().numpy()
        else:
            data_np = data
            
        print(f"   Shape: {data_np.shape}")
        print(f"   Dtype: {data_np.dtype}")
        print(f"   Size: {data_np.size} elements")
        print(f"   Memory: {data_np.nbytes / 1024:.2f} KB")
        
        if data_np.size > 0:
            print(f"   Min: {np.min(data_np):.4f}")
            print(f"   Max: {np.max(data_np):.4f}")
            print(f"   Mean: {np.mean(data_np):.4f}")
            print(f"   Std: {np.std(data_np):.4f}")
            
            # Check for special values
            num_zeros = np.sum(data_np == 0)
            num_ones = np.sum(data_np == 1)
            num_nan = np.sum(np.isnan(data_np))
            num_inf = np.sum(np.isinf(data_np))
            
            if num_zeros > 0:
                print(f"   Zeros: {num_zeros} ({num_zeros/data_np.size*100:.1f}%)")
            if num_ones > 0:
                print(f"   Ones: {num_ones} ({num_ones/data_np.size*100:.1f}%)")
            if num_nan > 0:
                print(f"   ⚠️  NaN values: {num_nan}")
            if num_inf > 0:
                print(f"   ⚠️  Inf values: {num_inf}")
        
        if show_samples and data_np.size > 0:
            print(f"   Sample data (first {sample_size}):")
            if len(data_np.shape) == 1:
                print(f"      {data_np[:sample_size]}")
            elif len(data_np.shape) == 2:
                print(f"      {data_np[:sample_size, :min(5, data_np.shape[1])]}")
            
    elif isinstance(data, list):
        print(f"   Length: {len(data)}")
        if len(data) > 0:
            print(f"   First element type: {type(data[0])}")
            if isinstance(data[0], (np.ndarray, th.Tensor)):
                inspect_array(f"{name}[0]", data[0], show_samples, sample_size)


def inspect_sparse_tensor(name, tensor, show_samples=False):
    """Inspect a sparse tensor"""
    print(f"\n🕸️  {name} (Sparse Tensor):")
    print(f"   Type: {type(tensor)}")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Layout: {tensor.layout}")
    
    # Convert to dense for analysis (careful with large matrices!)
    if tensor.shape[0] * tensor.shape[1] < 10000000:  # Only if not too large
        dense = tensor.to_dense()
        dense_np = dense.numpy()
        
        num_nonzero = th.count_nonzero(dense).item()
        sparsity = 1 - (num_nonzero / (tensor.shape[0] * tensor.shape[1]))
        
        print(f"   Non-zero elements: {num_nonzero}")
        print(f"   Sparsity: {sparsity*100:.2f}%")
        print(f"   Min: {dense_np.min():.4f}")
        print(f"   Max: {dense_np.max():.4f}")
        print(f"   Mean: {dense_np.mean():.4f}")
        
        if show_samples:
            print(f"   Sample (top-left 5x5):")
            print(f"      {dense_np[:5, :5]}")
    else:
        print(f"   ⚠️  Matrix too large for detailed analysis")
        print(f"   Total elements: {tensor.shape[0] * tensor.shape[1]}")


def inspect_neighbors(name, nei_list, show_samples=False, sample_size=3):
    """Inspect neighbor lists"""
    print(f"\n🔗 {name} (Neighbor Lists):")
    print(f"   Type: {type(nei_list)}")
    print(f"   Number of nodes: {len(nei_list)}")
    
    if len(nei_list) > 0:
        # Analyze neighbor counts
        neighbor_counts = [len(neighbors) if isinstance(neighbors, (list, np.ndarray, th.Tensor)) 
                          else neighbors.shape[0] for neighbors in nei_list]
        
        print(f"   Neighbor count statistics:")
        print(f"      Min neighbors: {min(neighbor_counts)}")
        print(f"      Max neighbors: {max(neighbor_counts)}")
        print(f"      Mean neighbors: {np.mean(neighbor_counts):.2f}")
        print(f"      Median neighbors: {np.median(neighbor_counts):.2f}")
        
        # Distribution
        count_dist = Counter(neighbor_counts)
        print(f"   Neighbor count distribution (top 5):")
        for count, freq in count_dist.most_common(5):
            print(f"      {count} neighbors: {freq} nodes")
        
        if show_samples:
            print(f"   Sample neighbors (first {sample_size} nodes):")
            for i in range(min(sample_size, len(nei_list))):
                neighbors = nei_list[i]
                if isinstance(neighbors, th.Tensor):
                    neighbors = neighbors.numpy()
                print(f"      Node {i}: {neighbors[:10]}{'...' if len(neighbors) > 10 else ''}")


def inspect_labels(labels, train_idx, val_idx, test_idx):
    """Inspect label distribution across splits"""
    print_section("LABEL ANALYSIS")
    
    labels_np = labels.numpy() if isinstance(labels, th.Tensor) else labels
    
    # Overall label distribution
    if len(labels_np.shape) == 2:  # One-hot encoded
        label_indices = np.argmax(labels_np, axis=1)
        num_classes = labels_np.shape[1]
        print(f"   Label encoding: One-hot")
        print(f"   Shape: {labels_np.shape}")
        print(f"   Number of classes: {num_classes}")
    else:
        label_indices = labels_np
        num_classes = len(np.unique(label_indices))
        print(f"   Label encoding: Integer")
        print(f"   Shape: {labels_np.shape}")
        print(f"   Number of classes: {num_classes}")
    
    # Distribution
    label_counts = Counter(label_indices)
    print(f"\n   Overall distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"      Class {label}: {count} nodes ({count/len(label_indices)*100:.1f}%)")
    
    # Per-split distribution
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        if len(indices) == 0:
            continue
        idx_np = indices.numpy() if isinstance(indices, th.Tensor) else indices
        split_labels = label_indices[idx_np]
        split_dist = Counter(split_labels)
        
        print(f"\n   {split_name} split distribution ({len(idx_np)} nodes):")
        for label in sorted(split_dist.keys()):
            count = split_dist[label]
            print(f"      Class {label}: {count} nodes ({count/len(idx_np)*100:.1f}%)")


def inspect_features(feats, feat_names):
    """Inspect feature matrices"""
    print_section("FEATURE MATRICES")
    
    for i, (feat, name) in enumerate(zip(feats, feat_names)):
        print(f"\n{'─'*70}")
        inspect_array(f"Feature: {name}", feat, show_samples=False)
        
        # Check if it's one-hot encoded
        feat_np = feat.numpy() if isinstance(feat, th.Tensor) else feat
        if len(feat_np.shape) == 2:
            is_onehot = (np.all(np.sum(feat_np, axis=1) == 1) and 
                        np.all((feat_np == 0) | (feat_np == 1)))
            if is_onehot:
                print(f"   ✓ This appears to be one-hot encoded (identity matrix)")


def inspect_metapaths(mps, mp_names):
    """Inspect metapath matrices"""
    print_section("METAPATH ADJACENCY MATRICES")
    
    for i, (mp, name) in enumerate(zip(mps, mp_names)):
        print(f"\n{'─'*70}")
        inspect_sparse_tensor(f"Metapath: {name}", mp, show_samples=False)


def inspect_dataset_files(dataset):
    """Inspect raw dataset files"""
    print_section("RAW DATASET FILES")
    
    data_dir = f"data/{dataset}"
    
    if not os.path.exists(data_dir):
        print(f"   ❌ Dataset directory not found: {data_dir}")
        return
    
    print(f"   Dataset directory: {data_dir}")
    
    # List all files
    files = sorted(os.listdir(data_dir))
    
    print(f"\n   Files in directory:")
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"      {file:<30} {size:>10.2f} KB")
    
    # Check for splits
    print(f"\n   Available splits:")
    split_files = [f for f in files if f.startswith(('train_', 'val_', 'test_'))]
    split_ratios = set()
    for f in split_files:
        ratio = f.split('_')[1].replace('.npy', '')
        split_ratios.add(ratio)
    
    for ratio in sorted(split_ratios):
        train_file = f"train_{ratio}.npy"
        val_file = f"val_{ratio}.npy"
        test_file = f"test_{ratio}.npy"
        
        has_all = all(os.path.exists(os.path.join(data_dir, f)) 
                     for f in [train_file, val_file, test_file])
        
        status = "✓" if has_all else "✗"
        print(f"      {status} Ratio '{ratio}'")


def main():
    parser = argparse.ArgumentParser(description="Inspect L-CoGNN dataset structure")
    
    parser.add_argument("--dataset", type=str, default="acm",
                       choices=["acm", "dblp", "aminer", "freebase"],
                       help="Dataset to inspect")
    parser.add_argument("--ratio", type=str, default="20",
                       help="Train/val/test split ratio")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed analysis")
    parser.add_argument("--show_samples", action="store_true",
                       help="Show sample data values")
    
    args = parser.parse_args()
    
    print("🔍 L-CoGNN DATA INSPECTION TOOL")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Split ratio: {args.ratio}")
    
    # Inspect raw files first
    inspect_dataset_files(args.dataset)
    
    # Define type_num for each dataset
    type_nums = {
        "acm": [4019, 7167, 60],        # p, a, s
        "dblp": [4057, 14328, 20, 7723],  # a, p, c, t
        "aminer": [6564, 13329, 35],     # p, a, r
        "freebase": [3492, 2502, 33401, 4459]  # m, d, a, w
    }
    
    node_type_names = {
        "acm": ["Paper (P)", "Author (A)", "Subject (S)"],
        "dblp": ["Author (A)", "Paper (P)", "Conference (C)", "Term (T)"],
        "aminer": ["Paper (P)", "Author (A)", "Reference (R)"],
        "freebase": ["Movie (M)", "Director (D)", "Actor (A)", "Writer (W)"]
    }
    
    metapath_names = {
        "acm": ["PAP (Paper-Author-Paper)", "PSP (Paper-Subject-Paper)"],
        "dblp": ["APA", "APCPA", "APTPA"],
        "aminer": ["PAP", "PRP"],
        "freebase": ["MDM", "MAM", "MWM"]
    }
    
    type_num = type_nums[args.dataset]
    
    print_section("DATASET METADATA")
    print(f"   Node types: {len(type_num)}")
    for i, (name, count) in enumerate(zip(node_type_names[args.dataset], type_num)):
        print(f"      Type {i} - {name}: {count} nodes")
    
    # Load data
    print(f"\n   Loading data...")
    try:
        nei, feats, mps, pos, label, train, val, test = load_data(args.dataset, args.ratio, type_num)
        print(f"   ✓ Data loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return
    
    # Inspect neighbors
    print_section("NEIGHBOR LISTS")
    if args.dataset == "acm":
        inspect_neighbors("Authors neighbors", nei[0], args.show_samples)
        inspect_neighbors("Subject neighbors", nei[1], args.show_samples)
    elif args.dataset == "dblp":
        inspect_neighbors("Paper neighbors", nei[0], args.show_samples)
    elif args.dataset == "aminer":
        inspect_neighbors("Author neighbors", nei[0], args.show_samples)
        inspect_neighbors("Reference neighbors", nei[1], args.show_samples)
    elif args.dataset == "freebase":
        inspect_neighbors("Director neighbors", nei[0], args.show_samples)
        inspect_neighbors("Actor neighbors", nei[1], args.show_samples)
        inspect_neighbors("Writer neighbors", nei[2], args.show_samples)
    
    # Inspect features
    inspect_features(feats, node_type_names[args.dataset])
    
    # Inspect metapaths
    inspect_metapaths(mps, metapath_names[args.dataset])
    
    # Inspect position matrix
    print_section("POSITION MATRIX")
    inspect_sparse_tensor("Position (pos)", pos, args.show_samples)
    
    # Inspect labels and splits
    inspect_labels(label, train[0], val[0], test[0])
    
    # Inspect split indices
    print_section("TRAIN/VAL/TEST SPLITS")
    inspect_array("Train indices", train[0], args.show_samples)
    inspect_array("Val indices", val[0], args.show_samples)
    inspect_array("Test indices", test[0], args.show_samples)
    
    # Summary
    print_section("SUMMARY")
    print(f"   Dataset: {args.dataset}")
    print(f"   Node types: {len(type_num)}")
    print(f"   Total nodes (target type): {type_num[0]}")
    print(f"   Features: {len(feats)} feature matrices")
    print(f"   Metapaths: {len(mps)} adjacency matrices")
    print(f"   Neighbor lists: {len(nei)} types")
    
    if isinstance(label, th.Tensor):
        num_classes = label.shape[1] if len(label.shape) == 2 else len(th.unique(label))
    else:
        num_classes = label.shape[1] if len(label.shape) == 2 else len(np.unique(label))
    print(f"   Classes: {num_classes}")
    
    train_size = len(train[0])
    val_size = len(val[0])
    test_size = len(test[0])
    total_split = train_size + val_size + test_size
    
    print(f"\n   Split sizes:")
    print(f"      Train: {train_size} ({train_size/total_split*100:.1f}%)")
    print(f"      Val:   {val_size} ({val_size/total_split*100:.1f}%)")
    print(f"      Test:  {test_size} ({test_size/total_split*100:.1f}%)")
    
    print(f"\n✅ Inspection complete!")


if __name__ == "__main__":
    main()
