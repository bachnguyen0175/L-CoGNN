#!/usr/bin/env python3
"""
Custom Data Splitting Script for L-CoGNN
=========================================

Script để tạo train/val/test splits với tỷ lệ tùy chỉnh cho datasets trong L-CoGNN.

Usage:
    python create_custom_splits.py --dataset acm --train_ratio 0.6 --val_ratio 0.2 --test_ratio 0.2
    python create_custom_splits.py --dataset acm --train_size 500 --val_size 800 --test_size 1000
"""

import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import sys


def load_labels(dataset):
    """Load labels for the specified dataset"""
    label_path = f"data/{dataset}/labels.npy"
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels file not found: {label_path}")
    
    labels = np.load(label_path)
    return labels


def create_stratified_splits(labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                           train_size=None, val_size=None, test_size=None, 
                           balanced_train=False, seed=42):
    """
    Create stratified train/val/test splits
    
    Args:
        labels: Node labels array
        train_ratio, val_ratio, test_ratio: Ratios for splits (must sum to 1.0)
        train_size, val_size, test_size: Absolute sizes (overrides ratios if provided)
        balanced_train: If True, creates balanced training set (equal samples per class)
        seed: Random seed for reproducibility
        
    Returns:
        train_idx, val_idx, test_idx: Arrays of node indices
    """
    np.random.seed(seed)
    
    # Validate inputs
    if train_size is None and val_size is None and test_size is None:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
    
    total_nodes = len(labels)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    print(f"📊 Dataset Info:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Classes: {num_classes} {unique_labels}")
    
    # Calculate label distribution
    label_counts = Counter(labels)
    print(f"   Label distribution: {dict(label_counts)}")
    
    # Create node indices for each class
    class_indices = {}
    for label in unique_labels:
        class_indices[label] = np.where(labels == label)[0]
        np.random.shuffle(class_indices[label])  # Shuffle within each class
    
    train_idx = []
    val_idx = []
    test_idx = []
    
    if balanced_train and train_size is not None:
        # Balanced training set with fixed size
        samples_per_class = train_size // num_classes
        if samples_per_class * num_classes != train_size:
            print(f"⚠️  Warning: train_size {train_size} not divisible by {num_classes} classes")
            print(f"   Using {samples_per_class * num_classes} samples instead")
            train_size = samples_per_class * num_classes
        
        print(f"🎯 Creating balanced training set: {samples_per_class} samples per class")
        
        for label in unique_labels:
            indices = class_indices[label]
            if len(indices) < samples_per_class:
                raise ValueError(f"Class {label} has only {len(indices)} samples, need {samples_per_class}")
            
            train_idx.extend(indices[:samples_per_class])
            remaining = indices[samples_per_class:]
            class_indices[label] = remaining
    
    elif train_size is not None:
        # Fixed training size, stratified
        for label in unique_labels:
            indices = class_indices[label]
            class_ratio = len(indices) / total_nodes
            class_train_size = int(train_size * class_ratio)
            
            if len(indices) < class_train_size:
                class_train_size = len(indices)
                print(f"⚠️  Warning: Class {label} has only {len(indices)} samples, using all")
            
            train_idx.extend(indices[:class_train_size])
            remaining = indices[class_train_size:]
            class_indices[label] = remaining
    
    else:
        # Ratio-based training split
        for label in unique_labels:
            indices = class_indices[label]
            class_train_size = int(len(indices) * train_ratio)
            
            train_idx.extend(indices[:class_train_size])
            remaining = indices[class_train_size:]
            class_indices[label] = remaining
    
    # Handle remaining splits (val/test)
    remaining_indices = []
    for label in unique_labels:
        remaining_indices.extend(class_indices[label])
    
    remaining_indices = np.array(remaining_indices)
    np.random.shuffle(remaining_indices)
    
    if val_size is not None and test_size is not None:
        # Fixed sizes for val/test
        if len(remaining_indices) < val_size + test_size:
            raise ValueError(f"Not enough remaining samples: {len(remaining_indices)} < {val_size + test_size}")
        
        val_idx = remaining_indices[:val_size]
        test_idx = remaining_indices[val_size:val_size + test_size]
        
    elif val_size is not None:
        # Fixed val size, rest goes to test
        if len(remaining_indices) < val_size:
            raise ValueError(f"Not enough samples for validation: {len(remaining_indices)} < {val_size}")
        
        val_idx = remaining_indices[:val_size]
        test_idx = remaining_indices[val_size:]
        
    elif test_size is not None:
        # Fixed test size, rest goes to val
        if len(remaining_indices) < test_size:
            raise ValueError(f"Not enough samples for test: {len(remaining_indices)} < {test_size}")
        
        test_idx = remaining_indices[-test_size:]
        val_idx = remaining_indices[:-test_size]
        
    else:
        # Ratio-based val/test split
        remaining_total = val_ratio + test_ratio
        val_ratio_norm = val_ratio / remaining_total
        
        val_size_calc = int(len(remaining_indices) * val_ratio_norm)
        val_idx = remaining_indices[:val_size_calc]
        test_idx = remaining_indices[val_size_calc:]
    
    # Convert to numpy arrays and sort
    train_idx = np.sort(np.array(train_idx, dtype=int))
    val_idx = np.sort(np.array(val_idx, dtype=int))
    test_idx = np.sort(np.array(test_idx, dtype=int))
    
    return train_idx, val_idx, test_idx


def analyze_splits(labels, train_idx, val_idx, test_idx):
    """Analyze and print statistics about the splits"""
    total_nodes = len(labels)
    
    print(f"\n📋 SPLIT ANALYSIS:")
    print(f"{'='*50}")
    
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_labels = labels[indices]
        split_dist = Counter(split_labels)
        
        print(f"{split_name:>5}: {len(indices):4d} nodes ({len(indices)/total_nodes*100:5.1f}%) - {dict(split_dist)}")
    
    total_used = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"{'Total':>5}: {total_used:4d} nodes ({total_used/total_nodes*100:5.1f}%)")
    
    # Check for overlaps
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    overlaps = [
        len(train_set & val_set),
        len(train_set & test_set), 
        len(val_set & test_set)
    ]
    
    if sum(overlaps) > 0:
        print(f"⚠️  Overlaps detected: Train∩Val={overlaps[0]}, Train∩Test={overlaps[1]}, Val∩Test={overlaps[2]}")
    else:
        print(f"✅ No overlaps between splits")


def save_splits(dataset, train_idx, val_idx, test_idx, suffix="custom"):
    """Save the splits to files"""
    data_dir = f"data/{dataset}"
    
    # Create backup directory if needed
    backup_dir = f"{data_dir}/backup_splits"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Save new splits
    train_path = f"{data_dir}/train_{suffix}.npy"
    val_path = f"{data_dir}/val_{suffix}.npy"
    test_path = f"{data_dir}/test_{suffix}.npy"
    
    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    np.save(test_path, test_idx)
    
    print(f"\n💾 SAVED SPLITS:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Create custom train/val/test splits")
    
    parser.add_argument("--dataset", type=str, default="acm", 
                       choices=["acm", "dblp", "aminer", "freebase"],
                       help="Dataset name")
    
    # Ratio-based splitting
    parser.add_argument("--train_ratio", type=float, default=None,
                       help="Training set ratio (0.0-1.0)")
    parser.add_argument("--val_ratio", type=float, default=None,
                       help="Validation set ratio (0.0-1.0)")
    parser.add_argument("--test_ratio", type=float, default=None,
                       help="Test set ratio (0.0-1.0)")
    
    # Size-based splitting
    parser.add_argument("--train_size", type=int, default=None,
                       help="Training set size (absolute number)")
    parser.add_argument("--val_size", type=int, default=None,
                       help="Validation set size (absolute number)")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Test set size (absolute number)")
    
    # Options
    parser.add_argument("--balanced_train", action="store_true", default=False,
                       help="Create balanced training set (equal samples per class)")
    parser.add_argument("--suffix", type=str, default="custom",
                       help="Suffix for output files (train_SUFFIX.npy)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("🔧 CUSTOM DATA SPLITTING TOOL")
    print("="*40)
    
    # Load labels
    try:
        labels = load_labels(args.dataset)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Set default ratios if not specified
    if all(x is None for x in [args.train_ratio, args.val_ratio, args.test_ratio, 
                              args.train_size, args.val_size, args.test_size]):
        # Default: 60/20/20 split
        args.train_ratio = 0.6
        args.val_ratio = 0.2 
        args.test_ratio = 0.2
        print("Using default ratios: 60/20/20")
    
    # Create splits
    try:
        train_idx, val_idx, test_idx = create_stratified_splits(
            labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            balanced_train=args.balanced_train,
            seed=args.seed
        )
    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        sys.exit(1)
    
    # Analyze splits
    analyze_splits(labels, train_idx, val_idx, test_idx)
    
    # Save splits
    save_splits(args.dataset, train_idx, val_idx, test_idx, args.suffix)
    
    print(f"\n✅ Successfully created custom splits for {args.dataset} dataset!")
    print(f"\n💡 To use in training, update ratio parameter in kd_params.py:")
    print(f"   args.ratio = ['{args.suffix}']  # Single split")
    print(f"   # or")
    print(f"   args.ratio = ['{args.suffix}', '60']  # Your split + existing 60 split")


if __name__ == "__main__":
    main()