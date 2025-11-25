#!/usr/bin/env python3
"""
Model Architecture Analyzer - Load and analyze saved model parameters
"""

import torch
import pickle
import os
import sys
from collections import OrderedDict

# Add project root and code directory to path for imports (portable)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
code_dir = os.path.join(project_root, 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary model classes
try:
    from code.models.kd_heco import *
    from code.training.hetero_augmentations import *
    from code.models.contrast import *
    from code.models.sc_encoder import *
    print("✅ Successfully imported model classes")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Will try to load as state dict only")

def analyze_model_architecture(model_path):
    """Load and analyze the full model architecture with parameter details"""
    
    print(f"🔍 Analyzing model: {model_path}")
    print("=" * 80)
    
    try:
        # Load the model with different strategies
        try:
            # Try loading with torch.load first (handles GPU/CPU mapping)
            model_data = torch.load(model_path, map_location='cpu')
            print("✅ Loaded with torch.load")
        except Exception as e1:
            print(f"torch.load failed: {e1}")
            try:
                # Fallback to pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print("✅ Loaded with pickle")
            except Exception as e2:
                print(f"pickle.load failed: {e2}")
                # Try manual unpickling
                import pickletools
                with open(model_path, 'rb') as f:
                    print("📋 Pickle file contents:")
                    pickletools.dis(f)
                return None, None, 0
        
        # Inspect the loaded data structure
        print(f"📋 Model data type: {type(model_data)}")
        print(f"📋 Model data keys/attrs: {list(model_data.keys()) if isinstance(model_data, dict) else dir(model_data)}")
        
        # Check if it's a state dict or full model
        if isinstance(model_data, dict):
            # Look for common keys
            if 'state_dict' in model_data:
                state_dict = model_data['state_dict']
                print("📦 Found state_dict in model data")
                if 'epoch' in model_data:
                    print(f"📅 Model saved at epoch: {model_data['epoch']}")
                if 'best_val' in model_data:
                    print(f"🎯 Best validation score: {model_data['best_val']:.4f}")
            elif 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
                print("📦 Found model_state_dict in model data")
            else:
                # Check if this IS the state dict
                if all(isinstance(v, torch.Tensor) for v in model_data.values()):
                    state_dict = model_data
                    print("📦 Model data is direct state_dict")
                else:
                    print("📦 Model data structure:")
                    for k, v in list(model_data.items())[:10]:  # Show first 10 items
                        print(f"   {k}: {type(v)}")
                    
                    # Try to find tensor data
                    tensor_keys = [k for k, v in model_data.items() if isinstance(v, torch.Tensor)]
                    if tensor_keys:
                        print(f"📦 Found {len(tensor_keys)} tensor keys, treating as state_dict")
                        state_dict = {k: v for k, v in model_data.items() if isinstance(v, torch.Tensor)}
                    else:
                        print("❌ No tensor data found in model file")
                        return None, None, 0
        else:
            # Try to get state_dict from model object
            if hasattr(model_data, 'state_dict'):
                state_dict = model_data.state_dict()
                print("📦 Extracted state_dict from model object")
            else:
                print(f"❌ Cannot extract state_dict from {type(model_data)}")
                return None, None, 0
        
        print(f"🔢 Total parameter tensors: {len(state_dict)}")
        print("\n" + "=" * 80)
        
        # Analyze architecture by parameter groups
        component_params = {}
        total_params = 0
        
        print("📊 DETAILED PARAMETER BREAKDOWN:")
        print("-" * 80)
        print(f"{'Component':<50} {'Shape':<25} {'Parameters':<15}")
        print("-" * 80)
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
            else:
                print(f"⚠️ Skipping non-tensor parameter: {name} (type: {type(param)})")
                continue
            
            # Group by component
            component = get_component_name(name)
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param_count
            
            # Print detailed info
            shape_str = str(list(param.shape))
            print(f"{name:<50} {shape_str:<25} {param_count:<15,}")
        
        print("-" * 80)
        print(f"{'TOTAL':<50} {'':<25} {total_params:<15,}")
        
        # Component summary
        print(f"\n🏗️ COMPONENT SUMMARY:")
        print("-" * 50)
        print(f"{'Component':<35} {'Parameters':<15} {'Percentage'}")
        print("-" * 50)
        
        sorted_components = sorted(component_params.items(), key=lambda x: x[1], reverse=True)
        for component, count in sorted_components:
            percentage = (count / total_params) * 100
            print(f"{component:<35} {count:<15,} {percentage:>6.2f}%")
        
        print("-" * 50)
        print(f"{'TOTAL':<35} {total_params:<15,} {'100.00%':>8}")
        
        # Architecture insights
        print(f"\n🎯 ARCHITECTURE INSIGHTS:")
        print("-" * 40)
        
        # Identify heavy components
        heavy_components = [(comp, count) for comp, count in sorted_components if (count/total_params) > 0.05]
        if heavy_components:
            print("🔥 Heavy Components (>5% of parameters):")
            for comp, count in heavy_components:
                percentage = (count / total_params) * 100
                print(f"   • {comp}: {count:,} params ({percentage:.1f}%)")
        
        # Check for autoencoders
        autoencoder_params = sum(count for comp, count in component_params.items() 
                               if 'autoencoder' in comp.lower() or 'encoder' in comp.lower() or 'decoder' in comp.lower())
        if autoencoder_params > 0:
            ae_percentage = (autoencoder_params / total_params) * 100
            print(f"🤖 Autoencoder Components: {autoencoder_params:,} params ({ae_percentage:.1f}%)")
        
        # Check for augmentation pipeline
        aug_params = sum(count for comp, count in component_params.items() 
                        if 'augmentation' in comp.lower() or 'mask' in comp.lower())
        if aug_params > 0:
            aug_percentage = (aug_params / total_params) * 100
            print(f"🔄 Augmentation Pipeline: {aug_params:,} params ({aug_percentage:.1f}%)")
        
        # Memory estimation
        param_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"💾 Parameter Memory: {param_memory_mb:.1f} MB")
        
        return state_dict, component_params, total_params
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, 0

def get_component_name(param_name):
    """Extract component name from parameter name"""
    parts = param_name.split('.')
    
    # Handle nested components
    if len(parts) >= 2:
        if 'augmentation_pipeline' in param_name:
            if 'autoencoder' in param_name:
                # Extract autoencoder type and layer
                autoencoder_idx = None
                layer_info = ""
                for i, part in enumerate(parts):
                    if part.isdigit():
                        autoencoder_idx = part
                    elif 'encoder' in part or 'decoder' in part:
                        layer_info = part
                
                if autoencoder_idx is not None:
                    return f"augmentation.autoencoder_{autoencoder_idx}.{layer_info}"
                else:
                    return "augmentation.autoencoder"
            else:
                return f"augmentation.{parts[2] if len(parts) > 2 else 'other'}"
        
        elif 'cross_aug_learning' in param_name:
            return f"cross_aug_learning.{parts[2] if len(parts) > 2 else 'unknown'}"
        
        elif any(x in param_name for x in ['fc_list', 'feat_drop']):
            return "core.feature_projection"
        
        elif any(x in param_name for x in ['mp_encoder', 'mp']):
            return "core.mp_encoder"
        
        elif any(x in param_name for x in ['sc_encoder', 'sc']):
            return "core.sc_encoder"
        
        elif 'contrast' in param_name:
            return "core.contrast"
        
        elif any(x in param_name for x in ['att', 'attention']):
            return "core.attention"
        
        elif 'expert' in param_name:
            return "expert_components"
        
        else:
            return f"other.{parts[0]}"
    
    return "unknown"

def main():
    """Main analysis function"""
    
    # Look for the model file
    model_files = [
        "results/middle_teacher_heco_acm.pkl",
        "middle_teacher_heco_acm.pkl",
    os.path.join(project_root, "results", "middle_teacher_heco_acm.pkl")
    ]
    
    model_path = None
    for path in model_files:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ Model file not found. Looking for available .pkl files...")
        
        # Search in current directory and results
        search_dirs = [".", os.path.join(project_root, "results"), "results"]
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                pkl_files = [f for f in os.listdir(search_dir) if f.endswith('.pkl')]
                if pkl_files:
                    print(f"📁 Found .pkl files in {search_dir}:")
                    for f in pkl_files:
                        print(f"   • {f}")
        
        # Try to find any middle teacher file
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                pkl_files = [f for f in os.listdir(search_dir) if 'middle' in f.lower() and f.endswith('.pkl')]
                if pkl_files:
                    model_path = os.path.join(search_dir, pkl_files[0])
                    print(f"🎯 Using found file: {model_path}")
                    break
    
    if model_path and os.path.exists(model_path):
        analyze_model_architecture(model_path)
    else:
        print("❌ No suitable model file found!")
        print("💡 Usage: python analyze_model_architecture.py")
        print("   Make sure 'middle_teacher_heco_acm.pkl' is in current directory or results/ folder")

if __name__ == "__main__":
    main()