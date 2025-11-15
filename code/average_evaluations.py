"""
Average Multiple Evaluation Results
Combines multiple comprehensive evaluation JSON files and computes mean ± std
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import glob
import sys


def load_json_files(pattern="comprehensive_eval_acm_*.json"):
    """Load all matching JSON files"""
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"❌ No files found matching pattern: {pattern}")
        sys.exit(1)
    
    print(f"📁 Found {len(json_files)} evaluation files:")
    for f in sorted(json_files):
        print(f"   - {Path(f).name}")
    
    data_list = []
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data_list.append(data)
    
    return data_list, json_files


def extract_metrics(data_list, model_name):
    """Extract all metrics for a specific model across all runs"""
    metrics = {
        'node_classification': {'accuracy': [], 'macro_f1': [], 'micro_f1': []},
        'link_prediction': {'auc': [], 'ap': [], 'hits_at_10': [], 'hits_at_50': [], 'hits_at_100': []},
        'parameters': []
    }
    
    for data in data_list:
        if model_name not in data['models']:
            continue
        
        model_data = data['models'][model_name]
        
        # Node classification
        for key in ['accuracy', 'macro_f1', 'micro_f1']:
            if key in model_data.get('node_classification', {}):
                metrics['node_classification'][key].append(model_data['node_classification'][key])
        
        # Link prediction
        for key in ['auc', 'ap', 'hits_at_10', 'hits_at_50', 'hits_at_100']:
            if key in model_data.get('link_prediction', {}):
                metrics['link_prediction'][key].append(model_data['link_prediction'][key])
        
        # Node clustering removed - not a primary metric for graph KD
        
        # Parameters
        if 'total_parameters' in model_data:
            metrics['parameters'].append(model_data['total_parameters'])
    
    return metrics


def compute_statistics(values):
    """Compute mean and std for a list of values"""
    if not values:
        return None, None
    
    values = np.array(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    
    return mean, std


def create_averaged_results(data_list):
    """Create averaged results from multiple evaluation runs"""
    # Get dataset and model types from first file
    first_data = data_list[0]
    dataset = first_data.get('dataset', 'unknown')
    model_names = list(first_data['models'].keys())
    
    averaged_results = {
        'dataset': dataset,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_runs': len(data_list),
        'source_files': [],
        'models': {}
    }
    
    # Process each model
    for model_name in model_names:
        print(f"\n📊 Processing {model_name}...")
        
        metrics = extract_metrics(data_list, model_name)
        
        # Get embedding dim and model type from first run
        model_info = first_data['models'][model_name]
        
        model_results = {
            'model_type': model_info.get('model_type', model_name),
            'embedding_dim': model_info.get('embedding_dim', 'unknown'),
            'node_classification': {},
            'link_prediction': {},
            'parameters': {}
        }
        
        # Node Classification
        for key in ['accuracy', 'macro_f1', 'micro_f1']:
            mean, std = compute_statistics(metrics['node_classification'][key])
            if mean is not None:
                model_results['node_classification'][key] = mean
                model_results['node_classification'][f'{key}_std'] = std
                print(f"   {key}: {mean:.4f} ± {std:.4f}")
        
        # Link Prediction
        print(f"\n   Link Prediction:")
        for key in ['auc', 'ap', 'hits_at_10', 'hits_at_50', 'hits_at_100']:
            mean, std = compute_statistics(metrics['link_prediction'][key])
            if mean is not None:
                model_results['link_prediction'][key] = mean
                model_results['link_prediction'][f'{key}_std'] = std
                print(f"   {key}: {mean:.4f} ± {std:.4f}")
        
        # Node Clustering removed - not a primary metric for graph KD evaluation
        print(f"   (Node clustering removed - focus on primary tasks)")
        
        # Parameters (should be constant)
        if metrics['parameters']:
            model_results['parameters']['total_parameters'] = int(metrics['parameters'][0])
            model_results['parameters']['trainable_parameters'] = int(metrics['parameters'][0])
        
        averaged_results['models'][model_name] = model_results
    
    return averaged_results


def compute_comparison_metrics(averaged_results):
    """Compute comparison metrics between teacher and student"""
    if 'teacher' not in averaged_results['models'] or 'student' not in averaged_results['models']:
        return averaged_results
    
    teacher = averaged_results['models']['teacher']
    student = averaged_results['models']['student']
    
    comparison = {
        'parameter_reduction': 1.0 - (student['parameters']['total_parameters'] / 
                                      teacher['parameters']['total_parameters']),
        'node_classification': {},
        'link_prediction': {}
    }
    
    # Store forgetting rates for weighted AF calculation
    # Focus on primary tasks: 50% node classification + 50% link prediction
    forgetting_rates = []
    task_weights = {
        'node_classification': 0.5,
        'link_prediction': 0.5
    }
    
    # Node classification retention and forgetting
    for key in ['accuracy', 'macro_f1', 'micro_f1']:
        if key in teacher['node_classification'] and key in student['node_classification']:
            teacher_val = teacher['node_classification'][key]
            student_val = student['node_classification'][key]
            retention = student_val / teacher_val
            comparison['node_classification'][f'{key}_retention'] = retention
            
            # Calculate forgetting: max(0, teacher - student) / teacher
            forgetting = max(0, teacher_val - student_val) / teacher_val
            comparison['node_classification'][f'{key}_forgetting'] = forgetting
            forgetting_rates.append((forgetting, task_weights['node_classification']))
    
    # Link prediction retention and forgetting
    for key in ['auc', 'ap']:  # Only use main metrics for AF
        if key in teacher['link_prediction'] and key in student['link_prediction']:
            teacher_val = teacher['link_prediction'][key]
            student_val = student['link_prediction'][key]
            retention = student_val / teacher_val
            comparison['link_prediction'][f'{key}_retention'] = retention
            
            # Calculate forgetting
            forgetting = max(0, teacher_val - student_val) / teacher_val
            comparison['link_prediction'][f'{key}_forgetting'] = forgetting
            forgetting_rates.append((forgetting, task_weights['link_prediction']))
    
    # Also compute retention for hits@k (but not for AF to avoid bias)
    for key in ['hits_at_10', 'hits_at_50', 'hits_at_100']:
        if key in teacher['link_prediction'] and key in student['link_prediction']:
            retention = student['link_prediction'][key] / teacher['link_prediction'][key]
            comparison['link_prediction'][f'{key}_retention'] = retention
    
    # Node clustering removed - not a primary metric for graph KD evaluation
    
    # Calculate Weighted Average Forget (AF) - Lower is better!
    # Focus on primary tasks: 50% node classification + 50% link prediction
    if forgetting_rates:
        # Compute weighted average
        total_forgetting = sum(forgetting * weight for forgetting, weight in forgetting_rates)
        total_weight = sum(weight for _, weight in forgetting_rates)
        average_forget = total_forgetting / total_weight if total_weight > 0 else 0
        
        comparison['average_forget'] = average_forget
        comparison['average_forget_percentage'] = average_forget * 100
        comparison['weighting_scheme'] = 'node_classification: 50%, link_prediction: 50%'
        
        # Categorize distillation quality based on AF
        if average_forget < 0.01:  # < 1% average forgetting
            quality = "Excellent ✨"
        elif average_forget < 0.03:  # 1-3% forgetting
            quality = "Very Good ✅"
        elif average_forget < 0.05:  # 3-5% forgetting
            quality = "Good ✓"
        elif average_forget < 0.10:  # 5-10% forgetting
            quality = "Fair ⚠️"
        else:  # > 10% forgetting
            quality = "Needs Improvement ⚠️⚠️"
        
        comparison['distillation_quality'] = quality
    else:
        comparison['average_forget'] = None
        comparison['distillation_quality'] = "Unknown"
    
    averaged_results['comparison'] = comparison
    
    return averaged_results


def print_summary(averaged_results):
    """Print a summary of the averaged results"""
    print("\n" + "="*80)
    print("📊 AVERAGED EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n📁 Dataset: {averaged_results['dataset']}")
    print(f"📈 Number of runs: {averaged_results['num_runs']}")
    print(f"⏰ Timestamp: {averaged_results['timestamp']}")
    
    if 'comparison' in averaged_results:
        comp = averaged_results['comparison']
        print(f"\n🎯 MODEL COMPARISON")
        print(f"   Parameter Reduction: {comp['parameter_reduction']*100:.1f}%")
        
        # Print Weighted Average Forget (AF) - Key distillation quality metric
        if comp.get('average_forget') is not None:
            print(f"\n📉 WEIGHTED AVERAGE FORGET (AF): {comp['average_forget']:.4f} ({comp['average_forget_percentage']:.2f}%)")
            print(f"🎯 DISTILLATION QUALITY: {comp['distillation_quality']}")
            print(f"   Weighting: {comp.get('weighting_scheme', 'Equal weights')}")
            print(f"   (Lower AF = Better knowledge retention, focus on primary tasks)")
        
        print(f"\n   Node Classification Retention:")
        for key, value in comp['node_classification'].items():
            if 'retention' in key:
                print(f"      {key}: {value*100:.2f}%")
        
        print(f"\n   Link Prediction Retention:")
        for key, value in comp['link_prediction'].items():
            if 'retention' in key:
                print(f"      {key}: {value*100:.2f}%")
        
        # Node clustering removed - not a primary metric for graph KD evaluation
    
    # Highlight key metrics
    if 'teacher' in averaged_results['models'] and 'student' in averaged_results['models']:
        teacher = averaged_results['models']['teacher']
        student = averaged_results['models']['student']
        
        print(f"\n🎓 TEACHER vs STUDENT")
        print(f"   {'Metric':<25} {'Teacher':<15} {'Student':<15} {'Retention':<10}")
        print(f"   {'-'*65}")
        
        # Node classification
        t_acc = teacher['node_classification']['accuracy']
        s_acc = student['node_classification']['accuracy']
        s_acc_std = student['node_classification']['accuracy_std']
        retention = s_acc / t_acc * 100
        print(f"   {'Node Classification':<25} {t_acc:.4f}         {s_acc:.4f}±{s_acc_std:.4f}   {retention:.1f}%")
        
        # Link prediction AUC
        t_auc = teacher['link_prediction']['auc']
        s_auc = student['link_prediction']['auc']
        s_auc_std = student['link_prediction']['auc_std']
        retention = s_auc / t_auc * 100
        print(f"   {'Link Prediction AUC':<25} {t_auc:.4f}         {s_auc:.4f}±{s_auc_std:.4f}   {retention:.1f}%")
        
        # Link prediction AP
        t_ap = teacher['link_prediction']['ap']
        s_ap = student['link_prediction']['ap']
        s_ap_std = student['link_prediction']['ap_std']
        retention = s_ap / t_ap * 100
        print(f"   {'Link Prediction AP':<25} {t_ap:.4f}         {s_ap:.4f}±{s_ap_std:.4f}   {retention:.1f}%")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    print("🔄 AVERAGING MULTIPLE EVALUATION RESULTS")
    print("="*80)
    
    # Load all JSON files
    data_list, file_names = load_json_files()
    
    if len(data_list) < 2:
        print(f"⚠️ Warning: Only {len(data_list)} file(s) found. Need at least 2 for averaging.")
        if len(data_list) == 1:
            print("   Copying single file as averaged result...")
    
    # Create averaged results
    averaged_results = create_averaged_results(data_list)
    
    # Add source file names
    averaged_results['source_files'] = [Path(f).name for f in file_names]
    
    # Compute comparison metrics
    averaged_results = compute_comparison_metrics(averaged_results)
    
    # Save to file
    output_file = f"comprehensive_eval_averaged_{averaged_results['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(averaged_results, f, indent=2)
    
    print(f"\n✅ Averaged results saved to: {output_file}")
    
    # Print summary
    print_summary(averaged_results)


if __name__ == "__main__":
    main()
