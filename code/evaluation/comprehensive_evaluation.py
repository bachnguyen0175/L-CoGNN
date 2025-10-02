"""
Comprehensive Evaluation Script for All Three Downstream Tasks
===============================================================

This script evaluates teacher and student models on:
1. Node Classification (Accuracy, Macro-F1, Micro-F1, AUC)
2. Link Prediction (AUC, AP, Hits@K)
3. Node Clustering (NMI, ARI, Accuracy, Modularity)

Usage:
    python comprehensive_evaluation.py --dataset acm --teacher_model_path teacher.pkl --student_path student.pkl
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import MyHeCo, StudentMyHeCo, PruningExpertTeacher
from models import kd_params
from utils.load_data import load_data
from utils.evaluate import (
    evaluate_node_classification,
    evaluate_link_prediction,
    evaluate_node_clustering,
    generate_negative_edges,
    split_edges_for_link_prediction
)


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all three downstream tasks"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        # Load data
        print(f"Loading {args.dataset} dataset...")
        self.load_dataset()

        # Initialize results storage
        self.results = {
            'dataset': args.dataset,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {}
        }

    def load_dataset(self):
        """Load dataset with all necessary components"""
        # Set dataset-specific parameters
        if self.args.dataset == "acm":
            self.args.type_num = [4019, 7167, 60]  # [paper, author, subject]
            self.args.nei_num = 2
        elif self.args.dataset == "dblp":
            self.args.type_num = [4057, 14328, 7723, 20]  # [paper, author, conference, term]
            self.args.nei_num = 3
        elif self.args.dataset == "aminer":
            self.args.type_num = [6564, 13329, 35890]  # [paper, author, reference]
            self.args.nei_num = 2
        elif self.args.dataset == "freebase":
            self.args.type_num = [3492, 2502, 33401, 4459]  # [movie, director, actor, writer]
            self.args.nei_num = 3

        # Set default ratio if not provided
        if not hasattr(self.args, 'ratio'):
            self.args.ratio = ["80_10_10"]

        data = load_data(self.args.dataset, self.args.ratio, self.args.type_num)

        if len(data) == 8:
            # Standard format: nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test
            self.nei_index, self.feats, self.mps, self.pos, self.label, self.idx_train, self.idx_val, self.idx_test = data
        else:
            raise ValueError(f"Unexpected data format. Expected 8 elements, got {len(data)}")

        # Dataset specific parameters
        self.nb_classes = self.label.shape[-1]
        self.feats_dim_list = [feat.shape[1] for feat in self.feats]
        self.P = len(self.mps)

        # Move data to device
        self.move_data_to_device()

        # Extract edges for link prediction
        self.extract_edges_from_pos()

        print(f"Dataset loaded: {len(self.feats[0])} nodes, {len(self.edges)} edges, {self.nb_classes} classes")

    def move_data_to_device(self):
        """Move all data to the specified device"""
        if torch.cuda.is_available() and self.args.gpu >= 0:
            self.feats = [feat.to(self.device) for feat in self.feats]
            self.mps = [mp.to(self.device) for mp in self.mps]
            self.pos = self.pos.to(self.device)
            self.label = self.label.to(self.device)

            # Handle different index formats
            if isinstance(self.idx_train, list):
                self.idx_train = [idx.to(self.device) for idx in self.idx_train]
                self.idx_val = [idx.to(self.device) for idx in self.idx_val]
                self.idx_test = [idx.to(self.device) for idx in self.idx_test]
            else:
                self.idx_train = self.idx_train.to(self.device)
                self.idx_val = self.idx_val.to(self.device)
                self.idx_test = self.idx_test.to(self.device)

    def extract_edges_from_pos(self):
        """Extract edge list from pos tensor for link prediction"""
        try:
            if hasattr(self.pos, 'coalesce'):
                # Sparse tensor
                pos_coalesced = self.pos.coalesce()
                indices = pos_coalesced.indices().t()  # [num_edges, 2]
                self.edges = indices.cpu().numpy()
            else:
                # Dense tensor - find non-zero entries
                nonzero = torch.nonzero(self.pos, as_tuple=False)
                self.edges = nonzero.cpu().numpy()

            self.num_nodes = self.feats[0].shape[0]
            print(f"Extracted {len(self.edges)} edges for link prediction")

        except Exception as e:
            print(f"Warning: Could not extract edges from pos tensor: {e}")
            self.edges = None
            self.num_nodes = None

    def load_model(self, model_path, model_type='teacher'):
        """Load a model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if model_type == 'teacher':
            model = MyHeCo(
                hidden_dim=self.args.hidden_dim,
                feats_dim_list=self.feats_dim_list,
                feat_drop=self.args.feat_drop,
                attn_drop=self.args.attn_drop,
                P=self.P,
                sample_rate=self.args.sample_rate,
                nei_num=self.args.nei_num,
                tau=self.args.tau,
                lam=self.args.lam
            ).to(self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        elif model_type == 'student':
            compression_ratio = checkpoint.get('compression_ratio', 0.5)

            # Check if this is an enhanced student model with guidance parameters
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            has_guidance = any('guidance' in key or 'fusion' in key for key in state_dict.keys())
            
            model = StudentMyHeCo(
                hidden_dim=self.args.hidden_dim,
                feats_dim_list=self.feats_dim_list,
                feat_drop=self.args.feat_drop,
                attn_drop=self.args.attn_drop,
                P=self.P,
                sample_rate=self.args.sample_rate,
                nei_num=self.args.nei_num,
                tau=self.args.tau,
                lam=self.args.lam,
                compression_ratio=compression_ratio,
                use_middle_teacher_guidance=has_guidance  # Enable guidance if model has guidance parameters
            ).to(self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        elif model_type == 'middle_teacher':
            compression_ratio = checkpoint.get('compression_ratio', 0.7)
            
            augmentation_config = {
                'use_node_masking': getattr(self.args, 'use_node_masking', True),
                'use_edge_augmentation': getattr(self.args, 'use_edge_augmentation', True),
                'mask_rate': getattr(self.args, 'mask_rate', 0.1),
                'remask_rate': getattr(self.args, 'remask_rate', 0.3),
                'edge_drop_rate': getattr(self.args, 'edge_drop_rate', 0.1),
                'num_remasking': getattr(self.args, 'num_remasking', 2)
            }

            model = PruningExpertTeacher(
                feats_dim_list=self.feats_dim_list,
                hidden_dim=self.args.hidden_dim,
                attn_drop=self.args.attn_drop,
                feat_drop=self.args.feat_drop,
                P=self.P,
                sample_rate=self.args.sample_rate,
                nei_num=self.args.nei_num,
                tau=self.args.tau,
                lam=self.args.lam,
                augmentation_config=augmentation_config
            ).to(self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        model.eval()
        return model

    def get_model_embeddings(self, model):
        """Extract embeddings from a model"""
        with torch.no_grad():
            if hasattr(model, 'get_embeds'):
                embeddings = model.get_embeds(self.feats, self.mps)
            else:
                # Get representations and combine them
                mp_repr, sc_repr = model.get_representations(self.feats, self.mps, self.nei_index)
                # Simple combination - you might want to modify this based on your model
                embeddings = (mp_repr + sc_repr) / 2

        return embeddings

    def evaluate_single_model(self, model, model_name, model_type='teacher'):
        """Evaluate a single model on all three tasks"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()} ({model_type})")
        print(f"{'='*60}")

        # Get embeddings
        embeddings = self.get_model_embeddings(model)
        print(f"Extracted embeddings: {embeddings.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_results = {
            'model_type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_dim': embeddings.shape[1]
        }

        # 1. Node Classification
        print("\n1. Node Classification...")
        try:
            # Use existing train/val/test splits
            if isinstance(self.idx_train, list):
                # Multiple splits - use the first one
                train_idx = self.idx_train[0]
                val_idx = self.idx_val[0]
                test_idx = self.idx_test[0]
            else:
                train_idx = self.idx_train
                val_idx = self.idx_val
                test_idx = self.idx_test

            accuracy, macro_f1, micro_f1 = evaluate_node_classification(
                embeddings, train_idx, val_idx, test_idx, self.label,
                self.nb_classes, self.device, self.args.eva_lr, self.args.eva_wd
            )

            model_results['node_classification'] = {
                'accuracy': float(accuracy),
                'macro_f1': float(macro_f1),
                'micro_f1': float(micro_f1)
            }

            print(f"   ✓ Accuracy: {accuracy:.4f}")
            print(f"   ✓ Macro-F1: {macro_f1:.4f}")
            print(f"   ✓ Micro-F1: {micro_f1:.4f}")

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            model_results['node_classification'] = {'error': str(e)}

        # 2. Link Prediction
        print("\n2. Link Prediction...")
        if self.edges is not None:
            try:
                # Split edges
                train_edges, val_edges, test_edges = split_edges_for_link_prediction(self.edges)
                test_neg_edges = generate_negative_edges(test_edges, self.num_nodes)

                auc_score, ap_score, hits_at_k = evaluate_link_prediction(
                    embeddings, test_edges, test_neg_edges, self.device
                )

                model_results['link_prediction'] = {
                    'auc': float(auc_score),
                    'ap': float(ap_score),
                    'hits_at_10': float(hits_at_k.get('hits_at_10', 0)),
                    'hits_at_50': float(hits_at_k.get('hits_at_50', 0)),
                    'hits_at_100': float(hits_at_k.get('hits_at_100', 0))
                }

                print(f"   ✓ AUC: {auc_score:.4f}")
                print(f"   ✓ AP: {ap_score:.4f}")
                print(f"   ✓ Hits@10: {hits_at_k.get('hits_at_10', 0):.4f}")
                print(f"   ✓ Hits@50: {hits_at_k.get('hits_at_50', 0):.4f}")
                print(f"   ✓ Hits@100: {hits_at_k.get('hits_at_100', 0):.4f}")

            except Exception as e:
                print(f"   ✗ Failed: {e}")
                model_results['link_prediction'] = {'error': str(e)}
        else:
            print("   ⚠ Skipped (no edges available)")
            model_results['link_prediction'] = {'skipped': 'no edges available'}


        return model_results

    def compare_models(self, teacher_results, student_results):
        """Generate comparison metrics between teacher and student"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON ANALYSIS")
        print(f"{'='*60}")

        comparison = {
            'parameter_reduction': 1 - (student_results['total_parameters'] / teacher_results['total_parameters']),
            'task_comparisons': {}
        }

        # Store forgetting rates for AF calculation
        # Focus on primary tasks: node_classification (50%) and link_prediction (50%)
        forgetting_rates = []
        task_weights = {
            'node_classification': 0.5,
            'link_prediction': 0.5
        }

        # Compare each task (node clustering removed - not primary for graph KD)
        for task in ['node_classification', 'link_prediction']:
            if task in teacher_results and task in student_results:
                if 'error' not in teacher_results[task] and 'error' not in student_results[task]:
                    task_comparison = {}

                    if task == 'node_classification':
                        for metric in ['accuracy', 'macro_f1', 'micro_f1']:
                            if metric in teacher_results[task] and metric in student_results[task]:
                                teacher_val = teacher_results[task][metric]
                                student_val = student_results[task][metric]
                                retention = student_val / teacher_val
                                task_comparison[f'{metric}_retention'] = retention
                                
                                # Calculate forgetting: max(0, teacher - student) / teacher
                                # Higher forgetting = worse distillation
                                forgetting = max(0, teacher_val - student_val) / teacher_val
                                task_comparison[f'{metric}_forgetting'] = forgetting
                                forgetting_rates.append((forgetting, task_weights[task]))

                    elif task == 'link_prediction':
                        for metric in ['auc', 'ap']:
                            if metric in teacher_results[task] and metric in student_results[task]:
                                teacher_val = teacher_results[task][metric]
                                student_val = student_results[task][metric]
                                retention = student_val / teacher_val
                                task_comparison[f'{metric}_retention'] = retention
                                
                                # Calculate forgetting
                                forgetting = max(0, teacher_val - student_val) / teacher_val
                                task_comparison[f'{metric}_forgetting'] = forgetting
                                forgetting_rates.append((forgetting, task_weights[task]))

                    comparison['task_comparisons'][task] = task_comparison

        # Calculate Weighted Average Forget (AF) - Lower is better!
        # Focus on primary tasks: 50% node classification + 50% link prediction
        # (Node clustering removed as it's not a primary metric for graph KD)
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

        # Print comparison summary
        print(f"\n📊 PARAMETER REDUCTION: {comparison['parameter_reduction']*100:.1f}%")
        
        if comparison['average_forget'] is not None:
            print(f"📉 WEIGHTED AVERAGE FORGET (AF): {comparison['average_forget']:.4f} ({comparison['average_forget_percentage']:.2f}%)")
            print(f"🎯 DISTILLATION QUALITY: {comparison['distillation_quality']}")
            print(f"   Weighting: {comparison['weighting_scheme']}")
            print(f"   (Lower AF = Better distillation, focus on primary tasks)")
        
        print(f"\n📈 PERFORMANCE RETENTION:")

        for task, task_comp in comparison['task_comparisons'].items():
            print(f"\n   {task.replace('_', ' ').title()}:")
            for metric, value in task_comp.items():
                if 'retention' in metric:
                    print(f"      {metric}: {value*100:.1f}%")

        return comparison

    def save_results(self, output_path=None):
        """Save comprehensive results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"comprehensive_eval_{self.args.dataset}_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")
        return output_path

    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        print(f"\n🚀 STARTING COMPREHENSIVE EVALUATION")
        print(f"Dataset: {self.args.dataset}")
        print(f"Device: {self.device}")

        # Evaluate Teacher Model
        if self.args.teacher_model_path and os.path.exists(self.args.teacher_model_path):
            print(f"\nLoading teacher model: {self.args.teacher_model_path}")
            teacher_model = self.load_model(self.args.teacher_model_path, 'teacher')
            teacher_results = self.evaluate_single_model(teacher_model, "Teacher", "teacher")
            self.results['models']['teacher'] = teacher_results
        else:
            print(f"⚠ Teacher model not found: {self.args.teacher_model_path}")
            return None

        # Evaluate Student Model
        if self.args.student_model_path and os.path.exists(self.args.student_model_path):
            print(f"\nLoading student model: {self.args.student_model_path}")
            student_model = self.load_model(self.args.student_model_path, 'student')
            student_results = self.evaluate_single_model(student_model, "Student", "student")
            self.results['models']['student'] = student_results

            # Generate comparison
            comparison = self.compare_models(teacher_results, student_results)
            self.results['comparison'] = comparison

        else:
            print(f"⚠ Student model not found: {self.args.student_model_path}")

        # Evaluate Middle Teacher (if provided)
        if hasattr(self.args, 'middle_teacher_path') and self.args.middle_teacher_path:
            if os.path.exists(self.args.middle_teacher_path):
                print(f"\nLoading middle teacher model: {self.args.middle_teacher_path}")
                middle_model = self.load_model(self.args.middle_teacher_path, 'middle_teacher')
                middle_results = self.evaluate_single_model(middle_model, "Middle Teacher", "middle_teacher")
                self.results['models']['middle_teacher'] = middle_results

        # Save results
        output_path = self.save_results()

        print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETED!")
        print(f"Results saved to: {output_path}")

        return self.results


def main():
    args = kd_params()

    # Set dataset-specific parameters
    if args.dataset == "acm":
        args.type_num = [4019, 7167, 60]
        args.nei_num = 2
    elif args.dataset == "dblp":
        args.type_num = [4057, 14328, 7723, 20]
        args.nei_num = 3
    elif args.dataset == "aminer":
        args.type_num = [6564, 13329, 35890]
        args.nei_num = 2
    elif args.dataset == "freebase":
        args.type_num = [3492, 2502, 33401, 4459]
        args.nei_num = 3

    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator(args)
    results = evaluator.run_comprehensive_evaluation()

    return results


if __name__ == "__main__":
    results = main()