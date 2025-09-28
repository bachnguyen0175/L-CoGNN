"""
Evaluation and Comparison Script for KD-HGRL
"""

import os
import torch
import numpy as np
import time
import sys
import argparse

# Add utils to path
sys.path.append('./utils')

from models.kd_heco import MyHeCo, StudentMyHeCo, count_parameters
from utils.load_data import load_data
from utils.evaluate import (
    evaluate_node_classification,
    evaluate_all_downstream_tasks,
)


class ModelEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Set dataset-specific parameters
        if args.dataset == "acm":
            args.ratio = [60, 40]  # 60% train, 20% val, 20% test
            args.type_num = [4019, 7167, 60]  # [paper, author, subject]
            args.nei_num = 2
        elif args.dataset == "dblp":
            args.ratio = [60, 40]   # 60% train, 20% val, 20% test
            args.type_num = [4057, 14328, 7723, 20]  # [paper, author, conference, term]
            args.nei_num = 3
        elif args.dataset == "aminer":
            args.ratio = [60, 40]   # 60% train, 20% val, 20% test
            args.type_num = [6564, 13329, 35890]  # [paper, author, reference]
            args.nei_num = 2
        elif args.dataset == "freebase":
            args.ratio = [60, 40]   # 60% train, 20% val, 20% test
            args.type_num = [3492, 2502, 33401, 4459]  # [movie, director, actor, writer]
            args.nei_num = 3

        # Load data
        print(f"Loading {args.dataset} dataset...")
        self.nei_index, self.feats, self.mps, self.pos, self.label, self.idx_train, self.idx_val, self.idx_test = load_data(args.dataset, args.ratio, args.type_num)
        
        # Dataset specific parameters
        self.nb_classes = self.label.shape[-1]
        self.feats_dim_list = [feat.shape[1] for feat in self.feats]
        self.P = len(self.mps)
        
        # Move data to device
        self.move_data_to_device()
        
    def move_data_to_device(self):
        """Move all data to the specified device"""
        if torch.cuda.is_available():
            self.feats = [feat.to(self.device) for feat in self.feats]
            self.mps = [mp.to(self.device) for mp in self.mps]
            self.pos = self.pos.to(self.device)
            self.label = self.label.to(self.device)
            self.idx_train = [idx.to(self.device) for idx in self.idx_train]
            self.idx_val = [idx.to(self.device) for idx in self.idx_val]
            self.idx_test = [idx.to(self.device) for idx in self.idx_test]
    
    def load_teacher_model(self, model_path):
        """Load teacher model"""
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
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def load_student_model(self, model_path):
        """Load student model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        compression_ratio = checkpoint.get('compression_ratio', 0.5)
        
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
            compression_ratio=compression_ratio
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, compression_ratio
    
    def evaluate_model(self, model, model_name="Model"):
        """Evaluate a model on all train/val/test splits"""
        print(f"\nEvaluating {model_name}...")
        
        # Get embeddings
        start_time = time.time()
        with torch.no_grad():
            embeds = model.get_embeds(self.feats, self.mps)
        embed_time = time.time() - start_time
        
        results = []
        for i in range(len(self.idx_train)):
            accuracy, macro_f1, micro_f1 = evaluate_node_classification(
                embeds, self.idx_train[i], self.idx_val[i], self.idx_test[i],
                self.label, self.nb_classes, self.device,
                self.args.eva_lr, self.args.eva_wd
            )
            results.append((accuracy, macro_f1, micro_f1))
        
        # Calculate averages
        avg_accuracy = np.mean([r[0] for r in results])
        avg_macro_f1 = np.mean([r[1] for r in results])
        avg_micro_f1 = np.mean([r[2] for r in results])
        
        return {
            'accuracy': avg_accuracy,
            'macro_f1': avg_macro_f1,
            'micro_f1': avg_micro_f1,
            'embed_time': embed_time,
            'results_per_split': results
        }
    
    def benchmark_inference_speed(self, model, num_runs=10):
        """Benchmark inference speed"""
        model.eval()
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.get_embeds(self.feats, self.mps)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model.get_embeds(self.feats, self.mps)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def memory_usage(self, model):
        """Estimate memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model.get_embeds(self.feats, self.mps)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            return peak_memory
        else:
            return 0
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation comparing teacher and student models"""
        print("="*80)
        print("COMPREHENSIVE EVALUATION - TEACHER vs STUDENT COMPARISON")
        print("="*80)
        
        results = {}
        
        # Evaluate teacher model
        if self.args.teacher_model_path and os.path.exists(self.args.teacher_model_path):
            print(f"\nLoading teacher model: {self.args.teacher_model_path}")
            teacher_model = self.load_teacher_model(self.args.teacher_model_path)
            teacher_params = count_parameters(teacher_model)
            
            # Evaluate performance
            teacher_results = self.evaluate_model(teacher_model, "Teacher")
            
            # Benchmark speed and memory
            teacher_speed = self.benchmark_inference_speed(teacher_model)
            teacher_memory = self.memory_usage(teacher_model)
            
            results['teacher'] = {
                'model': teacher_model,
                'parameters': teacher_params,
                'performance': teacher_results,
                'inference_speed': teacher_speed,
                'memory_usage': teacher_memory
            }
            
        else:
            print(f"Teacher model not found: {self.args.teacher_model_path}")
            return
        
        # Evaluate student model
        if self.args.student_model_path and os.path.exists(self.args.student_model_path):
            print(f"\nLoading student model: {self.args.student_model_path}")
            student_model, compression_ratio = self.load_student_model(self.args.student_model_path)
            student_params = count_parameters(student_model)
            
            # Evaluate performance
            student_results = self.evaluate_model(student_model, "Student")
            
            # Benchmark speed and memory
            student_speed = self.benchmark_inference_speed(student_model)
            student_memory = self.memory_usage(student_model)
            
            results['student'] = {
                'model': student_model,
                'parameters': student_params,
                'performance': student_results,
                'inference_speed': student_speed,
                'memory_usage': student_memory,
                'compression_ratio': compression_ratio
            }
            
        else:
            print(f"Student model not found: {self.args.student_model_path}")
            return
        
        # Print comprehensive comparison
        self.print_comparison_report(results)
        
        return results
    
    def print_comparison_report(self, results):
        """Print detailed comparison report"""
        print("\n" + "="*80)
        print("DETAILED COMPARISON REPORT")
        print("="*80)
        
        teacher = results['teacher']
        student = results['student']
        
        # Model statistics
        print(f"\n📊 MODEL STATISTICS")
        print("-" * 40)
        print(f"{'Metric':<25} {'Teacher':<15} {'Student':<15} {'Ratio':<10}")
        print("-" * 65)
        print(f"{'Parameters':<25} {teacher['parameters']:<15,} {student['parameters']:<15,} {student['parameters']/teacher['parameters']:<10.3f}")
        print(f"{'Compression Ratio':<25} {'1.000':<15} {student['compression_ratio']:<15.3f} {student['compression_ratio']:<10.3f}")
        print(f"{'Size Reduction':<25} {'0.0%':<15} {(1-student['compression_ratio'])*100:<14.1f}% {'-':<10}")
        
        # Performance comparison
        print(f"\n📈 PERFORMANCE COMPARISON")
        print("-" * 40)
        print(f"{'Metric':<25} {'Teacher':<15} {'Student':<15} {'Retention':<10}")
        print("-" * 65)
        
        t_perf = teacher['performance']
        s_perf = student['performance']
        
        print(f"{'Accuracy':<25} {t_perf['accuracy']:<15.4f} {s_perf['accuracy']:<15.4f} {s_perf['accuracy']/t_perf['accuracy']:<10.3f}")
        print(f"{'Macro F1':<25} {t_perf['macro_f1']:<15.4f} {s_perf['macro_f1']:<15.4f} {s_perf['macro_f1']/t_perf['macro_f1']:<10.3f}")
        print(f"{'Micro F1':<25} {t_perf['micro_f1']:<15.4f} {s_perf['micro_f1']:<15.4f} {s_perf['micro_f1']/t_perf['micro_f1']:<10.3f}")
        
        # Speed and efficiency
        print(f"\n⚡ EFFICIENCY COMPARISON")
        print("-" * 40)
        print(f"{'Metric':<25} {'Teacher':<15} {'Student':<15} {'Speedup':<10}")
        print("-" * 65)
        print(f"{'Inference Time (s)':<25} {teacher['inference_speed']:<15.4f} {student['inference_speed']:<15.4f} {teacher['inference_speed']/student['inference_speed']:<10.2f}x")
        
        if teacher['memory_usage'] > 0 and student['memory_usage'] > 0:
            print(f"{'Memory Usage (MB)':<25} {teacher['memory_usage']:<15.1f} {student['memory_usage']:<15.1f} {teacher['memory_usage']/student['memory_usage']:<10.2f}x")
        
        # Summary
        print(f"\n📋 SUMMARY")
        print("-" * 40)
        param_reduction = (1 - student['compression_ratio']) * 100
        perf_retention = (s_perf['accuracy'] / t_perf['accuracy']) * 100
        speed_improvement = teacher['inference_speed'] / student['inference_speed']
        
        print(f"✅ Parameter Reduction: {param_reduction:.1f}%")
        print(f"✅ Performance Retention: {perf_retention:.1f}%")
        print(f"✅ Speed Improvement: {speed_improvement:.1f}x")

        if perf_retention > 90:
            print("🎉 Excellent knowledge distillation! Student retains >90% of teacher performance.")
        elif perf_retention > 80:
            print("👍 Good knowledge distillation! Student retains >80% of teacher performance.")
        else:
            print("⚠️  Consider adjusting distillation parameters to improve student performance.")

    def run_all_downstream_tasks_evaluation(self):
        """Run comprehensive evaluation on all three downstream tasks"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DOWNSTREAM TASKS EVALUATION")
        print("="*80)

        results = {}

        # Extract edges for link prediction
        edges = None
        num_nodes = None
        try:
            if hasattr(self.pos, 'coalesce'):
                pos_coalesced = self.pos.coalesce()
                indices = pos_coalesced.indices().t()
                edges = indices.cpu().numpy()
            else:
                nonzero = torch.nonzero(self.pos, as_tuple=False)
                edges = nonzero.cpu().numpy()
            num_nodes = self.feats[0].shape[0]
            print(f"Extracted {len(edges)} edges for link prediction")
        except Exception as e:
            print(f"Warning: Could not extract edges: {e}")

        # Evaluate teacher model
        if self.args.teacher_model_path and os.path.exists(self.args.teacher_model_path):
            print(f"\n🔍 Evaluating Teacher Model on All Tasks...")
            teacher_model = self.load_teacher_model(self.args.teacher_model_path)

            # Get embeddings
            with torch.no_grad():
                teacher_embeds = teacher_model.get_embeds(self.feats, self.mps)

            teacher_results = evaluate_all_downstream_tasks(
                teacher_embeds, self.label, edges, num_nodes, self.device
            )
            results['teacher'] = teacher_results

        # Evaluate student model
        if self.args.student_model_path and os.path.exists(self.args.student_model_path):
            print(f"\n🔍 Evaluating Student Model on All Tasks...")
            student_model, compression_ratio = self.load_student_model(self.args.student_model_path)

            # Get embeddings
            with torch.no_grad():
                student_embeds = student_model.get_embeds(self.feats, self.mps)

            student_results = evaluate_all_downstream_tasks(
                student_embeds, self.label, edges, num_nodes, self.device
            )
            results['student'] = student_results

        # Print comprehensive comparison
        if 'teacher' in results and 'student' in results:
            self.print_all_tasks_comparison(results['teacher'], results['student'])

        return results

    def print_all_tasks_comparison(self, teacher_results, student_results):
        """Print detailed comparison across all three tasks"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print("="*80)

        # Node Classification Comparison
        if 'node_classification' in teacher_results and 'node_classification' in student_results:
            t_nc = teacher_results['node_classification']
            s_nc = student_results['node_classification']

            if 'error' not in t_nc and 'error' not in s_nc:
                print(f"\n📊 NODE CLASSIFICATION")
                print("-" * 40)
                print(f"{'Metric':<15} {'Teacher':<12} {'Student':<12} {'Retention':<12}")
                print("-" * 52)
                print(f"{'Accuracy':<15} {t_nc['accuracy']:<12.4f} {s_nc['accuracy']:<12.4f} {s_nc['accuracy']/t_nc['accuracy']:<12.3f}")
                print(f"{'Macro-F1':<15} {t_nc['macro_f1']:<12.4f} {s_nc['macro_f1']:<12.4f} {s_nc['macro_f1']/t_nc['macro_f1']:<12.3f}")
                print(f"{'Micro-F1':<15} {t_nc['micro_f1']:<12.4f} {s_nc['micro_f1']:<12.4f} {s_nc['micro_f1']/t_nc['micro_f1']:<12.3f}")

        # Link Prediction Comparison
        if 'link_prediction' in teacher_results and 'link_prediction' in student_results:
            t_lp = teacher_results['link_prediction']
            s_lp = student_results['link_prediction']

            if 'error' not in t_lp and 'error' not in s_lp and 'skipped' not in t_lp:
                print(f"\n🔗 LINK PREDICTION")
                print("-" * 40)
                print(f"{'Metric':<15} {'Teacher':<12} {'Student':<12} {'Retention':<12}")
                print("-" * 52)
                print(f"{'AUC':<15} {t_lp['auc']:<12.4f} {s_lp['auc']:<12.4f} {s_lp['auc']/t_lp['auc']:<12.3f}")
                print(f"{'AP':<15} {t_lp['ap']:<12.4f} {s_lp['ap']:<12.4f} {s_lp['ap']/t_lp['ap']:<12.3f}")
                print(f"{'Hits@10':<15} {t_lp['hits_at_10']:<12.4f} {s_lp['hits_at_10']:<12.4f} {s_lp['hits_at_10']/max(t_lp['hits_at_10'], 1e-6):<12.3f}")

        # Node Clustering Comparison
        if 'node_clustering' in teacher_results and 'node_clustering' in student_results:
            t_cl = teacher_results['node_clustering']
            s_cl = student_results['node_clustering']

            if 'error' not in t_cl and 'error' not in s_cl:
                print(f"\n🎯 NODE CLUSTERING")
                print("-" * 40)
                print(f"{'Metric':<15} {'Teacher':<12} {'Student':<12} {'Retention':<12}")
                print("-" * 52)
                print(f"{'NMI':<15} {t_cl['nmi']:<12.4f} {s_cl['nmi']:<12.4f} {s_cl['nmi']/t_cl['nmi']:<12.3f}")
                print(f"{'ARI':<15} {t_cl['ari']:<12.4f} {s_cl['ari']:<12.4f} {s_cl['ari']/t_cl['ari']:<12.3f}")
                print(f"{'Accuracy':<15} {t_cl['accuracy']:<12.4f} {s_cl['accuracy']:<12.4f} {s_cl['accuracy']/t_cl['accuracy']:<12.3f}")
                print(f"{'Modularity':<15} {t_cl['modularity']:<12.4f} {s_cl['modularity']:<12.4f} {s_cl['modularity']/max(t_cl['modularity'], 1e-6):<12.3f}")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare teacher vs student models')
    parser.add_argument('--dataset', type=str, default='acm', help='Dataset name')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Path to teacher model')
    parser.add_argument('--student_model_path', type=str, required=True, help='Path to student model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='Feature dropout')
    parser.add_argument('--attn_drop', type=float, default=0.5, help='Attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='Temperature parameter')
    parser.add_argument('--lam', type=float, default=0.5, help='Lambda parameter')
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1], help='Sample rates')
    parser.add_argument('--eva_lr', type=float, default=0.05, help='Evaluation learning rate')
    parser.add_argument('--eva_wd', type=float, default=0, help='Evaluation weight decay')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    
    args = parser.parse_args()
    
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
    
    # Run evaluation
    evaluator = ModelEvaluator(args)
    results = evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()