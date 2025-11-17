"""
Beautiful Visualization for Model Comparison Results
Creates publication-quality figures suitable for presentation slides
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Load data
# Load results JSON from a configurable relative path
default_json = os.path.join(os.path.dirname(__file__), 'main_augmentation_123456.json')
json_path = os.environ.get('LCOGNN_RESULTS_JSON', default_json)
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract model data
teacher = data['models']['teacher']
middle_teacher = data['models']['middle_teacher']
student = data['models']['student']
comparison = data['comparison']

# Color scheme
colors = {
    'teacher': '#FF6B6B',      # Coral Red
    'middle_teacher': '#4ECDC4', # Turquoise
    'student': '#95E1D3',       # Mint Green
    'accent': '#FFA07A',        # Light Salmon
    'dark': '#2C3E50',          # Dark Blue-Gray
    'light': '#ECF0F1'          # Light Gray
}

# Create comprehensive visualization - 2 rows x 3 columns = 6 plots
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25, 
              left=0.08, right=0.95, top=0.90, bottom=0.08)

# ============================================================================
# 1. MODEL ARCHITECTURE COMPARISON (Top Left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

models = ['Teacher', 'Middle\nTeacher', 'Student']
params = [
    teacher['parameters']['total_parameters'] / 1e6,
    middle_teacher['parameters']['total_parameters'] / 1e6,
    student['parameters']['total_parameters'] / 1e6
]
dims = [teacher['embedding_dim'], middle_teacher['embedding_dim'], student['embedding_dim']]

x_pos = np.arange(len(models))
bars = ax1.bar(x_pos, params, color=[colors['teacher'], colors['middle_teacher'], colors['student']],
               edgecolor='white', linewidth=2, alpha=0.85)

# Add value labels on bars
for i, (bar, param, dim) in enumerate(zip(bars, params, dims)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{param:.2f}M\n(dim={dim})',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax1.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(params) * 1.25)

# ============================================================================
# 2. NODE CLASSIFICATION PERFORMANCE (Top Middle)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

metrics = ['Accuracy', 'Macro F1', 'Micro F1']
teacher_nc = [teacher['node_classification']['accuracy'],
              teacher['node_classification']['macro_f1'],
              teacher['node_classification']['micro_f1']]
middle_nc = [middle_teacher['node_classification']['accuracy'],
             middle_teacher['node_classification']['macro_f1'],
             middle_teacher['node_classification']['micro_f1']]
student_nc = [student['node_classification']['accuracy'],
              student['node_classification']['macro_f1'],
              student['node_classification']['micro_f1']]

x = np.arange(len(metrics))
width = 0.25

bars1 = ax2.bar(x - width, teacher_nc, width, label='Teacher', 
                color=colors['teacher'], alpha=0.85, edgecolor='white', linewidth=1.5)
bars2 = ax2.bar(x, middle_nc, width, label='Middle Teacher',
                color=colors['middle_teacher'], alpha=0.85, edgecolor='white', linewidth=1.5)
bars3 = ax2.bar(x + width, student_nc, width, label='Student',
                color=colors['student'], alpha=0.85, edgecolor='white', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Node Classification Performance', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10)
ax2.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
ax2.set_ylim(0.82, 0.92)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# ============================================================================
# 3. LINK PREDICTION PERFORMANCE (Top Right) - WITHOUT Hits@K
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

lp_metrics = ['AUC', 'AP']
teacher_lp = [teacher['link_prediction']['auc'],
              teacher['link_prediction']['ap']]
middle_lp = [middle_teacher['link_prediction']['auc'],
             middle_teacher['link_prediction']['ap']]
student_lp = [student['link_prediction']['auc'],
              student['link_prediction']['ap']]

x = np.arange(len(lp_metrics))
width = 0.25

bars1 = ax3.bar(x - width, teacher_lp, width, label='Teacher',
                color=colors['teacher'], alpha=0.85, edgecolor='white', linewidth=1.5)
bars2 = ax3.bar(x, middle_lp, width, label='Middle Teacher',
                color=colors['middle_teacher'], alpha=0.85, edgecolor='white', linewidth=1.5)
bars3 = ax3.bar(x + width, student_lp, width, label='Student',
                color=colors['student'], alpha=0.85, edgecolor='white', linewidth=1.5)

ax3.set_ylabel('Score (AUC/AP: 0-1, Hits: %)', fontsize=12, fontweight='bold')
ax3.set_title('Link Prediction Performance', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(lp_metrics, fontsize=9, rotation=15, ha='right')
ax3.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# ============================================================================
# 4. RETENTION & FORGETTING ANALYSIS (Bottom Left)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 0])

retention_metrics = ['Accuracy\nRetention', 'Macro F1\nRetention', 'Micro F1\nRetention',
                     'AUC\nRetention', 'AP\nRetention']
retention_values = [
    comparison['node_classification']['accuracy_retention'],
    comparison['node_classification']['macro_f1_retention'],
    comparison['node_classification']['micro_f1_retention'],
    comparison['link_prediction']['auc_retention'],
    comparison['link_prediction']['ap_retention']
]

colors_retention = ['#2ECC71' if v >= 1.0 else '#E74C3C' for v in retention_values]
bars = ax4.barh(retention_metrics, retention_values, color=colors_retention, 
                alpha=0.85, edgecolor='white', linewidth=2)

# Add reference line at 100%
ax4.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='100% Retention')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, retention_values)):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
             f'{val:.2%}',
             ha='left', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax4.set_xlabel('Retention Ratio (Student/Teacher)', fontsize=12, fontweight='bold')
ax4.set_title('Knowledge Retention Analysis', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlim(0.95, max(retention_values) * 1.05)
ax4.legend(loc='lower right', framealpha=0.95)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# ============================================================================
# 5. PERFORMANCE WITH ERROR BARS (Bottom Center)
# ============================================================================
ax5 = fig.add_subplot(gs[1, 1])

# Student performance with standard deviations
student_metrics_plot = ['Accuracy', 'Macro F1', 'Micro F1', 'AUC', 'AP']
student_means = [
    student['node_classification']['accuracy'],
    student['node_classification']['macro_f1'],
    student['node_classification']['micro_f1'],
    student['link_prediction']['auc'],
    student['link_prediction']['ap']
]
student_stds = [
    student['node_classification']['accuracy_std'],
    student['node_classification']['macro_f1_std'],
    student['node_classification']['micro_f1_std'],
    student['link_prediction']['auc_std'],
    student['link_prediction']['ap_std']
]

x = np.arange(len(student_metrics_plot))
bars = ax5.bar(x, student_means, yerr=student_stds, capsize=5,
               color=colors['student'], alpha=0.85, edgecolor='white', 
               linewidth=2, error_kw={'linewidth': 2, 'ecolor': colors['dark']})

# Add value labels
for i, (bar, mean, std) in enumerate(zip(bars, student_means, student_stds)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + std,
             f'{mean:.4f}\n±{std:.4f}',
             ha='center', va='bottom', fontsize=8, fontweight='bold')

ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_title('Student Model Performance with Uncertainty', fontsize=14, fontweight='bold', pad=15)
ax5.set_xticks(x)
ax5.set_xticklabels(student_metrics_plot, fontsize=10)
ax5.set_ylim(0.75, 0.95)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# ============================================================================
# 6. DISTILLATION QUALITY SCORECARD (Bottom Right)
# ============================================================================
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

# Create a beautiful scorecard with key metrics
scorecard_data = [
    ('Parameter Reduction', f"{comparison['parameter_reduction']*100:.1f}%", colors['teacher']),
    ('Avg Performance Gain', f"{(1-comparison['average_forget'])*100:.2f}%", colors['student']),
    ('Quality Rating', comparison['distillation_quality'], colors['middle_teacher']),
    ('Dataset', data['dataset'].upper(), colors['accent']),
    ('Number of Runs', str(data['num_runs']), colors['dark'])
]

y_start = 0.88
for i, (label, value, color) in enumerate(scorecard_data):
    y_pos = y_start - i * 0.16
    
    # Draw fancy box
    fancy_box = FancyBboxPatch((0.05, y_pos - 0.07), 0.9, 0.13,
                               boxstyle="round,pad=0.01", 
                               edgecolor=color, facecolor=color,
                               alpha=0.2, linewidth=2.5,
                               transform=ax6.transAxes)
    ax6.add_patch(fancy_box)
    
    # Add text
    ax6.text(0.1, y_pos, label, transform=ax6.transAxes,
             fontsize=13, fontweight='bold', va='center')
    ax6.text(0.9, y_pos, value, transform=ax6.transAxes,
             fontsize=14, fontweight='bold', va='center', ha='right',
             color=color)

ax6.set_title('Distillation Quality Summary', fontsize=14, fontweight='bold',
              loc='center', pad=20)

# ============================================================================
# MAIN TITLE
# ============================================================================
fig.suptitle('Knowledge Distillation Performance Analysis: L-CoGNN Framework',
             fontsize=18, fontweight='bold', y=0.98)

# Save the figure
results_dir = os.environ.get('LCOGNN_RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'comprehensive_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Comprehensive visualization saved to: {output_path}")

# Also save as PDF for high-quality presentation
output_path_pdf = os.path.join(results_dir, 'comprehensive_visualization.pdf')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
print(f"PDF version saved to: {output_path_pdf}")

plt.show()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"Files created:")
print(f"  1. PNG: {output_path}")
print(f"  2. PDF: {output_path_pdf}")
print("\nThe visualization includes (2 rows × 3 columns = 6 plots):")
print("  Row 1:")
print("    ✓ Model architecture comparison")
print("    ✓ Node classification performance")
print("    ✓ Link prediction performance (AUC & AP only)")
print("  Row 2:")
print("    ✓ Knowledge retention analysis")
print("    ✓ Student performance with uncertainty")
print("    ✓ Distillation quality summary")
print("="*80)
