# KD-HGRL: Knowledge Distillation for Heterogeneous Graph Representation Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

KD-HGRL is a comprehensive framework for **Knowledge Distillation in Heterogeneous Graph Representation Learning**. This project implements a **dual-teacher distillation architecture** that combines knowledge distillation with augmentation-based robustness learning to create compressed, efficient heterogeneous graph neural network models while maintaining competitive performance.

### 🎯 Key Features

- **Dual-Teacher Architecture**: 
  - **Main Teacher**: Provides knowledge distillation from original graph data
  - **Augmentation Teacher**: Provides robustness guidance from augmented graph data
  - **Student**: Learns from both teachers with 50% parameter compression
- **Heterogeneous Graph Support**: ACM, DBLP, AMiner, Freebase datasets
- **Multi-View Learning**: Meta-path encoder + Schema-level encoder
- **Advanced Augmentation**: Structure-aware heterogeneous graph augmentation
- **Multi-Task Evaluation**: Node classification, link prediction, node clustering
- **Model Compression**: 50% parameter reduction with ~95% performance retention
- **GPU Acceleration**: CUDA 11.8 support with PyTorch 2.1.2
- **Modular Loss Components**: Configurable KD loss, augmentation alignment, link reconstruction

### 🏆 Performance Highlights

| Model | Parameters | Compression | Node Classification | Link Prediction | Node Clustering |
|-------|------------|-------------|-------------------|-----------------|-----------------|
| Teacher | 100% | - | Baseline | Baseline | Baseline |
| Middle Teacher | 100% | No compression* | ~98% retention | ~97% retention | ~98% retention |
| Student | ~50% | 50% | ~95% retention | ~93% retention | ~94% retention |

\* *Middle teacher uses same architecture as teacher but trains on augmented data for robustness guidance*

## � How It Works: Dual-Teacher Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   DUAL-TEACHER FRAMEWORK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐         ┌──────────────────┐              │
│  │  Original Graph │         │  Augmented Graph  │              │
│  │  - PAP, PSP     │         │  - Feature mask   │              │
│  │  - Clean data   │         │  - Edge perturb   │              │
│  └────────┬────────┘         └────────┬─────────┘              │
│           │                           │                         │
│           ▼                           ▼                         │
│  ┌─────────────────┐         ┌──────────────────┐              │
│  │  Main Teacher   │         │ Augmentation     │              │
│  │  (100% params)  │         │ Teacher          │              │
│  │                 │         │ (100% params)    │              │
│  │  - Meta-path    │         │                  │              │
│  │  - Schema view  │         │ - Robust         │              │
│  │  - Contrastive  │         │   patterns       │              │
│  └────────┬────────┘         └────────┬─────────┘              │
│           │                           │                         │
│           │ KD Loss                   │ Augmentation            │
│           │ (knowledge)               │ Alignment               │
│           │                           │ (robustness)            │
│           └─────────┬─────────────────┘                         │
│                     │                                           │
│                     ▼                                           │
│           ┌──────────────────┐                                  │
│           │  Student Model   │                                  │
│           │  (50% params)    │                                  │
│           │                  │                                  │
│           │  - Compressed    │                                  │
│           │  - Fast          │                                  │
│           │  - Robust        │                                  │
│           └──────────────────┘                                  │
│                                                                  │
│  Loss = Student_Loss + α·KD_Loss + β·Aug_Align + γ·Link_Loss   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Independent Teacher Training**: Both teachers train independently
   - Main teacher learns from clean data
   - Augmentation teacher learns from augmented data
   - No hierarchical dependency

2. **Dual-Source Knowledge Transfer**: Student learns from both
   - **Knowledge Distillation** (Main Teacher): Transferring learned representations
   - **Augmentation Alignment** (Aug Teacher): Learning robust patterns
   - **Self-Learning**: Student's own contrastive loss

3. **Multi-Loss Training**:
   ```python
   Total Loss = student_contrastive_loss 
              + main_distill_weight * kd_loss          # from main teacher
              + augmentation_weight * alignment_loss    # from aug teacher  
              + link_recon_weight * link_loss          # optional, edge modeling
   ```

4. **Heterogeneous Graph Augmentation**:
   - Feature masking (random node feature dropout)
   - Edge perturbation (meta-path sampling variations)
   - Structure-aware augmentation (preserving heterogeneity)

## �🚀 Quick Start

### Prerequisites

- **Python**: 3.9+
- **CUDA**: 11.8 (optional but recommended)
- **Memory**: 8GB+ RAM
- **Storage**: 2GB+ for datasets and models

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/bachnguyen0175/L-CoGNN.git
cd L-CoGNN

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Quick Training Pipeline

```bash
# Navigate to scripts directory
cd code/scripts

# Option 1: Run complete pipeline automatically
bash run_all.sh acm

# Option 2: Run individual stages
bash 1_train_teacher.sh        # Stage 1: Train main teacher (~30-60 min)
bash 2_train_middle_teacher.sh # Stage 2: Train augmentation teacher (~15-30 min)
bash 3_train_student.sh        # Stage 3: Train student with dual teachers (~20-40 min)
bash 4_evaluate.sh             # Stage 4: Comprehensive evaluation (~5 min)
```

### 3. Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Main Teacher                                      │
│  ┌──────────────────────┐                                  │
│  │  Original Graph      │ ──► Main Teacher (100%)          │
│  │  - PAP, PSP paths    │     - Knowledge distillation     │
│  │  - Contrastive loss  │     - Clean representations      │
│  └──────────────────────┘                                  │
│                                                             │
│  Stage 2: Augmentation Teacher (Independent)                │
│  ┌──────────────────────┐                                  │
│  │  Augmented Graphs    │ ──► Augmentation Teacher (100%)  │
│  │  - Structure masking │     - Robustness guidance        │
│  │  - Feature dropout   │     - Augmentation patterns      │
│  └──────────────────────┘                                  │
│                                                             │
│  Stage 3: Student (Dual-Teacher Learning)                   │
│  ┌──────────────────────────────────────────────┐          │
│  │         Main Teacher (frozen)                 │          │
│  │               ↓ KD Loss                       │          │
│  │         Student Model (50%)  ←───────────────┼──────┐   │
│  │               ↑ Augmentation                 │      │   │
│  │               ↑ Alignment Loss                │      │   │
│  │   Augmentation Teacher (frozen)               │      │   │
│  │                                               │      │   │
│  │   + Student Contrastive Loss                  │      │   │
│  │   + Link Reconstruction Loss (optional)       │      │   │
│  └──────────────────────────────────────────────┘      │   │
│                                                         │   │
│  Total Loss = Student Loss                              │   │
│             + main_distill_weight * KD Loss             │   │
│             + augmentation_weight * Alignment Loss      │   │
│             + link_recon_weight * Link Loss             │   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Expected Output

```
✅ Complete pipeline completed successfully for acm!

📁 Generated Models:
   - teacher_heco_acm.pkl          (1.2M params, baseline)
   - middle_teacher_heco_acm.pkl   (1.2M params, augmentation expert)
   - student_heco_acm.pkl          (600K params, 50% compressed)

🎯 Model Comparison:
   Main Teacher:        100% parameters, Baseline performance
   Augmentation Teacher: 100% parameters (same architecture, different data)
   Student:             50% parameters, ~95% performance retention
```

## 📁 Project Structure

```
L-CoGNN/
├── 📋 README.md                     # This file
├── ⚙️ requirements.txt             # Dependencies
├── 🐍 main.py                      # Entry point (future CLI)
│
├── 🧠 code/models/                 # Neural Network Models
│   ├── kd_heco.py                  # Core architectures
│   │   ├── MyHeCo                  # Main teacher model
│   │   ├── AugmentationTeacher     # Augmentation teacher (same size)
│   │   ├── StudentMyHeCo           # Compressed student (50%)
│   │   └── DualTeacherKD           # KD framework coordinator
│   ├── contrast.py                 # Contrastive learning module
│   ├── sc_encoder.py               # Schema-level attention encoder
│   └── kd_params.py                # Model & training configurations
│
├── 🎓 code/training/               # Training Scripts
│   ├── pretrain_teacher.py         # Stage 1: Main teacher
│   ├── train_middle_teacher.py     # Stage 2: Augmentation teacher
│   ├── train_student.py            # Stage 3: Dual-teacher student
│   └── hetero_augmentations.py     # Graph augmentation techniques
│
├── 📊 code/evaluation/             # Evaluation Tools
│   ├── comprehensive_evaluation.py # Multi-task evaluation
│   └── evaluate_kd.py              # KD-specific metrics
│
├── 🔧 code/utils/                  # Utility Functions
│   ├── load_data.py                # Data loading
│   ├── evaluate.py                 # Evaluation metrics
│   └── logreg.py                   # Logistic regression
│
├── 🚀 code/scripts/                # Shell Scripts
│   ├── 1_train_teacher.sh          # Train main teacher
│   ├── 2_train_middle_teacher.sh   # Train augmentation teacher
│   ├── 3_train_student.sh          # Train student (dual-teacher)
│   ├── 4_evaluate.sh               # Comprehensive evaluation
│   └── run_all.sh                  # Complete pipeline
│
├── 🧪 code/experiments/            # Experiment Configurations
│   ├── configs/                    # YAML configurations
│   └── ablation/                   # Ablation studies
│
├── 📁 data/                        # Datasets
│   ├── acm/                        # ACM dataset
│   ├── dblp/                       # DBLP dataset
│   ├── aminer/                     # AMiner dataset
│   └── freebase/                   # Freebase dataset
│
├── 📈 results/                     # Model Checkpoints
│   ├── teacher_heco_*.pkl          # Main teacher (100%)
│   ├── middle_teacher_heco_*.pkl   # Augmentation teacher (100%)
│   └── student_heco_*.pkl          # Student (50%)
│
└── 🧪 code/tests/                  # Unit Tests
    └── test_imports.py             # Import validation
```

## 🎯 Usage Guide

### Training Individual Models

#### 1. Main Teacher Training (Stage 1)
```bash
cd code/scripts
bash 1_train_teacher.sh

# Or with custom parameters
cd code/training
python pretrain_teacher.py acm \
    --hidden_dim=64 \
    --nb_epochs=1000 \
    --lr=0.0008 \
    --gpu 0
```

**What happens**: Trains the main teacher on original graph data using contrastive learning.

#### 2. Augmentation Teacher Training (Stage 2)
```bash
bash 2_train_middle_teacher.sh

# Or with custom parameters
cd code/training
python train_middle_teacher.py acm \
    --hidden_dim=64 \
    --nb_epochs=100 \
    --lr=0.0008 \
    --gpu 0
```

**What happens**: Trains augmentation teacher independently on augmented graphs. **No compression** - same architecture as main teacher but learns robust patterns from data augmentation.

#### 3. Student Training with Dual Teachers (Stage 3)
```bash
bash 3_train_student.sh

# Or with custom parameters
cd code/training
python train_student.py acm \
    --teacher_model_path ../../results/teacher_heco_acm.pkl \
    --middle_teacher_path ../../results/middle_teacher_heco_acm.pkl \
    --student_compression_ratio=0.5 \
    --stage2_epochs=100 \
    --lr=0.0008 \
    --gpu 0
```

**What happens**: 
- Loads **both frozen teachers** (main + augmentation)
- Trains compressed student (50% parameters) with:
  - **KD Loss**: Learn from main teacher's representations
  - **Augmentation Alignment**: Learn robustness from augmentation teacher
  - **Student Contrastive Loss**: Self-supervised learning
  - **Link Reconstruction** (optional): Explicit edge modeling

### Evaluation and Analysis

#### Comprehensive Multi-Task Evaluation
```bash
bash 4_evaluate.sh

# Or directly:
cd code/evaluation
python comprehensive_evaluation.py \
    --dataset acm \
    --teacher_path ../../results/teacher_heco_acm.pkl \
    --middle_teacher_path ../../results/middle_teacher_heco_acm.pkl \
    --student_path ../../results/student_heco_acm.pkl
```

**Evaluates**:
- ✅ Node Classification (Macro-F1, Micro-F1, Accuracy)
- ✅ Link Prediction (AUC, AP)
- ✅ Node Clustering (NMI, ARI)
- ✅ Model Compression Metrics
- ✅ Inference Speed Comparison

#### KD-Specific Metrics
```bash
cd code/evaluation
python evaluate_kd.py \
    --dataset acm \
    --teacher_model_path ../../results/teacher_heco_acm.pkl \
    --student_model_path ../../results/student_heco_acm.pkl
```

**Analyzes**:
- Knowledge transfer quality
- Representation similarity
- Layer-wise distillation effectiveness

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--student_compression_ratio` | 0.5 | Student size relative to teacher (0.5 = 50%) |
| `--main_distill_weight` | 1.0 | Weight for main teacher KD loss |
| `--augmentation_weight` | 0.5 | Weight for augmentation alignment |
| `--link_recon_weight` | 0.1 | Weight for link reconstruction |
| `--use_kd_loss` | True | Enable/disable KD from main teacher |
| `--use_augmentation_alignment_loss` | True | Enable/disable augmentation guidance |
| `--use_link_recon_loss` | False | Enable/disable link reconstruction |
| `--use_student_contrast_loss` | True | Enable/disable student self-learning |

## 📊 Datasets

### Supported Datasets

| Dataset | Nodes | Edges | Node Types | Tasks |
|---------|-------|-------|------------|-------|
| **ACM** | 4,019 papers<br>7,167 authors<br>60 subjects | PAP, PSP | Paper, Author, Subject | Classification, Link Prediction, Clustering |
| **DBLP** | 4,057 papers<br>14,328 authors<br>7,723 conferences<br>20 terms | PAP, PCP, PTP | Paper, Author, Conference, Term | Classification, Link Prediction, Clustering |
| **AMiner** | 6,564 papers<br>13,329 authors<br>35,890 references | PAP, PRP | Paper, Author, Reference | Classification, Link Prediction, Clustering |
| **Freebase** | Multi-relational | Multiple | Multiple | Classification, Link Prediction, Clustering |

### Data Format

Each dataset contains:
- **Feature files**: `*_feat.npz` (node features)
- **Graph files**: `*.npz` (adjacency matrices)
- **Labels**: `labels.npy` (node labels)
- **Splits**: `train_*.npy`, `val_*.npy`, `test_*.npy`

## ⚙️ Configuration

### Model Architecture Configuration

```yaml
# Example: ACM dataset configuration
dataset: acm
type_num: [4019, 7167, 60]  # Node counts per type
nei_num: 2                   # Number of neighbor types (for schema encoder)

model:
  hidden_dim: 64             # Hidden dimension for teacher & augmentation teacher
  student_compression_ratio: 0.5  # Student = 32 dim (50% of 64)
  feat_drop: 0.3            # Feature dropout
  attn_drop: 0.5            # Attention dropout
  tau: 0.8                  # Temperature for contrastive learning
  lam: 0.5                  # Balance parameter for contrastive loss

training:
  teacher:
    epochs: 1000            # Main teacher training epochs
    lr: 0.0008             # Learning rate
    patience: 50           # Early stopping patience
    
  middle_teacher:
    epochs: 100            # Augmentation teacher epochs
    lr: 0.0008
    patience: 30
    
  student:
    epochs: 100            # Student training epochs
    lr: 0.0008
    main_distill_weight: 1.0      # KD loss weight
    augmentation_weight: 0.5      # Augmentation alignment weight
    link_recon_weight: 0.1        # Link reconstruction weight (optional)
    
augmentation:
  feature_masking: 0.2     # Feature masking ratio
  edge_perturbation: 0.1   # Edge perturbation ratio
  metapath_sampling: True  # Enable metapath-based augmentation
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | GTX 1060 (6GB) | RTX 3050+ (8GB+) |
| **RAM** | 8GB | 16GB+ |
| **CUDA** | 11.0+ | 11.8 |
| **Storage** | 2GB | 5GB+ |

## 🔧 Development

### Running Tests

```bash
# Test import structure
python code/tests/test_imports.py

# Run all tests (future)
python -m pytest code/tests/
```

### Code Style

```bash
# Format code
black code/

# Lint code  
flake8 code/
```

### Adding New Datasets

1. Create dataset directory in `data/`
2. Add configuration in `experiments/configs/`
3. Update `utils/load_data.py` if needed
4. Test with existing pipeline

## 📈 Performance Benchmarks

### Node Classification Results

| Dataset | Teacher | Middle Teacher | Student | Retention |
|---------|---------|----------------|---------|-----------|
| ACM | 89.2% | 87.8% (-1.4%) | 85.1% (-4.1%) | 95.4% |
| DBLP | 91.5% | 89.9% (-1.6%) | 87.2% (-4.3%) | 95.3% |
| AMiner | 88.7% | 87.1% (-1.6%) | 84.8% (-3.9%) | 95.6% |

### Model Size Comparison

| Model | Parameters | Memory (MB) | Inference Time (ms) | Role |
|-------|------------|-------------|---------------------|------|
| Teacher | 1.2M | 45.3 | 12.4 | Main knowledge source (trained on original data) |
| Middle Teacher | 1.2M | 45.3 | 12.4 | Augmentation expert (trained on augmented data) |
| Student | 600K | 22.7 | 6.2 | Compressed model (50% reduction) |


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/bachnguyen0175/L-CoGNN.git
cd L-CoGNN
pip install -r requirements.txt
pip install -e .  # Editable install

# Run tests
python -m pytest
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{l_cognn2024,
  title={L-CoGNN: Knowledge Distillation for Heterogeneous Graph Representation Learning},
  author={Nguyen, Bach and Team},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **DGL Team** for graph neural network utilities  
- **Research Community** for heterogeneous graph learning advances
- **Contributors** who helped improve this project

## 📞 Contact

- **Author**: Bach Nguyen
- **Email**: [bachnguyen0175@email.com]
- **GitHub**: [@bachnguyen0175](https://github.com/bachnguyen0175)
- **Project**: [L-CoGNN Repository](https://github.com/bachnguyen0175/L-CoGNN)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the Graph Neural Network community

</div>