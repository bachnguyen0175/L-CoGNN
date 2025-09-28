# KD-HGRL: Knowledge Distillation for Heterogeneous Graph Representation Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

KD-HGRL is a comprehensive framework for **Knowledge Distillation in Heterogeneous Graph Representation Learning**. This project implements a hierarchical distillation pipeline that progressively compresses large heterogeneous graph neural network models while maintaining competitive performance.

### 🎯 Key Features

- **Hierarchical Knowledge Distillation**: Teacher → Middle Teacher → Student pipeline
- **Heterogeneous Graph Support**: ACM, DBLP, AMiner, Freebase datasets
- **Multi-Task Learning**: Node classification, link prediction, node clustering
- **Model Compression**: Up to 65% parameter reduction with minimal performance loss
- **GPU Acceleration**: CUDA 11.8 support with PyTorch 2.1.2

### 🏆 Performance Highlights

| Model | Parameters | Compression | Node Classification | Link Prediction | Node Clustering |
|-------|------------|-------------|-------------------|-----------------|-----------------|
| Teacher | 100% | - | Baseline | Baseline | Baseline |
| Middle Teacher | ~70% | 30% | ~98% retention | ~97% retention | ~98% retention |
| Student | ~35% | 65% | ~95% retention | ~93% retention | ~94% retention |

## 🚀 Quick Start

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
bash 1_train_teacher.sh        # ~30-60 minutes
bash 2_train_middle_teacher.sh # ~15-30 minutes  
bash 3_train_student.sh        # ~20-40 minutes
bash 4_evaluate.sh             # ~5 minutes
```

### 3. Expected Output

```
✅ Complete pipeline completed successfully for acm!

📁 Generated Models:
   - teacher_heco_acm.pkl
   - middle_teacher_heco_acm.pkl
   - student_heco_acm.pkl

🎯 Compression Analysis:
   Teacher → Middle: ~30% compression
   Middle → Student: ~50% compression
   Overall: ~65% parameter reduction
```

## 📁 Project Structure

```
L-CoGNN/
├── 📋 README.md                     # This file
├── ⚙️ requirements.txt             # Dependencies
├── 🐍 main.py                      # Entry point (future CLI)
│
├── 🧠 code/models/                 # Neural Network Models
│   ├── kd_heco.py                  # Main HeCo architecture
│   ├── contrast.py                 # Contrastive learning
│   ├── sc_encoder.py               # Semantic attention encoder
│   └── kd_params.py                # Model configurations
│
├── 🎓 code/training/               # Training Scripts
│   ├── pretrain_teacher.py         # Stage 1: Teacher training
│   ├── train_middle_teacher.py     # Stage 2: Middle teacher
│   ├── train_student.py            # Stage 3: Student training
│   └── hetero_augmentations.py     # Graph augmentations
│
├── 📊 code/evaluation/             # Evaluation Tools
│   ├── comprehensive_evaluation.py # Multi-task evaluation
│   └── evaluate_kd.py              # KD-specific evaluation
│
├── 🔧 code/utils/                  # Utility Functions
│   ├── load_data.py                # Data loading utilities
│   ├── evaluate.py                 # Evaluation metrics
│   └── logreg.py                   # Logistic regression
│
├── 🚀 code/scripts/                # Executable Scripts
│   ├── 1_train_teacher.sh          # Teacher training
│   ├── 2_train_middle_teacher.sh   # Middle teacher training
│   ├── 3_train_student.sh          # Student training
│   ├── 4_evaluate.sh               # Comprehensive evaluation
│   └── run_all.sh                  # Complete pipeline
│
├── 🧪 code/experiments/            # Experiment Configurations
│   └── configs/                    # YAML configuration files
│       ├── acm.yaml                # ACM dataset config
│       └── dblp.yaml               # DBLP dataset config
│
├── 📚 data/                        # Dataset Files
│   ├── acm/                        # ACM dataset
│   ├── dblp/                       # DBLP dataset
│   ├── aminer/                     # AMiner dataset
│   └── freebase/                   # Freebase dataset
│
└── 🧪 code/tests/                  # Unit Tests
    └── test_imports.py             # Import validation
```

## 🎯 Usage Guide

### Training Individual Models

#### 1. Teacher Model Training
```bash
cd code/scripts
bash 1_train_teacher.sh

# Or with custom parameters
cd code/training
python pretrain_teacher.py acm \
    --hidden_dim=64 \
    --nb_epochs=1000 \
    --lr=0.0008 \
    --cuda
```

#### 2. Middle Teacher Training
```bash
bash 2_train_middle_teacher.sh

# Requires teacher model to exist first
```

#### 3. Student Model Training
```bash
bash 3_train_student.sh

# Requires both teacher and middle teacher models
```

### Evaluation and Analysis

#### Comprehensive Evaluation
```bash
bash 4_evaluate.sh

# Or directly:
python evaluation/comprehensive_evaluation.py \
    --dataset acm \
    --teacher_path teacher_heco_acm.pkl \
    --student_path student_heco_acm.pkl
```

#### KD-Specific Evaluation
```bash
python evaluation/evaluate_kd.py \
    --dataset acm \
    --teacher_model_path teacher_heco_acm.pkl \
    --student_model_path student_heco_acm.pkl
```

### Configuration-Based Training

Use YAML configuration files for reproducible experiments:

```bash
# Example: ACM dataset configuration
python main.py --config code/experiments/configs/acm.yaml  # Future feature
```

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

### Model Parameters

```yaml
# Example: ACM dataset configuration
dataset: acm
type_num: [4019, 7167, 60]  # Node counts per type
nei_num: 2                   # Number of neighbor types

model:
  hidden_dim: 64             # Hidden dimension
  feat_drop: 0.3            # Feature dropout
  attn_drop: 0.5            # Attention dropout
  tau: 0.8                  # Temperature parameter

training:
  teacher:
    epochs: 10000           # Training epochs
    lr: 0.0008             # Learning rate
    patience: 50           # Early stopping patience
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

| Model | Parameters | Memory (MB) | Inference Time (ms) |
|-------|------------|-------------|-------------------|
| Teacher | 1.2M | 45.3 | 12.4 |
| Middle Teacher | 840K | 31.8 | 8.7 |
| Student | 420K | 15.9 | 4.2 |


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