# KD-HGRL Scripts Documentation

This directory contains all the shell scripts for running the KD-HGRL training pipeline.

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
bash run_all.sh acm
```

### Option 2: Run Individual Stages
```bash
bash 1_train_teacher.sh      # Stage 1: Teacher training
bash 2_train_middle_teacher.sh   # Stage 2: Middle teacher training  
bash 3_train_student.sh      # Stage 3: Student training
bash 4_evaluate.sh           # Stage 4: Evaluation
```

## 📋 Script Descriptions

### **1_train_teacher.sh**
- **Purpose**: Train the teacher model from scratch
- **Output**: `teacher_heco_acm.pkl` (or other dataset)
- **Time**: ~30-60 minutes
- **GPU**: Required for optimal performance

### **2_train_middle_teacher.sh** 
- **Purpose**: Train middle teacher (augmentation expert) on augmented graphs
- **Input**: None (trains independently, no teacher needed)
- **Output**: `middle_teacher_heco_acm.pkl`
- **Time**: ~15-30 minutes
- **Compression**: No compression (same size as teacher, different training data)

### **3_train_student.sh**
- **Purpose**: Train student model via dual-teacher distillation  
- **Input**: Teacher + Middle teacher checkpoints
- **Output**: `student_heco_acm.pkl`
- **Time**: ~20-40 minutes
- **Compression**: ~50% parameter reduction (compressed student model)

### **4_evaluate.sh**
- **Purpose**: Comprehensive evaluation on all three downstream tasks
- **Input**: All model checkpoints
- **Output**: Performance comparison and analysis
- **Time**: ~5 minutes

### **run_all.sh**
- **Purpose**: Execute complete pipeline automatically
- **Usage**: `bash run_all.sh [dataset]` 
- **Default**: ACM dataset
- **Features**: Progress tracking, error handling, summary

## 🔧 Requirements

- **CUDA-compatible GPU** (RTX 3050 or better recommended)
- **PyTorch 2.1.2** with CUDA 11.8 support
- **Virtual Environment** activated
- **Data files** in correct directory structure

## 📊 Expected Results

| Model | Parameters | Performance | Compression | Role |
|-------|------------|-------------|-------------|------|
| Teacher | 100% | Baseline | - | Knowledge source (original data) |
| Middle Teacher | 100% | ~98% retention | No compression | Augmentation expert (augmented data) |
| Student | ~50% | ~95% retention | 50% reduction | Compressed final model |

## 🚨 Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch size or use smaller model
2. **Import errors**: Check Python environment and paths  
3. **Missing models**: Run stages in sequence (1→2→3→4)
4. **Permission denied**: Make scripts executable with `chmod +x`

### Getting Help:
- Check console output for specific error messages
- Verify all dependencies are installed
- Ensure data files are in correct locations
- Try running individual stages to isolate issues

## 🎯 Performance Tips

- **Use GPU**: Add `--cuda` flag (included in scripts)
- **Monitor memory**: Use `nvidia-smi` to check GPU usage
- **Parallel execution**: Can run multiple datasets simultaneously on different GPUs
- **Early stopping**: Adjust patience parameter if training takes too long