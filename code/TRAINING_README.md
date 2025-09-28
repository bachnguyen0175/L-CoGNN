# KD-HGRL Training Scripts

## 🚀 GPU-Optimized Training Pipeline

This directory contains 4 clean scripts for hierarchical knowledge distillation training on GPU:

### Scripts Overview:

1. **`1_train_teacher.sh`** - Train Teacher Model
   - Trains the full teacher model from scratch
   - Uses GPU acceleration (`--cuda`)
   - Output: `teacher_heco_acm.pkl`

2. **`2_train_middle_teacher.sh`** - Train Middle Teacher  
   - Distills teacher → middle teacher (30% compression)
   - Requires teacher model to exist
   - Output: `middle_teacher_heco_acm.pkl`

3. **`3_train_student.sh`** - Train Student Model
   - Distills middle teacher → student (50% compression) 
   - Requires teacher and middle teacher models
   - Output: `student_heco_acm.pkl`

4. **`4_evaluate.sh`** - Evaluate & Compare
   - Evaluates all three models
   - Shows compression ratios and performance
   - Requires all models to exist

### 🎯 Usage:

```bash
# Run each stage in sequence:
bash 1_train_teacher.sh        # ~30-60 minutes
bash 2_train_middle_teacher.sh # ~15-30 minutes  
bash 3_train_student.sh        # ~20-40 minutes
bash 4_evaluate.sh             # ~5 minutes

# Or run all at once:
bash 1_train_teacher.sh && bash 2_train_middle_teacher.sh && bash 3_train_student.sh && bash 4_evaluate.sh
```

### 📊 Expected Results:
- **Teacher Model**: Full capacity, baseline performance
- **Middle Teacher**: ~30% smaller, minimal performance loss
- **Student Model**: ~65% smaller overall, competitive performance

### 🛠️ Requirements:
- CUDA-compatible GPU (RTX 3050 Mobile detected)
- PyTorch 1.12.1+cu116
- All dependencies from requirements.txt

### ✅ GPU Status:
- NVIDIA GeForce RTX 3050 Laptop GPU (3.7GB)
- CUDA 11.6 support enabled
- All scripts use `--cuda` flag for acceleration
