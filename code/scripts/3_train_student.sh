#!/bin/bash
# Dual-Teacher Student Training Script
# Train student model using both teachers:
# 1. Main Teacher: Provides knowledge distillation (trained on original data)  
# 2. Middle Teacher: Provides pruning guidance (trained on augmented data)
echo "🟡 Stage 3: Training Student with Dual-Teacher Guidance"
echo "========================================================"

# Check if required models exist
if [ ! -f "../../results/teacher_heco_acm.pkl" ]; then
    echo "❌ Main teacher model not found: ../../results/teacher_heco_acm.pkl"
    echo "   Please train the main teacher first using: bash 1_train_teacher.sh"
    exit 1
fi

echo "✅ Both teacher models found"
echo "🚀 Starting dual-teacher student training..."

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using CUDA acceleration"
    GPU_FLAG="--gpu 0"
else
    echo "No GPU detected, using CPU"
    GPU_FLAG="--gpu -1"
fi

# Training configuration
DATASET="acm"
TEACHER_PATH="../results/teacher_heco_acm.pkl"
MIDDLE_TEACHER_PATH="../results/middle_teacher_heco_acm.pkl"
STUDENT_SAVE_PATH="../results/student_heco_acm.pkl"

# Run dual-teacher student training
cd .. && PYTHONPATH=. ../.venv/bin/python training/train_student.py \
    $DATASET \
    --teacher_model_path $TEACHER_PATH \
    --middle_teacher_path $MIDDLE_TEACHER_PATH \
    --student_save_path $STUDENT_SAVE_PATH \
    --stage2_epochs=100 \
    --lr=0.0008 \
    --student_compression_ratio=0.5 \
    $GPU_FLAG\

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dual-teacher student training completed successfully!"
    echo "📁 Student model saved to: $STUDENT_SAVE_PATH"
else
    echo "❌ Dual-teacher student training failed!"
    exit 1
fi