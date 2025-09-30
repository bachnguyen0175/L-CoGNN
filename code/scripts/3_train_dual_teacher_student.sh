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

if [ ! -f "../../results/middle_teacher_heco_acm.pkl" ]; then
    echo "❌ Middle teacher model not found: ../../results/middle_teacher_heco_acm.pkl"
    echo "   Please train the middle teacher first using: bash 2_train_middle_teacher.sh"
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
TEACHER_PATH="../../results/teacher_heco_acm.pkl"
MIDDLE_TEACHER_PATH="../../results/middle_teacher_heco_acm.pkl"
STUDENT_SAVE_PATH="../../results/student_heco_acm.pkl"

# Enhanced student training parameters - optimized for dual-teacher learning
STUDENT_EPOCHS=500
STUDENT_LR=0.0008
STUDENT_COMPRESSION=0.6  # Less aggressive compression for better learning from two teachers
DISTILL_WEIGHT=0.5       # Reduced main teacher weight (since middle teacher is better)
PRUNING_WEIGHT=1.0       # High weight for better middle teacher

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Student epochs: $STUDENT_EPOCHS"
echo "  Student compression: $STUDENT_COMPRESSION"
echo "  Distillation weight: $DISTILL_WEIGHT"
echo "  Teacher path: $TEACHER_PATH"
echo "  Middle teacher path: $MIDDLE_TEACHER_PATH"
echo ""

# Run dual-teacher student training
python ../training/train_student.py \
    --dataset $DATASET \
    --teacher_model_path $TEACHER_PATH \
    --middle_teacher_path $MIDDLE_TEACHER_PATH \
    --student_save_path $STUDENT_SAVE_PATH \
    --stage2_epochs $STUDENT_EPOCHS \
    --lr $STUDENT_LR \
    --stage2_distill_weight $DISTILL_WEIGHT \
    --pruning_weight $PRUNING_WEIGHT \
    --student_compression_ratio $STUDENT_COMPRESSION \
    --kd_temperature 3.0 \
    --patience 30 \
    --save_interval 20 \
    --eval_interval 10 \
    --log_interval 5 \
    $GPU_FLAG

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dual-teacher student training completed successfully!"
    echo "📁 Student model saved to: $STUDENT_SAVE_PATH"
    echo ""
    echo "📊 Model Summary:"
    echo "   - Main Teacher: Knowledge distillation from original data"
    echo "   - Middle Teacher: Pruning guidance from augmented data"
    echo "   - Student: Learned from both teachers with $STUDENT_COMPRESSION compression"
    echo ""
    echo "🎯 Next Step: Evaluate the student model using bash 4_evaluate.sh"
else
    echo "❌ Dual-teacher student training failed!"
    exit 1
fi