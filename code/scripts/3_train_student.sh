#!/bin/bash

# Stage 3: Student Training from Middle Teacher
echo "🟠 Stage 3: Training Student from Middle Teacher"
echo "==============================================="

DATASET="acm"
TEACHER_MODEL="teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL="middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL="student_heco_${DATASET}.pkl"

# Check if required models exist
if [ ! -f "$TEACHER_MODEL" ]; then
    echo "❌ Teacher model not found: $TEACHER_MODEL"
    echo "Please run 1_train_teacher.sh first"
    exit 1
fi

if [ ! -f "$MIDDLE_TEACHER_MODEL" ]; then
    echo "❌ Middle teacher model not found: $MIDDLE_TEACHER_MODEL"
    echo "Please run 2_train_middle_teacher.sh first"
    exit 1
fi

echo "Training student from middle teacher on GPU..."

../.venv/bin/python ../training/train_student.py \
    $DATASET \
    --hidden_dim=64 \
    --stage2_epochs=300 \
    --patience=50 \
    --lr=0.001 \
    --tau=0.8 \
    --feat_drop=0.3 \
    --attn_drop=0.5 \
    --sample_rate 7 1 \
    --lam=0.5 \
    --middle_teacher_path="$MIDDLE_TEACHER_MODEL" \
    --student_save_path="$STUDENT_MODEL" \
    --student_compression_ratio=0.5 \
    --distill_from_middle \
    --use_embedding_kd \
    --use_heterogeneous_kd \
    --use_multi_level_kd \
    --use_progressive_pruning \
    --use_multi_stage \
    --embedding_weight=0.5 \
    --heterogeneous_weight=0.3 \
    --multi_level_weight=0.4 \
    --subspace_weight=0.3 \
    --embedding_temp=4.0 \
    --mask_epochs=100 \
    --fixed_epochs=100 \
    --pruning_start=10 \
    --pruning_interval=10 \
    --emb_prune_ratio=0.1 \
    --mp_prune_ratio=0.05 \
    --cuda \
    --seed=42

if [ $? -eq 0 ]; then
    echo "✅ Student training completed!"
    echo "📁 Model saved: $STUDENT_MODEL"
else
    echo "❌ Student training failed!"
    exit 1
fi