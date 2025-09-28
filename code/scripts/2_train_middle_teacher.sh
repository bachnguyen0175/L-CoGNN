#!/bin/bash

# Stage 2: Middle Teacher Training from Teacher
echo "🟡 Stage 2: Training Middle Teacher from Teacher"
echo "==============================================="

DATASET="acm"
TEACHER_MODEL="teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL="middle_teacher_heco_${DATASET}.pkl"

# Check if teacher exists
if [ ! -f "$TEACHER_MODEL" ]; then
    echo "❌ Teacher model not found: $TEACHER_MODEL"
    echo "Please run 1_train_teacher.sh first"
    exit 1
fi

echo "Training middle teacher from teacher on GPU..."

../.venv/bin/python ../training/train_middle_teacher.py \
    $DATASET \
    --hidden_dim=64 \
    --stage1_epochs=500 \
    --patience=50 \
    --lr=0.001 \
    --tau=0.8 \
    --feat_drop=0.3 \
    --attn_drop=0.5 \
    --sample_rate 7 1 \
    --lam=0.5 \
    --teacher_model_path="$TEACHER_MODEL" \
    --middle_teacher_save_path="$MIDDLE_TEACHER_MODEL" \
    --middle_compression_ratio=0.7 \
    --stage1_distill_weight=0.7 \
    --use_node_masking \
    --use_edge_augmentation \
    --use_autoencoder \
    --mask_rate=0.1 \
    --remask_rate=0.3 \
    --edge_drop_rate=0.1 \
    --num_remasking=2 \
    --reconstruction_weight=0.1 \
    --cuda \
    --seed=42

if [ $? -eq 0 ]; then
    echo "✅ Middle teacher training completed!"
    echo "📁 Model saved: $MIDDLE_TEACHER_MODEL"
else
    echo "❌ Middle teacher training failed!"
    exit 1
fi
