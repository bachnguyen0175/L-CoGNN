#!/bin/bash

# Stage 1: Teacher Model Training
echo "🔵 Stage 1: Training Teacher Model"
echo "=================================="

DATASET="acm"
TEACHER_MODEL="teacher_heco_${DATASET}.pkl"

if [ -f "$TEACHER_MODEL" ]; then
    echo "✅ Teacher model already exists: $TEACHER_MODEL"
    echo "Delete the file if you want to retrain."
    exit 0
fi

echo "Training teacher model on GPU..."

../.venv/bin/python ../training/pretrain_teacher.py \
    $DATASET \
    --hidden_dim=64 \
    --nb_epochs=10000 \
    --patience=50 \
    --lr=0.0008 \
    --tau=0.8 \
    --feat_drop=0.3 \
    --attn_drop=0.5 \
    --sample_rate 7 1 \
    --lam=0.5 \
    --teacher_save_path="$TEACHER_MODEL" \
    --cuda \
    --seed=42

if [ $? -eq 0 ]; then
    echo "✅ Teacher training completed!"
    echo "📁 Model saved: $TEACHER_MODEL"
else
    echo "❌ Teacher training failed!"
    exit 1
fi
