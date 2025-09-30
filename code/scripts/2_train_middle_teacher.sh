#!/bin/bash

# Stage 2: Middle Teacher Training from Teacher
echo "🟡 Stage 2: Training Middle Teacher from Teacher"
echo "==============================================="

DATASET="acm"
# Paths for checking (from scripts directory)
MIDDLE_TEACHER_MODEL_CHECK="../../results/middle_teacher_heco_${DATASET}.pkl"

# Paths for Python script (from code directory after cd ..)  
MIDDLE_TEACHER_MODEL="../results/middle_teacher_heco_${DATASET}.pkl"

# Check if middle teacher already exists
if [ -f "$MIDDLE_TEACHER_MODEL_CHECK" ]; then
    echo "✅ Middle teacher model already exists: $MIDDLE_TEACHER_MODEL_CHECK"
    echo "Delete the file if you want to retrain."
    exit 0
fi

echo "Training middle teacher from teacher on GPU..."

cd .. && PYTHONPATH=. ../.venv/bin/python training/train_middle_teacher.py \
    $DATASET \
    --hidden_dim=64 \
    --stage1_epochs=300 \
    --patience=30 \
    --lr=0.0008 \
    --tau=0.8 \
    --feat_drop=0.3 \
    --attn_drop=0.5 \
    --sample_rate 7 1 \
    --lam=0.5 \
    --middle_teacher_save_path="$MIDDLE_TEACHER_MODEL" \
    --cuda \
    --seed=42

if [ $? -eq 0 ]; then
    echo "✅ Middle teacher training completed!"
    echo "📁 Model saved: $MIDDLE_TEACHER_MODEL"
else
    echo "❌ Middle teacher training failed!"
    exit 1
fi