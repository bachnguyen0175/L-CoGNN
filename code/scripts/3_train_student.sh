#!/bin/bash

# Stage 3: Dual-Teacher Student Training (Main Teacher + Pruning Expert)
echo "🟠 Stage 3: Dual-Teacher Student Training"
echo "========================================="
echo "Main Teacher: Knowledge Distillation (trained on original data)"
echo "Pruning Expert: Pruning Guidance (trained on augmented data)"
echo "========================================="

DATASET="acm"
# Paths for checking (from scripts directory)
TEACHER_MODEL_CHECK="../../results/teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL_CHECK="../../results/middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL_CHECK="../../results/student_heco_${DATASET}.pkl"

# Paths for Python script (from code directory after cd ..)
TEACHER_MODEL="../results/teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL="../results/middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL="../results/student_heco_${DATASET}.pkl"

# Check if student model already exists
if [ -f "$STUDENT_MODEL_CHECK" ]; then
    echo "✅ Student model already exists: $STUDENT_MODEL_CHECK"
    echo "Delete the file if you want to retrain."
    exit 0
fi

# Check if required models exist
MODELS_MISSING=false

if [ ! -f "$TEACHER_MODEL_CHECK" ]; then
    echo "⚠️  Main teacher model not found: $TEACHER_MODEL_CHECK"
    echo "   Training without knowledge distillation"
    MODELS_MISSING=true
fi

if [ ! -f "$MIDDLE_TEACHER_MODEL_CHECK" ]; then
    echo "⚠️  Pruning expert model not found: $MIDDLE_TEACHER_MODEL_CHECK"
    echo "   Training without pruning guidance"
    MODELS_MISSING=true
fi

if [ "$MODELS_MISSING" = true ]; then
    echo ""
    echo "📝 Recommendation: Run previous stages for full dual-teacher training:"
    echo "   1_train_teacher.sh (for knowledge distillation)"
    echo "   2_train_middle_teacher.sh (for pruning guidance)"
    echo ""
    read -p "Continue with available models? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled by user"
        exit 1
    fi
fi

echo "Starting dual-teacher student training on GPU..."
if [ -f "$TEACHER_MODEL_CHECK" ]; then
    echo "✅ Using main teacher for knowledge distillation"
fi
if [ -f "$MIDDLE_TEACHER_MODEL_CHECK" ]; then
    echo "✅ Using pruning expert for guidance"
fi

cd .. && PYTHONPATH=. ../.venv/bin/python training/train_student.py \
    $DATASET \
    --hidden_dim=64 \
    --stage2_epochs=200 \
    --teacher_model_path="$TEACHER_MODEL" \
    --patience=30 \
    --lr=0.0008 \
    --tau=0.8 \
    --feat_drop=0.3 \
    --attn_drop=0.5 \
    --sample_rate 7 1 \
    --lam=0.5 \
    --middle_teacher_path="$MIDDLE_TEACHER_MODEL" \
    --student_save_path="$STUDENT_MODEL" \
    --cuda \
    --seed=42

if [ $? -eq 0 ]; then
    echo "✅ Student training completed!"
    echo "📁 Model saved: $STUDENT_MODEL"
else
    echo "❌ Student training failed!"
    exit 1
fi