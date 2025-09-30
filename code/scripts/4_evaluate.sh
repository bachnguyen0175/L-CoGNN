#!/bin/bash

# Stage 4: Evaluation and Comparison
echo "🟣 Stage 4: Evaluation and Comparison"
echo "===================================="

DATASET="acm"
TEACHER_MODEL="../../results/teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL="../../results/middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL="../../results/student_heco_${DATASET}.pkl"

# Check if all models exist
missing_models=()

if [ ! -f "$TEACHER_MODEL" ]; then
    missing_models+=("$TEACHER_MODEL")
fi

if [ ! -f "$MIDDLE_TEACHER_MODEL" ]; then
    missing_models+=("$MIDDLE_TEACHER_MODEL")
fi

if [ ! -f "$STUDENT_MODEL" ]; then
    missing_models+=("$STUDENT_MODEL")
fi

if [ ${#missing_models[@]} -gt 0 ]; then
    echo "❌ Missing model files:"
    for model in "${missing_models[@]}"; do
        echo "   - $model"
    done
    echo ""
    echo "Please train the missing models first:"
    echo "   1. bash 1_train_teacher.sh"
    echo "   2. bash 2_train_middle_teacher.sh"
    echo "   3. bash 3_train_student.sh"
    exit 1
fi

echo "Evaluating all models on GPU..."

echo ""
cd ..
TEACHER_MODEL_EVAL="../results/teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL_EVAL="../results/middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL_EVAL="../results/student_heco_${DATASET}.pkl"

echo "📊 Running KD-specific evaluation..."
PYTHONPATH=. ../.venv/bin/python evaluation/evaluate_kd.py \
    --dataset="$DATASET" \
    --teacher_model_path="$TEACHER_MODEL_EVAL" \
    --student_model_path="$STUDENT_MODEL_EVAL" \
    --hidden_dim=64 \
    --gpu=0

echo ""
echo "📊 Running comprehensive evaluation on all three tasks..."
PYTHONPATH=. ../.venv/bin/python evaluation/comprehensive_evaluation.py \
    --dataset="$DATASET" \
    --teacher_path="$TEACHER_MODEL_EVAL" \
    --student_path="$STUDENT_MODEL_EVAL" \
    --middle_teacher_path="$MIDDLE_TEACHER_MODEL_EVAL" \
    --hidden_dim=64 \
    --gpu=0

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed!"
    echo ""
    echo "📊 Model Summary:"
    echo "   Teacher:        $TEACHER_MODEL"
    echo "   Middle Teacher: $MIDDLE_TEACHER_MODEL"
    echo "   Student:        $STUDENT_MODEL"
else
    echo "❌ Evaluation failed!"
    exit 1
fi
