#!/bin/bash

# Stage 4: Evaluation and Comparison
echo "🟣 Stage 4: Evaluation and Comparison"
echo "===================================="

# Simple approach: work from scripts directory, use relative paths
DATASET="acm"
# Paths relative to scripts directory for checking
TEACHER_MODEL_CHECK="../../results/teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL_CHECK="../../results/middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL_CHECK="../../results/student_heco_${DATASET}.pkl"

# Paths relative to code directory for evaluation scripts
TEACHER_MODEL="../results/teacher_heco_${DATASET}.pkl"
MIDDLE_TEACHER_MODEL="../results/middle_teacher_heco_${DATASET}.pkl"
STUDENT_MODEL="../results/student_heco_${DATASET}.pkl"

# Check if all models exist
missing_models=()

if [ ! -f "$TEACHER_MODEL_CHECK" ]; then
    missing_models+=("$TEACHER_MODEL_CHECK")
fi

if [ ! -f "$MIDDLE_TEACHER_MODEL_CHECK" ]; then
    missing_models+=("$MIDDLE_TEACHER_MODEL_CHECK")
fi

if [ ! -f "$STUDENT_MODEL_CHECK" ]; then
    missing_models+=("$STUDENT_MODEL_CHECK")
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

echo "✅ All models found. Starting evaluation..."

# Move to code directory and run evaluations using kd_params.py for all parameters
cd ..

echo ""
echo "📊 Running KD-specific evaluation..."
PYTHONPATH=. ../.venv/bin/python evaluation/evaluate_kd.py \
    --dataset "$DATASET" \
    --teacher_model_path "$TEACHER_MODEL" \
    --student_model_path "$STUDENT_MODEL"

echo ""
echo "📊 Running comprehensive evaluation on all three tasks..."
PYTHONPATH=. ../.venv/bin/python evaluation/comprehensive_evaluation.py \
    --dataset "$DATASET" \
    --teacher_model_path "$TEACHER_MODEL" \
    --student_model_path "$STUDENT_MODEL" \
    --middle_teacher_path "$MIDDLE_TEACHER_MODEL"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📊 Model Summary:"
    echo "   Teacher:        $TEACHER_MODEL_CHECK"
    echo "   Middle Teacher: $MIDDLE_TEACHER_MODEL_CHECK"  
    echo "   Student:        $STUDENT_MODEL_CHECK"
else
    echo "❌ Evaluation failed!"
    exit 1
fi
