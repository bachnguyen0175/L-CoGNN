#!/bin/bash

# Stage 3: Evaluation and Comparison
echo "🟣 Stage 3: Evaluation and Comparison"
echo "===================================="

# Simple approach: work from scripts directory, use relative paths
DATASET="acm"
# Paths relative to scripts directory for checking
TEACHER_MODEL_CHECK="../../results/teacher_heco_${DATASET}.pkl"
STUDENT_MODEL_CHECK="../../results/student_heco_${DATASET}.pkl"

# Paths relative to code directory for evaluation scripts
TEACHER_MODEL="../results/teacher_heco_${DATASET}.pkl"
STUDENT_MODEL="../results/student_heco_${DATASET}.pkl"


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
    --student_model_path "$STUDENT_MODEL"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📊 Model Summary:"
    echo "   Teacher: $TEACHER_MODEL_CHECK"
    echo "   Student: $STUDENT_MODEL_CHECK"
else
    echo "❌ Evaluation failed!"
    exit 1
fi
