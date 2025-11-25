#!/bin/bash

# KD-HGRL Complete Training Pipeline
# ==================================

DATASET=${1:-acm}
echo "🚀 Running complete KD-HGRL pipeline for dataset: $DATASET"
echo ""

# Check if we're in the scripts directory
if [ ! -f "1_train_teacher.sh" ]; then
    echo "❌ Please run this script from the scripts/ directory"
    exit 1
fi

echo "📋 Pipeline Overview:"
echo "   Stage 1: Teacher Training (~30-60 min)"
echo "   Stage 2: Middle Teacher Training (~15-30 min)" 
echo "   Stage 3: Student Training (~20-40 min)"
echo "   Stage 4: Evaluation (~5 min)"
echo ""

# Stage 1: Teacher Training
echo "🔵 Stage 1: Teacher Training"
echo "=============================="
bash 1_train_teacher.sh
if [ $? -ne 0 ]; then
    echo "❌ Teacher training failed!"
    exit 1
fi
echo ""

# Stage 2: Middle Teacher Training  
echo "🟡 Stage 2: Middle Teacher Training"
echo "==================================="
bash 2_train_middle_teacher.sh
if [ $? -ne 0 ]; then
    echo "❌ Middle teacher training failed!"
    exit 1
fi
echo ""

# Stage 3: Student Training
echo "🟠 Stage 3: Student Training" 
echo "============================"
bash 3_train_student.sh
if [ $? -ne 0 ]; then
    echo "❌ Student training failed!"
    exit 1
fi
echo ""

# Stage 4: Evaluation
echo "🟣 Stage 4: Evaluation"
echo "======================"
bash 4_evaluate.sh
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Complete pipeline completed successfully for $DATASET!"
echo ""
echo "📁 Generated Models:"
echo "   - teacher_heco_${DATASET}.pkl"
echo "   - middle_teacher_heco_${DATASET}.pkl" 
echo "   - student_heco_${DATASET}.pkl"
echo ""
echo "🎉 Training pipeline finished!"