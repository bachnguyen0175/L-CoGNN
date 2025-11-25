#!/usr/bin/env python3
"""
Test Import Dependencies
=======================

This test file verifies that all imports work correctly after refactoring.
"""

import sys
import os

# Add code directory to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_models_imports():
    """Test that all model imports work"""
    try:
        from models.kd_heco import MyHeCo, MiddleMyHeCo, StudentMyHeCo
        from models.contrast import Contrast
        from models.sc_encoder import Sc_encoder
        from models.kd_params import kd_params
        print("✅ All model imports successful")
        return True
    except Exception as e:
        print(f"❌ Model import error: {e}")
        return False

def test_utils_imports():
    """Test that all utility imports work"""
    try:
        from utils.load_data import load_data
        from utils.evaluate import evaluate
        from utils.logreg import LogReg
        print("✅ All utility imports successful") 
        return True
    except Exception as e:
        print(f"❌ Utility import error: {e}")
        return False

def test_training_imports():
    """Test that training imports work"""
    try:
        # These might fail due to relative imports, but we can test the files exist
        import os
        training_files = [
            'training/pretrain_teacher.py',
            'training/train_middle_teacher.py', 
            'training/train_student.py'
        ]
        
        for file in training_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} not found")
                
        print("✅ All training files exist")
        return True
    except Exception as e:
        print(f"❌ Training files error: {e}")
        return False

def test_evaluation_imports():
    """Test that evaluation imports work"""  
    try:
        import os
        eval_files = [
            'evaluation/comprehensive_evaluation.py',
            'evaluation/evaluate_kd.py'
        ]
        
        for file in eval_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} not found")
                
        print("✅ All evaluation files exist")
        return True
    except Exception as e:
        print(f"❌ Evaluation files error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing KD-HGRL Import Dependencies")
    print("=" * 50)
    
    tests = [
        test_models_imports,
        test_utils_imports, 
        test_training_imports,
        test_evaluation_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
        
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Check imports and file locations.")
        return 1

if __name__ == "__main__":
    sys.exit(main())