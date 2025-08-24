import json
import os
from src.eval.step2_evaluator import evaluate_step2_output
from config.config import DATA_DIR

def test_step2_evaluation():
    """Test the Step 2 evaluation functionality"""
    
    # Load ground truth
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Test evaluation
    results = evaluate_step2_output(ground_truth, ground_truth)  # Self-comparison for testing
    
    print("Step 2 Evaluation Test Results:")
    print(f"Topic Accuracy: {results.get('topic_accuracy', 0):.3f}")
    print(f"Action Item Precision: {results.get('action_precision', 0):.3f}")
    print(f"Action Item Recall: {results.get('action_recall', 0):.3f}")
    print(f"Action Item F1: {results.get('action_f1', 0):.3f}")

if __name__ == "__main__":
    test_step2_evaluation()
