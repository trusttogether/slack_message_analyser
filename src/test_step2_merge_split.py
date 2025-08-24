import json
import os
from src.eval.step2_evaluator import evaluate_step2_output
from config.config import DATA_DIR

def test_merge_split_functionality():
    """Test the merge/split functionality for Step 2 topics"""
    
    # Load ground truth
    ground_truth_path = os.path.join(DATA_DIR, "benchmark_topics_corrected_fixed.json")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Simulate merged topics (combine first two topics)
    if len(ground_truth.get('topics', [])) >= 2:
        merged_topics = ground_truth.copy()
        topic1 = merged_topics['topics'][0]
        topic2 = merged_topics['topics'][1]
        
        # Merge topics
        merged_topic = {
            'id': f"{topic1['id']}_merged_{topic2['id']}",
            'title': f"{topic1['title']} + {topic2['title']}",
            'participants': list(set(topic1.get('participants', []) + topic2.get('participants', []))),
            'messages': topic1.get('messages', []) + topic2.get('messages', []),
            'action_items': topic1.get('action_items', []) + topic2.get('action_items', [])
        }
        
        merged_topics['topics'] = [merged_topic] + merged_topics['topics'][2:]
        
        # Test evaluation with merged topics
        results = evaluate_step2_output(ground_truth, merged_topics)
        
        print("Merge/Split Test Results:")
        print(f"Topic Count Difference: {len(ground_truth['topics']) - len(merged_topics['topics'])}")
        print(f"Topic Accuracy: {results.get('topic_accuracy', 0):.3f}")
        print(f"Action Item Precision: {results.get('action_precision', 0):.3f}")

if __name__ == "__main__":
    test_merge_split_functionality()
