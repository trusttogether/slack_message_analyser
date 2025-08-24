import os
import json
from src.eval.metrics import precision_recall_f1, jaccard_similarity
from src.eval.json_diff_utils import json_diff
from config.config import DATA_DIR, OUTPUT_DIR

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def match_topics(gt_topics, pred_topics):
    # Simple matching by thread_ids overlap
    matches = []
    for gt in gt_topics:
        best_match = None
        best_score = 0
        for pred in pred_topics:
            score = jaccard_similarity(set(gt["thread_ids"]), set(pred.get("thread_ids", [])))
            if score > best_score:
                best_score = score
                best_match = pred
        matches.append((gt, best_match, best_score))
    return matches

def main():
    gt_path = os.path.join(DATA_DIR, "ground_truth_topics.json")
    out1_path = os.path.join(OUTPUT_DIR, "output_approach1.json")
    out2_path = os.path.join(OUTPUT_DIR, "output_approach2.json")

    gt = load_json(gt_path)
    out1 = load_json(out1_path)
    out2 = load_json(out2_path)

    print("=== Approach 1 ===")
    matches1 = match_topics(gt, out1)
    for gt, pred, score in matches1:
        print(f"GT: {gt['summary']}")
        print(f"Pred: {pred.get('summary', 'N/A') if pred else 'N/A'}")
        print(f"Thread Jaccard: {score:.2f}")
        p, r, f1 = precision_recall_f1(gt["participants"], pred.get("participants", []))
        print(f"Participants P/R/F1: {p:.2f}/{r:.2f}/{f1:.2f}")
        print("----")

    print("=== Approach 2 ===")
    topics2 = out2["topics"] if "topics" in out2 else out2
    matches2 = match_topics(gt, topics2)
    for gt, pred, score in matches2:
        print(f"GT: {gt['summary']}")
        print(f"Pred: {pred.get('summary', 'N/A') if pred else 'N/A'}")
        print(f"Thread Jaccard: {score:.2f}")
        p, r, f1 = precision_recall_f1(gt["participants"], pred.get("participants", []))
        print(f"Participants P/R/F1: {p:.2f}/{r:.2f}/{f1:.2f}")
        print("----")

    print("=== JSON Diff (Approach 1 vs GT) ===")
    print(json_diff(gt, out1))

    print("=== JSON Diff (Approach 2 vs GT) ===")
    print(json_diff(gt, topics2))

if __name__ == "__main__":
    main() 