from sklearn.metrics import precision_score, recall_score, f1_score

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 1.0

def precision_recall_f1(true, pred):
    true_set = set(true)
    pred_set = set(pred)
    precision = len(true_set & pred_set) / len(pred_set) if pred_set else 0
    recall = len(true_set & pred_set) / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1 