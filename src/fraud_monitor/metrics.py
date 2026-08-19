import numpy as np


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction, positive = probability >= threshold, y == 1
    tp = int(np.sum(prediction & positive)); fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive)); tn = int(np.sum(~prediction & ~positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def average_precision(y: np.ndarray, probability: np.ndarray) -> float:
    ranked = y[np.argsort(-probability)]
    positives = int(ranked.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def choose_threshold(y: np.ndarray, probability: np.ndarray, target_recall: float = 0.80) -> dict:
    candidates = np.unique(np.quantile(probability, np.linspace(0, 1, 2001)))
    results = [classification_metrics(y, probability, value) for value in candidates]
    eligible = [item for item in results if item["recall"] >= target_recall]
    return max(eligible or results, key=lambda item: (item["precision"], item["f1"], item["recall"]))
