"""Binary classification metrics computed from probabilities, without sklearn."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BinaryMetrics:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float  # recall on Loose - the clinically expensive miss
    specificity: float
    precision: float
    f1: float
    auc: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"acc={self.accuracy:.3f} bal_acc={self.balanced_accuracy:.3f} "
            f"auc={self.auc:.3f} sens={self.sensitivity:.3f} spec={self.specificity:.3f} "
            f"f1={self.f1:.3f} @thr={self.threshold:.2f}"
        )


def roc_auc(labels: list[int], probs: list[float]) -> float:
    """Rank-based AUC (Mann-Whitney U) with average ranks for ties."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and probs[order[stop + 1]] == probs[order[index]]:
            stop += 1
        average_rank = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = average_rank
        index = stop + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def compute_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> BinaryMetrics:
    tp = sum(1 for label, prob in zip(labels, probs) if label == 1 and prob >= threshold)
    fp = sum(1 for label, prob in zip(labels, probs) if label == 0 and prob >= threshold)
    tn = sum(1 for label, prob in zip(labels, probs) if label == 0 and prob < threshold)
    fn = sum(1 for label, prob in zip(labels, probs) if label == 1 and prob < threshold)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return BinaryMetrics(
        threshold=threshold,
        accuracy=(tp + tn) / len(labels) if labels else 0.0,
        balanced_accuracy=(sensitivity + specificity) / 2,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        auc=roc_auc(labels, probs),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def best_threshold(labels: list[int], probs: list[float]) -> float:
    """Threshold maximising Youden's J, picked on validation data only."""
    candidates = sorted({round(prob, 4) for prob in probs} | {0.5})
    best, best_j = 0.5, -1.0
    for threshold in candidates:
        metrics = compute_metrics(labels, probs, threshold)
        j = metrics.sensitivity + metrics.specificity - 1
        if j > best_j:
            best, best_j = threshold, j
    return best
