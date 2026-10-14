"""metrics_utils.py - Tính 5 metric của đề tài: Accuracy, Precision, Recall, F1, FPR.

Nhãn 1 = lừa đảo (positive), nhãn 0 = tin hợp lệ.
FPR = FP / (FP + TN) = tỉ lệ tin hợp lệ bị báo nhầm là lừa đảo.
Module dùng chung cho train_*.py và evaluate_models.py.
"""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
