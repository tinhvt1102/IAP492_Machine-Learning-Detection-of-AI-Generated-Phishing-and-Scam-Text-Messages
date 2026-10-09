"""
train_lr.py
Nhiệm vụ DUY NHẤT: train Logistic Regression trên đặc trưng TF-IDF.

- Đọc X/y train, val từ models/baseline/ (do tfidf_vectorizer.py tạo).
- Dò tham số C trên VAL, chấm theo F1 (lớp 1 = phishing/scam).
- Quy tắc chọn (dung sai): trong các C có F1 >= F1_max - selection_tolerance,
  chọn C NHỎ NHẤT (model được kiềm chế nhiều nhất = đơn giản nhất).
- Model cuối = model fit trên TRAIN với C tốt nhất.
- Lưu: models/baseline/lr_model.joblib
       results/metrics/lr_val_search.csv (điểm trên val của từng C)
- KHÔNG đánh giá trên test (việc của evaluate_models.py).

Chạy từ thư mục gốc project:
    python src/models/train_lr.py --config configs/model_config.yaml
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Logistic Regression (TF-IDF)")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lcfg = cfg["lr"]
    base_dir = Path(cfg["paths"]["baseline_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])

    # Bước 1: đọc đặc trưng và nhãn (chỉ train + val, KHÔNG đọc test)
    X_train = sparse.load_npz(base_dir / "X_train.npz")
    y_train = np.load(base_dir / "y_train.npy")
    X_val = sparse.load_npz(base_dir / "X_val.npz")
    y_val = np.load(base_dir / "y_val.npy")
    log.info("Train: %s | Val: %s", X_train.shape, X_val.shape)

    # Bước 2: thử từng giá trị C: fit trên train, chấm trên val
    rows, models = [], {}
    for C in sorted(lcfg["C_grid"]):
        model = LogisticRegression(
            C=C, class_weight=lcfg["class_weight"],
            max_iter=lcfg["max_iter"], random_state=lcfg["random_seed"])
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        row = {"C": C,
               "val_f1": f1_score(y_val, pred),
               "val_precision": precision_score(y_val, pred, zero_division=0),
               "val_recall": recall_score(y_val, pred)}
        rows.append(row)
        log.info("C=%-6s | val F1=%.4f  P=%.4f  R=%.4f",
                 C, row["val_f1"], row["val_precision"], row["val_recall"])
        models[C] = model

    # Bước 3: áp quy tắc dung sai
    f1_max = max(r["val_f1"] for r in rows)
    threshold = f1_max - lcfg["selection_tolerance"]
    best_row = min((r for r in rows if r["val_f1"] >= threshold), key=lambda r: r["C"])
    best_model = models[best_row["C"]]
    for r in rows:
        r["within_tolerance"] = r["val_f1"] >= threshold
        r["selected"] = r["C"] == best_row["C"]
    log.info("F1 max=%.4f | ngưỡng=%.4f | C đạt ngưỡng: %s", f1_max, threshold,
             [r["C"] for r in rows if r["within_tolerance"]])

    # Bước 4: lưu model được chọn và bảng dò tham số
    base_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, base_dir / "lr_model.joblib")
    pd.DataFrame(rows).to_csv(metrics_dir / "lr_val_search.csv", index=False)
    log.info("Chọn C=%s (val F1=%.4f)", best_row["C"], best_row["val_f1"])
    log.info("Đã lưu -> %s và %s",
             base_dir / "lr_model.joblib", metrics_dir / "lr_val_search.csv")


if __name__ == "__main__":
    main()
