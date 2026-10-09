"""
train_mnb.py
Nhiệm vụ DUY NHẤT: train Multinomial Naive Bayes trên đặc trưng TF-IDF.

- Đọc X/y train, val từ models/baseline/ (do tfidf_vectorizer.py tạo).
- Dò tham số alpha trên VAL, chấm theo F1 (lớp 1 = phishing/scam).
- Quy tắc chọn (dung sai): trong các alpha có F1 >= F1_max - selection_tolerance,
  chọn alpha LỚN NHẤT (làm mượt nhiều nhất = đơn giản nhất).
  Lưu ý: chiều ngược với C của LR (C nhỏ = đơn giản; alpha lớn = đơn giản).
- MultinomialNB không có tham số class_weight -> dùng sample_weight
  "balanced" (cùng công thức với class_weight='balanced' của LR).
- Model cuối = model fit trên TRAIN với alpha tốt nhất.
- Lưu: models/baseline/mnb_model.joblib
       results/metrics/mnb_val_search.csv
- KHÔNG đánh giá trên test (việc của evaluate_models.py).

Chạy từ thư mục gốc project:
    python src/models/train_mnb.py --config configs/model_config.yaml
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multinomial Naive Bayes (TF-IDF)")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["mnb"]
    base_dir = Path(cfg["paths"]["baseline_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])

    # Bước 1: đọc đặc trưng và nhãn (chỉ train + val, KHÔNG đọc test)
    X_train = sparse.load_npz(base_dir / "X_train.npz")
    y_train = np.load(base_dir / "y_train.npy")
    X_val = sparse.load_npz(base_dir / "X_val.npz")
    y_val = np.load(base_dir / "y_val.npy")
    log.info("Train: %s | Val: %s", X_train.shape, X_val.shape)

    # Bước 2: trọng số mẫu cân bằng lớp (None nếu không dùng)
    sw = (compute_sample_weight("balanced", y_train)
          if mcfg["class_weight"] == "balanced" else None)

    # Bước 3: thử từng alpha: fit trên train, chấm trên val
    rows, models = [], {}
    for alpha in sorted(mcfg["alpha_grid"]):
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train, sample_weight=sw)
        pred = model.predict(X_val)
        row = {"alpha": alpha,
               "val_f1": f1_score(y_val, pred),
               "val_precision": precision_score(y_val, pred, zero_division=0),
               "val_recall": recall_score(y_val, pred)}
        rows.append(row)
        log.info("alpha=%-6s | val F1=%.4f  P=%.4f  R=%.4f",
                 alpha, row["val_f1"], row["val_precision"], row["val_recall"])
        models[alpha] = model

    # Bước 4: áp quy tắc dung sai
    f1_max = max(r["val_f1"] for r in rows)
    threshold = f1_max - mcfg["selection_tolerance"]
    best_row = max((r for r in rows if r["val_f1"] >= threshold), key=lambda r: r["alpha"])
    best_model = models[best_row["alpha"]]
    for r in rows:
        r["within_tolerance"] = r["val_f1"] >= threshold
        r["selected"] = r["alpha"] == best_row["alpha"]
    log.info("F1 max=%.4f | ngưỡng=%.4f | alpha đạt ngưỡng: %s", f1_max, threshold,
             [r["alpha"] for r in rows if r["within_tolerance"]])

    # Bước 5: lưu model được chọn và bảng dò tham số
    base_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, base_dir / "mnb_model.joblib")
    pd.DataFrame(rows).to_csv(metrics_dir / "mnb_val_search.csv", index=False)
    log.info("Chọn alpha=%s (val F1=%.4f)", best_row["alpha"], best_row["val_f1"])
    log.info("Đã lưu -> %s và %s",
             base_dir / "mnb_model.joblib", metrics_dir / "mnb_val_search.csv")


if __name__ == "__main__":
    main()
