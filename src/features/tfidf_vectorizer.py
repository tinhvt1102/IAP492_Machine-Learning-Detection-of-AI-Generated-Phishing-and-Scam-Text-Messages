"""
tfidf_vectorizer.py
Nhiệm vụ DUY NHẤT: tạo đặc trưng TF-IDF cho 2 model baseline (LR, MNB).

- Fit TF-IDF CHỈ trên train (tránh rò rỉ từ vựng của val/test).
- Transform train/val/test bằng cùng vectorizer.
- Lưu vào models/baseline/:
    tfidf_vectorizer.joblib          vectorizer đã fit
    X_{train,val,test}.npz           ma trận TF-IDF (sparse)
    y_{train,val,test}.npy           nhãn 0/1 tương ứng
- KHÔNG train model (việc của train_lr.py / train_mnb.py).

Chạy từ thư mục gốc project:
    python src/features/tfidf_vectorizer.py --config configs/model_config.yaml
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TF-IDF trên train, transform 3 tập")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["tfidf"]
    in_dir = Path(cfg["paths"]["processed_phishing_dir"])
    out_dir = Path(cfg["paths"]["baseline_dir"])

    # Bước 1: đọc 3 tập đã chia
    data = {}
    for s in SPLITS:
        path = in_dir / f"{s}.csv"
        data[s] = pd.read_csv(path, dtype={tcfg["text_column"]: str}, keep_default_na=False)
        log.info("Đọc %-5s: %d bản ghi từ %s", s, len(data[s]), path)

    # Bước 2: khởi tạo TF-IDF theo cấu hình nhóm đã chốt
    vectorizer = TfidfVectorizer(
        lowercase=tcfg["lowercase"],
        analyzer=tcfg["analyzer"],
        ngram_range=tuple(tcfg["ngram_range"]),
        min_df=tcfg["min_df"],
        max_features=tcfg["max_features"],
        sublinear_tf=tcfg["sublinear_tf"],
    )

    # Bước 3: fit CHỈ trên train, rồi transform train
    X = {"train": vectorizer.fit_transform(data["train"][tcfg["text_column"]])}
    log.info("Fit trên train: từ vựng = %d đặc trưng", len(vectorizer.vocabulary_))

    # Bước 4: transform val/test bằng vectorizer đã fit (không fit lại)
    for s in ["val", "test"]:
        X[s] = vectorizer.transform(data[s][tcfg["text_column"]])

    # Bước 5: lưu vectorizer, ma trận và nhãn
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    for s in SPLITS:
        sparse.save_npz(out_dir / f"X_{s}.npz", X[s])
        np.save(out_dir / f"y_{s}.npy", data[s][tcfg["label_column"]].to_numpy(dtype=int))
        # Tin không chứa từ nào trong từ vựng -> vector toàn 0 (chỉ báo, không xoá)
        n_empty = int((X[s].getnnz(axis=1) == 0).sum())
        log.info("%-5s: X shape = %s | vector rỗng: %d", s, X[s].shape, n_empty)
    log.info("Đã lưu vectorizer + X/y -> %s", out_dir)


if __name__ == "__main__":
    main()
