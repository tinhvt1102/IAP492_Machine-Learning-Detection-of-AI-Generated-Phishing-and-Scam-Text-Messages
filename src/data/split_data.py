"""
split_data.py
Nhiệm vụ DUY NHẤT: chia data/interim/cleaned.csv thành train/val/test
-> data/processed/phishing_detection/{train,val,test}.csv

- Tỉ lệ, seed, cột stratify đọc từ configs/data_config.yaml (mục `split`).
- Stratify theo tổ hợp (label_binary, source) để tỉ lệ nhãn và nguồn
  giống nhau ở cả 3 tập (cần cho việc báo cáo metric tách theo nguồn).
- KHÔNG làm sạch thêm, KHÔNG biến đổi text.

Chạy từ thư mục gốc project:
    python src/data/split_data.py --config configs/data_config.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def summarize(name: str, df: pd.DataFrame) -> None:
    """In số bản ghi và phân bố (source, label) của một tập."""
    dist = df.groupby(["source", "label_binary"]).size().to_dict()
    pos = df["label_binary"].mean() * 100
    log.info("%-5s: %5d bản ghi | %%label=1: %.1f%% | %s", name, len(df), pos, dist)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chia cleaned.csv -> train/val/test")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--input", default=None, help="Ghi đè đường dẫn input")
    parser.add_argument("--output_dir", default=None, help="Ghi đè thư mục output")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scfg = cfg["split"]
    in_path = Path(args.input or cfg["paths"]["interim_cleaned"])
    out_dir = Path(args.output_dir or cfg["paths"]["processed_phishing_dir"])

    # Kiểm tra tỉ lệ hợp lệ (tổng = 1)
    ratios = (scfg["train_ratio"], scfg["val_ratio"], scfg["test_ratio"])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1, đang là {sum(ratios)}")

    # Bước 1: đọc dữ liệu đã làm sạch
    df = pd.read_csv(in_path, dtype={"text": str}, keep_default_na=False)
    log.info("Đọc %d bản ghi từ %s", len(df), in_path)

    # Bước 2: tạo khoá stratify từ tổ hợp các cột (vd "1|smishtank")
    strat_key = df[scfg["stratify_columns"]].astype(str).agg("|".join, axis=1)

    # Bước 3: tách test ra trước
    trainval, test = train_test_split(
        df, test_size=scfg["test_ratio"], stratify=strat_key,
        random_state=scfg["random_seed"])

    # Bước 4: tách val từ phần còn lại.
    # val_ratio tính trên toàn bộ dữ liệu -> quy đổi sang tỉ lệ trên phần trainval
    val_size = scfg["val_ratio"] / (scfg["train_ratio"] + scfg["val_ratio"])
    train, val = train_test_split(
        trainval, test_size=val_size, stratify=strat_key.loc[trainval.index],
        random_state=scfg["random_seed"])

    # Bước 5: kiểm tra không có text nào nằm ở 2 tập (chống rò rỉ dữ liệu)
    sets = {"train": set(train["text"]), "val": set(val["text"]), "test": set(test["text"])}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = len(sets[a] & sets[b])
        if overlap:
            raise ValueError(f"Rò rỉ: {overlap} text trùng giữa {a} và {b}")

    # Bước 6: ghi file và log phân bố
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8")
        summarize(name, part)
        log.info("       -> %s", path)
    log.info("Tổng: %d bản ghi (khớp input: %s)",
             len(train) + len(val) + len(test), len(train) + len(val) + len(test) == len(df))


if __name__ == "__main__":
    main()
