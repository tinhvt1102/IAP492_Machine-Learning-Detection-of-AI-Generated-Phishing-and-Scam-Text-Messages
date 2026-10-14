"""balance_data.py - Lấy mẫu nguồn nhãn 1 lớn (IMC25) để tỉ lệ nhãn 1 : nhãn 0 theo config.

Input : data/interim/cleaned_full.csv
Output: data/interim/cleaned.csv  (input của split_data.py)
Giữ nguyên toàn bộ nhãn 0 và nhãn 1 của các nguồn khác; chỉ lấy mẫu nguồn `subsample_source`,
phân tầng theo `stratify_by` (giữ tỉ lệ scam_type), seed cố định.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def allocate(counts, n_total):
    """Chia n_total cho các nhóm tỉ lệ theo counts (largest remainder), không vượt quá size nhóm."""
    exact = counts / counts.sum() * n_total
    alloc = exact.astype(int)
    rest = n_total - alloc.sum()
    order = (exact - alloc).sort_values(ascending=False).index
    for g in order[:rest]:
        alloc[g] += 1
    return alloc.clip(upper=counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["balance"]
    in_path, out_path = Path(cfg["paths"]["interim_cleaned_full"]), Path(cfg["paths"]["interim_cleaned"])
    if "raw" in out_path.parts:
        raise ValueError("Không được ghi output vào thư mục raw")

    df = pd.read_csv(in_path, keep_default_na=False, na_values=[""], dtype={"scam_type": "string"})
    print(f"[INFO] Đọc {len(df)} bản ghi từ {in_path}")

    n0 = int((df["label_binary"] == 0).sum())
    target_n1 = round(n0 * bcfg["target_ratio_label1"])
    pool = df[(df["label_binary"] == 1) & (df["source"] == bcfg["subsample_source"])]
    fixed = df.drop(pool.index)
    need = target_n1 - int((fixed["label_binary"] == 1).sum())
    print(f"[INFO] nhãn 0 = {n0}; mục tiêu nhãn 1 = {target_n1}; nhãn 1 cố định (nguồn khác) = {target_n1 - need}; "
          f"cần lấy từ {bcfg['subsample_source']}: {need} / có {len(pool)}")
    if need <= 0:
        raise ValueError("Nhãn 1 từ nguồn khác đã đủ/vượt mục tiêu; kiểm tra lại target_ratio_label1")
    if need > len(pool):
        print(f"[WARN] Không đủ {bcfg['subsample_source']} ({len(pool)} < {need}); lấy hết, tỉ lệ sẽ lệch mục tiêu")
        need = len(pool)

    # Lấy mẫu phân tầng theo scam_type
    col = bcfg["stratify_by"]
    counts = pool[col].value_counts()
    alloc = allocate(counts, need)
    parts = [pool[pool[col] == g].sample(n=int(k), random_state=bcfg["random_seed"]) for g, k in alloc.items() if k > 0]
    sampled = pd.concat(parts)

    out = pd.concat([fixed, sampled]).sample(frac=1, random_state=bcfg["random_seed"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Ghi {len(out)} bản ghi -> {out_path}")
    print(out.groupby(["source", "label_binary"]).size().to_string())
    print(f"[INFO] Tỉ lệ nhãn 1 = {out['label_binary'].mean():.3f}")
    cmp = pd.DataFrame({"truoc": counts, "sau": sampled[col].value_counts()}).fillna(0).astype(int)
    print("[INFO] scam_type trước/sau lấy mẫu:\n" + cmp.to_string())


if __name__ == "__main__":
    main()
