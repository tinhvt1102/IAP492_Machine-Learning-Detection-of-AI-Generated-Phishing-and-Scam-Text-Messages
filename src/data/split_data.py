"""split_data.py - Chia train/val/test THEO NHÓM (tin cùng "khuôn" không nằm ở 2 tập khác nhau).

Input : data/interim/cleaned.csv (output của balance_data.py)
Output: data/processed/phishing_detection/{train,val,test}.csv
Khuôn của tin = viết thường, bỏ placeholder (<URL>...), bỏ chữ số và dấu câu, gộp khoảng trắng.
Các tin cùng khuôn = 1 nhóm; mỗi nhóm được xếp nguyên vào 1 tập.
Cân theo (label_binary, source) theo cách xấp xỉ (greedy), seed cố định.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PLACEHOLDER_RE = re.compile(r"<[A-Za-z_]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")  # sau khi bỏ placeholder: bỏ số và dấu câu, chỉ giữ chữ cái
WS_RE = re.compile(r"\s+")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def template_of(text):
    """Khuôn của tin: bỏ khác biệt về hoa/thường, placeholder, chữ số, dấu câu, khoảng trắng."""
    t = PLACEHOLDER_RE.sub(" ", text.lower())
    return WS_RE.sub(" ", NON_ALPHA_RE.sub(" ", t)).strip()


def assign_groups(df, split_names, ratios, strata_cols, seed):
    """Xếp từng nhóm vào 1 tập sao cho tỉ lệ từng stratum gần mục tiêu nhất."""
    df = df.copy()
    df["_tpl"] = df["text"].map(template_of)
    df["_gid"] = pd.factorize(df["_tpl"])[0]
    df["_stratum"] = df[strata_cols].astype(str).agg("|".join, axis=1)
    strata = sorted(df["_stratum"].unique())
    s_idx = {s: i for i, s in enumerate(strata)}

    # Vector đếm stratum của từng nhóm
    gcount = {}
    for gid, st in zip(df["_gid"], df["_stratum"]):
        gcount.setdefault(gid, np.zeros(len(strata)))[s_idx[st]] += 1
    gids = list(gcount)

    rng = np.random.RandomState(seed)
    rng.shuffle(gids)                       # xáo trước để phá thế hoà
    gids.sort(key=lambda g: -gcount[g].sum())  # nhóm to xếp trước (ổn định nên giữ thứ tự xáo khi bằng nhau)

    total = np.sum(list(gcount.values()), axis=0)
    target = {n: total * r for n, r in zip(split_names, ratios)}
    cur = {n: np.zeros(len(strata)) for n in split_names}
    owner = {}
    for g in gids:
        best, best_cost = None, None
        for n in split_names:
            new = cur[n] + gcount[g]
            # chi phí = mức THAY ĐỔI độ lệch bình phương so với mục tiêu (âm = tiến gần mục tiêu hơn),
            # chuẩn hoá theo cỡ stratum -> nhóm được xếp vào tập còn "thiếu" nhiều nhất
            cost = float(np.sum(((new - target[n]) ** 2 - (cur[n] - target[n]) ** 2) / np.maximum(total, 1)))
            if best_cost is None or cost < best_cost:
                best, best_cost = n, cost
        owner[g] = best
        cur[best] += gcount[g]
    df["_split"] = df["_gid"].map(owner)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["split"]
    in_path = Path(cfg["paths"]["interim_cleaned"])
    out_dir = Path(cfg["paths"]["processed_phishing_dir"])
    if "raw" in out_dir.parts:
        raise ValueError("Không được ghi output vào thư mục raw")

    ratios = [scfg["train_ratio"], scfg["val_ratio"], scfg["test_ratio"]]
    if abs(sum(ratios) - 1) > 1e-9:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1, hiện là {sum(ratios)}")

    df = pd.read_csv(in_path, keep_default_na=False, na_values=[""], dtype={"scam_type": "string"})
    print(f"[INFO] Đọc {len(df)} bản ghi từ {in_path}")

    names = ["train", "val", "test"]
    df = assign_groups(df, names, ratios, scfg["stratify_columns"], scfg["random_seed"])
    n_groups = df["_gid"].nunique()
    sizes = df.groupby("_gid").size().sort_values(ascending=False)
    print(f"[INFO] Số nhóm (khuôn): {n_groups} / {len(df)} tin; nhóm lớn nhất: {sizes.head(5).tolist()}")

    parts = {n: df[df["_split"] == n] for n in names}

    # Kiểm tra không rò rỉ: không text và không nhóm nào nằm ở 2 tập
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(parts[a]["text"]) & set(parts[b]["text"]):
                raise ValueError(f"Có text nằm ở cả {a} và {b}")
            if set(parts[a]["_gid"]) & set(parts[b]["_gid"]):
                raise ValueError(f"Có nhóm nằm ở cả {a} và {b}")

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["text", "label_binary", "source", "scam_type"]
    for n in names:
        p = out_dir / f"{n}.csv"
        parts[n][keep].sample(frac=1, random_state=scfg["random_seed"]).to_csv(p, index=False, encoding="utf-8")
        print(f"[INFO] Ghi {len(parts[n])} bản ghi -> {p}")

    # Thống kê từng tập
    rows = []
    for n in names:
        p = parts[n]
        rows.append({
            "tap": n, "so_tin": len(p), "ti_le": round(len(p) / len(df), 3),
            "%label1": round(100 * p["label_binary"].mean(), 1),
            "mishra_0": int(((p.source == "mishra_soni") & (p.label_binary == 0)).sum()),
            "mishra_1": int(((p.source == "mishra_soni") & (p.label_binary == 1)).sum()),
            "imc25_1": int(((p.source == "imc25") & (p.label_binary == 1)).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
