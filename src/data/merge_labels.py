"""merge_labels.py - Gộp IMC25 + Mishra & Soni thành 1 bảng, chuẩn hoá nhãn nhị phân.

Output: data/interim/merged.csv  (cột: text, label_original, label_binary, source, scam_type)
Không bao giờ ghi vào data/raw/.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

BINARY_MAP = {"smishing": 1, "ham": 0, "spam": pd.NA}  # spam: để clean_text.py loại


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_source(raw_dir, scfg):
    """Đọc 1 file thô theo config; thiếu file/cột thì dừng (không đoán)."""
    fpath = Path(raw_dir) / scfg["file_name"]
    if not fpath.exists():
        raise FileNotFoundError(f"Không thấy file: {fpath}")
    df = pd.read_csv(fpath, **scfg.get("read_kwargs", {}))
    print(f"[INFO] Đọc {fpath}: {len(df)} bản ghi")
    needed = [scfg["text_column"]] + [c for c in [scfg.get("label_column"), scfg.get("extra_column")] if c]
    needed += list(scfg.get("include", {})) + list(scfg.get("exclude", {}))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"{fpath}: thiếu cột {missing}. Cột hiện có: {list(df.columns)}")
    return df


def prep_imc25(df, scfg):
    # Lọc theo include (ví dụ language == English)
    for col, vals in scfg.get("include", {}).items():
        before = len(df)
        df = df[df[col].isin(vals)]
        print(f"[INFO] imc25 include {col} in {vals}: {before} -> {len(df)}")
    # Loại theo exclude (ví dụ scam_type spam, wrong number)
    for col, vals in scfg.get("exclude", {}).items():
        before = len(df)
        df = df[~df[col].isin(vals)]
        print(f"[INFO] imc25 exclude {col} in {vals}: {before} -> {len(df)} (loại {before - len(df)})")
    df = df[df[scfg["text_column"]].notna()]
    out = pd.DataFrame({
        "text": df[scfg["text_column"]].astype(str),
        "label_original": scfg["fixed_label"],
        "source": "imc25",
        "scam_type": df[scfg["extra_column"]].fillna("unknown"),
    })
    return out


def prep_mishra(df, scfg):
    # Gộp các cách viết nhãn (Smishing/smishing, spam/Spam) về nhãn chuẩn
    lookup = {v: std for std, vals in scfg["label_values"].items() for v in vals}
    raw = df[scfg["label_column"]]
    unknown = sorted(set(raw.unique()) - set(lookup))
    if unknown:
        raise ValueError(f"Nhãn chưa khai báo trong config: {unknown}")
    out = pd.DataFrame({
        "text": df[scfg["text_column"]].astype(str),
        "label_original": raw.map(lookup),
        "source": "mishra_soni",
        "scam_type": pd.NA,
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths, mcfg = cfg["paths"], cfg["merge"]

    out_path = Path(paths["interim_merged"])
    if "raw" in out_path.parts:
        raise ValueError("Không được ghi output vào thư mục raw")

    imc = prep_imc25(read_source(paths["raw_imc25_dir"], mcfg["imc25"]), mcfg["imc25"])
    mis = prep_mishra(read_source(paths["raw_mishra_soni_dir"], mcfg["mishra_soni"]), mcfg["mishra_soni"])

    merged = pd.concat([imc, mis], ignore_index=True)
    merged["label_binary"] = merged["label_original"].map(BINARY_MAP).astype("Int64")
    merged = merged[["text", "label_original", "label_binary", "source", "scam_type"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Ghi {len(merged)} bản ghi -> {out_path}")
    print(merged.groupby(["source", "label_original"]).size().to_string())


if __name__ == "__main__":
    main()
