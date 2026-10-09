"""
merge_labels.py
Nhiệm vụ DUY NHẤT: đọc 2 dataset thô (SmishTank, Mishra & Soni), chuẩn hoá nhãn
theo quy ước dự án, thêm cột `source`, và ghi ra data/interim/merged.csv.

Quy ước nhãn:
    smishing (cả 2 nguồn) -> label_binary = 1
    ham                   -> label_binary = 0
    spam                  -> label_binary để trống (NA), label_original = "spam"

KHÔNG làm sạch text, KHÔNG xoá trùng lặp, KHÔNG chia tập (việc của các script khác).
data/raw/ chỉ được ĐỌC, không bao giờ ghi.

Chạy từ thư mục gốc project:
    python src/data/merge_labels.py --config configs/data_config.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Tên cột đầu ra thống nhất cho mọi bước sau
OUT_COLUMNS = ["text", "label_original", "label_binary", "source"]

# Ánh xạ nhãn chuẩn hoá -> nhãn nhị phân (spam cố ý = NA)
BINARY_MAP = {"smishing": 1, "ham": 0, "spam": pd.NA}


def load_config(path: str) -> dict:
    """Đọc file YAML cấu hình."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_filled(cfg: dict, keys: list, section: str) -> None:
    """Dừng chương trình nếu còn giá trị TODO/rỗng trong config
    -> tránh chạy với tên cột/tên file đoán mò."""
    for k in keys:
        v = cfg.get(k)
        if v in (None, "", "TODO"):
            raise ValueError(f"Config '{section}.{k}' chưa được điền (đang là {v!r}).")


def read_table(file_path: Path, read_kwargs: dict) -> pd.DataFrame:
    """Đọc file theo phần mở rộng. Chỉ đọc, không sửa file gốc."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path, **read_kwargs)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(file_path, sep="\t", **read_kwargs)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path, **read_kwargs)
    if ext == ".json":
        return pd.read_json(file_path, **read_kwargs)
    raise ValueError(f"Định dạng chưa hỗ trợ: {ext}")


def require_columns(df: pd.DataFrame, cols: list, name: str) -> None:
    """Kiểm tra cột khai báo trong config có thật trong file không."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] thiếu cột {missing}. Cột thực tế: {list(df.columns)}")


def load_smishtank(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """Bước 1: SmishTank — mọi dòng là smishing."""
    check_filled(cfg, ["file_name", "text_column"], "merge.smishtank")
    df = read_table(raw_dir / cfg["file_name"], cfg.get("read_kwargs") or {})
    log.info("Đọc SmishTank: %d bản ghi từ %s", len(df), raw_dir / cfg["file_name"])
    require_columns(df, [cfg["text_column"]], "smishtank")

    # Chỉ giữ cột text, gán nhãn gốc chuẩn hoá = "smishing"
    out = pd.DataFrame({"text": df[cfg["text_column"]], "label_original": "smishing"})
    out["source"] = "smishtank"
    return out


def load_mishra_soni(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """Bước 2: Mishra & Soni — ánh xạ nhãn gốc về ham/spam/smishing."""
    check_filled(cfg, ["file_name", "text_column", "label_column"], "merge.mishra_soni")
    check_filled(cfg.get("label_values") or {}, ["ham", "spam", "smishing"],
                 "merge.mishra_soni.label_values")
    df = read_table(raw_dir / cfg["file_name"], cfg.get("read_kwargs") or {})
    log.info("Đọc Mishra & Soni: %d bản ghi từ %s", len(df), raw_dir / cfg["file_name"])
    require_columns(df, [cfg["text_column"], cfg["label_column"]], "mishra_soni")

    # Đảo ánh xạ: giá trị trong file gốc -> tên nhãn chuẩn hoá.
    # Mỗi nhãn chuẩn có thể ứng với 1 giá trị hoặc 1 list giá trị gốc
    # (vd file gốc có cả "Smishing" và "smishing").
    raw_to_std = {}
    for std, raw_vals in cfg["label_values"].items():
        for v in (raw_vals if isinstance(raw_vals, list) else [raw_vals]):
            raw_to_std[str(v)] = std
    raw_labels = df[cfg["label_column"]].astype(str)

    # Không bỏ qua âm thầm nhãn lạ: báo lỗi để người dùng kiểm tra
    unknown = set(raw_labels.unique()) - set(raw_to_std)
    if unknown:
        raise ValueError(f"[mishra_soni] nhãn chưa khai báo trong config: {sorted(unknown)}")

    out = pd.DataFrame({"text": df[cfg["text_column"]],
                        "label_original": raw_labels.map(raw_to_std)})
    out["source"] = "mishra_soni"
    log.info("Mishra & Soni theo nhãn: %s", out["label_original"].value_counts().to_dict())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Gộp nhãn 2 dataset thô -> merged.csv")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--output", default=None, help="Ghi đè đường dẫn output trong config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, mcfg = cfg["paths"], cfg["merge"]
    out_path = Path(args.output or paths["interim_merged"])

    # Chặn ghi nhầm vào data/raw/
    if "raw" in out_path.resolve().parts:
        raise ValueError("Không được ghi output vào data/raw/.")

    # Bước 1 + 2: đọc từng nguồn
    smish = load_smishtank(Path(paths["raw_smishtank_dir"]), mcfg["smishtank"])
    mishra = load_mishra_soni(Path(paths["raw_mishra_soni_dir"]), mcfg["mishra_soni"])

    # Bước 3: nối 2 nguồn
    merged = pd.concat([smish, mishra], ignore_index=True)

    # Bước 4: tạo nhãn nhị phân (spam -> NA, dùng kiểu Int64 để chứa NA)
    merged["label_binary"] = merged["label_original"].map(BINARY_MAP).astype("Int64")
    merged = merged[OUT_COLUMNS]

    # Bước 5: chỉ cảnh báo text rỗng, KHÔNG xoá (việc của clean_text.py)
    n_empty = merged["text"].isna().sum()
    if n_empty:
        log.warning("Có %d bản ghi text rỗng (giữ nguyên, xử lý ở clean_text.py)", n_empty)

    # Bước 6: ghi file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    log.info("Tổng theo source: %s", merged["source"].value_counts().to_dict())
    log.info("Tổng theo label_binary: %s",
             merged["label_binary"].value_counts(dropna=False).to_dict())
    log.info("Đã ghi %d bản ghi -> %s", len(merged), out_path)


if __name__ == "__main__":
    main()
