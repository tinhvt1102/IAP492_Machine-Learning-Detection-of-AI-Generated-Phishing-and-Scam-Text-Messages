"""
clean_text.py
Nhiệm vụ DUY NHẤT: làm sạch nhẹ data/interim/merged.csv -> data/interim/cleaned.csv

Các bước (theo thứ tự):
  1. Loại các nhãn trong config `clean.drop_labels` (mặc định: spam — phương án a)
  2. Sửa lỗi encoding (mojibake), vd "â\x80\x99" -> "’", "\x92" -> "’"
  3. Bỏ ký tự điều khiển (\t, \n...) và ký tự lỗi "�", gộp khoảng trắng thừa
  4. Bỏ dòng text rỗng sau khi làm sạch
  5. Xử lý trùng lặp: text giống nhau nhưng nhãn khác nhau -> bỏ hết;
     text giống nhau cùng nhãn -> giữ 1 bản
  6. Ép label_binary về int (0/1)

CỐ Ý KHÔNG làm: lowercase, xoá URL/số điện thoại/dấu câu, xoá stopwords
(đây là tín hiệu phishing; baseline lowercase riêng trong tfidf_vectorizer.py).
KHÔNG chia train/val/test (việc của split_data.py).

Chạy từ thư mục gốc project:
    python src/data/clean_text.py --config configs/data_config.yaml
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ký tự điều khiển (C0, DEL, C1) -> thay bằng khoảng trắng
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# Ký tự thay thế "�": thông tin gốc đã mất trong file raw, không khôi phục được
REPLACEMENT_CHAR = "\ufffd"
WHITESPACE_RE = re.compile(r"\s+")


def fix_encoding(text: str) -> str:
    """Sửa mojibake do file bị đọc sai bảng mã.
    a) Text UTF-8 bị đọc thành latin-1 (vd emoji -> 'ð\x9f...', ’ -> 'â\x80\x99'):
       mã hoá lại latin-1 rồi giải mã UTF-8. Chỉ áp dụng nếu giải mã thành công.
    b) Ký tự Windows-1252 nằm ở vùng C1 (vd '\x92' là dấu ’): đổi đúng ký tự.
    Text bình thường (vd '£', 'Ü') không bị thay đổi."""
    if not any("\x80" <= ch <= "\xff" for ch in text):
        return text                                  # không có ký tự nghi vấn
    try:
        return text.encode("latin-1").decode("utf-8")      # trường hợp (a)
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    out = []
    for ch in text:                                         # trường hợp (b)
        if "\x80" <= ch <= "\x9f":
            try:
                ch = ch.encode("latin-1").decode("cp1252")
            except UnicodeDecodeError:
                pass                       # byte không có nghĩa -> bước 3 sẽ xoá
        out.append(ch)
    return "".join(out)


def normalize_text(text: str) -> str:
    """Chuẩn hoá nhẹ: bỏ ký tự điều khiển + '�', gộp khoảng trắng, strip."""
    text = CONTROL_RE.sub(" ", text)
    text = text.replace(REPLACEMENT_CHAR, " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Làm sạch nhẹ merged.csv -> cleaned.csv")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--input", default=None, help="Ghi đè đường dẫn input")
    parser.add_argument("--output", default=None, help="Ghi đè đường dẫn output")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    in_path = Path(args.input or cfg["paths"]["interim_merged"])
    out_path = Path(args.output or cfg["paths"]["interim_cleaned"])
    drop_labels = cfg.get("clean", {}).get("drop_labels", [])

    # Đọc file; text đọc dạng str để không bị pandas tự đổi kiểu
    df = pd.read_csv(in_path, dtype={"text": str}, keep_default_na=False,
                     na_values={"label_binary": [""]})
    log.info("Đọc %d bản ghi từ %s", len(df), in_path)

    # Bước 1: loại nhãn mơ hồ
    n = len(df)
    df = df[~df["label_original"].isin(drop_labels)].copy()
    log.info("Bước 1 - loại nhãn %s: -%d bản ghi", drop_labels, n - len(df))
    if df["label_binary"].isna().any():
        raise ValueError("Vẫn còn label_binary rỗng sau bước 1 — kiểm tra drop_labels.")

    # Bước 2 + 3: sửa encoding rồi chuẩn hoá nhẹ
    original = df["text"].copy()
    df["text"] = df["text"].map(fix_encoding).map(normalize_text)
    log.info("Bước 2-3 - sửa encoding/chuẩn hoá: %d bản ghi có text thay đổi",
             (original != df["text"]).sum())

    # Bước 4: bỏ text rỗng
    n = len(df)
    df = df[df["text"] != ""]
    log.info("Bước 4 - text rỗng: -%d bản ghi", n - len(df))

    # Bước 5a: text trùng nhưng nhãn mâu thuẫn -> bỏ toàn bộ (không biết nhãn nào đúng)
    n = len(df)
    conflict = df.groupby("text")["label_binary"].transform("nunique") > 1
    df = df[~conflict]
    log.info("Bước 5a - trùng text nhưng khác nhãn: -%d bản ghi", n - len(df))

    # Bước 5b: text trùng cùng nhãn -> giữ bản đầu tiên
    n = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    log.info("Bước 5b - trùng text cùng nhãn: -%d bản ghi", n - len(df))

    # Bước 6: ép nhãn về int và ghi file
    df["label_binary"] = df["label_binary"].astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    log.info("Theo source x label_binary: %s",
             df.groupby(["source", "label_binary"]).size().to_dict())
    log.info("Đã ghi %d bản ghi -> %s", len(df), out_path)


if __name__ == "__main__":
    main()
