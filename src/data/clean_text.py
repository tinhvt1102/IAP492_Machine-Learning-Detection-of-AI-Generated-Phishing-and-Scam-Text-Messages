"""clean_text.py - Làm sạch nhẹ + chuẩn hoá thực thể + bỏ spam + xoá trùng.

Input : data/interim/merged.csv
Output: data/interim/cleaned_full.csv (chưa cân bằng; balance_data.py sẽ xử lý tiếp)
Không lowercase, không xoá dấu câu/stopwords.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\ufffd]")
WS_RE = re.compile(r"\s+")

# Thực thể thật trong Mishra & Soni (và vài chỗ còn sót ở IMC25) -> placeholder
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_RE = re.compile(
    r"(?:https?://|www\.)\S*"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|co\.uk|info|biz|uk|in|ly|me|us|ac)\b(?:/\S*)?",
    re.IGNORECASE,
)
# >= 9 chữ số (cho phép khoảng trắng/gạch ngang giữa các số), không dính tiền tệ hay chữ
PHONE_RE = re.compile(r"(?<![\w£$€.,])\+?\d(?:[\s\-]?\d){8,}(?!\w)")
ANY_PLACEHOLDER_RE = re.compile(r"<[A-Z_]{2,}>")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fix_encoding(text):
    """Sửa mojibake: thử latin-1 -> utf-8; nếu lỗi thì đổi ký tự vùng C1 theo cp1252."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return "".join(
            bytes([ord(c)]).decode("cp1252", errors="replace") if "\x80" <= c <= "\x9f" else c
            for c in text
        )


def normalize_text(text):
    """Ký tự điều khiển và U+FFFD -> khoảng trắng, gộp khoảng trắng, strip."""
    return WS_RE.sub(" ", CTRL_RE.sub(" ", text)).strip()


def normalize_entities(text, ph, drop_other):
    """Đưa URL/email/số điện thoại về cùng placeholder cho cả 2 nguồn."""
    keep = set(ph.values())
    if drop_other:
        # Xoá placeholder chỉ có ở IMC25 (không thể tái tạo cho Mishra & Soni)
        text = ANY_PLACEHOLDER_RE.sub(lambda m: m.group(0) if m.group(0) in keep else " ", text)
    # Bảo vệ placeholder đã có để regex không đụng vào
    text = EMAIL_RE.sub(ph["email"], text)
    text = URL_RE.sub(ph["url"], text)
    text = PHONE_RE.sub(ph["phone"], text)
    return WS_RE.sub(" ", text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["clean"]
    in_path, out_path = Path(cfg["paths"]["interim_merged"]), Path(cfg["paths"]["interim_cleaned_full"])
    if "raw" in out_path.parts:
        raise ValueError("Không được ghi output vào thư mục raw")

    df = pd.read_csv(in_path, keep_default_na=False, na_values=[""], dtype={"scam_type": "string"})
    df = df[df["text"].notna()]
    print(f"[INFO] Đọc {len(df)} bản ghi từ {in_path}")

    # (1) loại nhãn không dùng (spam của Mishra & Soni)
    n = len(df)
    df = df[~df["label_original"].isin(ccfg["drop_labels"])].copy()
    print(f"[INFO] Loại nhãn {ccfg['drop_labels']}: -{n - len(df)}")

    # (2)(3) sửa encoding + chuẩn hoá nhẹ
    orig = df["text"].copy()
    df["text"] = df["text"].map(fix_encoding).map(normalize_text)
    print(f"[INFO] Text thay đổi sau sửa encoding/chuẩn hoá nhẹ: {(orig != df['text']).sum()}")

    # (4) chuẩn hoá thực thể (URL/email/phone) cho cả 2 nguồn
    before = df["text"].copy()
    df["text"] = df["text"].map(lambda t: normalize_entities(t, ccfg["placeholders"], ccfg["drop_other_placeholders"]))
    print(f"[INFO] Text thay đổi sau chuẩn hoá thực thể: {(before != df['text']).sum()}")

    # (5) bỏ text rỗng
    n = len(df)
    df = df[df["text"].str.len() > 0]
    print(f"[INFO] Text rỗng: -{n - len(df)}")

    # (6a) text trùng nhưng khác nhãn -> bỏ hết
    nlab = df.groupby("text")["label_binary"].nunique()
    conflict = nlab[nlab > 1].index
    n = len(df)
    df = df[~df["text"].isin(conflict)]
    print(f"[INFO] Trùng text khác nhãn: -{n - len(df)}")

    # (6b) text trùng cùng nhãn -> giữ bản đầu
    n = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    print(f"[INFO] Trùng text cùng nhãn: -{n - len(df)}")

    df["label_binary"] = df["label_binary"].astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Ghi {len(df)} bản ghi -> {out_path}")
    print(df.groupby(["source", "label_binary"]).size().to_string())


if __name__ == "__main__":
    main()
