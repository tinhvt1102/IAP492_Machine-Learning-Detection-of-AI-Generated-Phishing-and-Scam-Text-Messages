from pathlib import Path
import pandas as pd


def read_any(p):
    """Đọc csv/xlsx/json; thử utf-8 trước, lỗi thì thử latin-1."""
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p), None
    if ext == ".json":
        return pd.read_json(p), None
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc), enc
        except UnicodeDecodeError:
            continue


for d in ["data/raw/smishtank", "data/raw/mishra_soni"]:
    files = [p for p in sorted(Path(d).rglob("*"))
             if p.is_file() and not p.name.startswith(".")]
    if not files:
        print(f"[WARN] Không có file nào trong {Path(d).resolve()}")
        continue
    for p in files:
        print("=" * 60, "\nFILE:", p)
        df, enc = read_any(p)
        print("encoding:", enc, "| shape:", df.shape)
        print("columns:", list(df.columns))
        print(df.head(3).to_string())
        # Cột ít giá trị khác nhau thường là cột nhãn
        for c in df.columns:
            if df[c].nunique() <= 10:
                print(f"unique[{c}]:", df[c].value_counts(dropna=False).to_dict())