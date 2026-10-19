"""prepare_ai_testset.py - Chuẩn hoá tập tin nhắn do AI viết thành cùng định dạng với tập test của người.

Input : <robustness.raw_dir>/{scam_file,ham_file}   (mỗi dòng "loai|noi dung"; dòng bắt đầu bằng # bị bỏ)
Output: <robustness.output_dir>/test.csv  (cột: text, label_binary, source, scam_type)
Áp dụng ĐÚNG các hàm chuẩn hoá của clean_text.py (sửa mã hoá, chuẩn hoá thực thể -> <URL>/<PHONE_NUMBER>/<EMAIL_ADDRESS>)
để khác biệt so với tập người chỉ nằm ở phong cách viết do AI, không phải ở cách tiền xử lý.
Kiểm tra rò rỉ: bỏ tin trùng text hoặc trùng "khuôn" với train/val/test của người.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clean_text import fix_encoding, normalize_entities, normalize_text  # noqa: E402
from src.data.split_data import template_of  # noqa: E402

ENTITY_PAT = r"<(?:URL|EMAIL_ADDRESS|PHONE_NUMBER)>"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw(path, label):
    """Đọc file thô: 'loai|noi dung' theo từng dòng."""
    rows = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.count("|") < 1:
            raise ValueError(f"{path} dòng {ln}: thiếu dấu '|' (định dạng 'loai|noi dung')")
        kind, text = line.split("|", 1)
        rows.append({"text": text.strip(), "label_binary": label, "scam_type": kind.strip()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/data_config.yaml")
    ap.add_argument("--raw_dir", default=None, help="Ghi đè robustness.raw_dir (ví dụ data/raw/ai_generated_v2)")
    ap.add_argument("--output_dir", default=None, help="Ghi đè robustness.output_dir (ví dụ data/processed/ai_generated_v2)")
    ap.add_argument("--source_name", default=None, help="Ghi đè robustness.source_name (tên LLM sinh dữ liệu, ví dụ ai_chatgpt)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rcfg, ccfg = cfg["robustness"], cfg["clean"]
    raw_dir, out_dir = Path(args.raw_dir or rcfg["raw_dir"]), Path(args.output_dir or rcfg["output_dir"])
    if "raw" in out_dir.parts:
        raise ValueError("Không được ghi output vào thư mục raw")

    df = pd.DataFrame(read_raw(raw_dir / rcfg["scam_file"], 1) + read_raw(raw_dir / rcfg["ham_file"], 0))
    print(f"[INFO] Đọc {len(df)} tin thô (lừa đảo {int((df.label_binary == 1).sum())}, hợp lệ {int((df.label_binary == 0).sum())})")

    # chuẩn hoá giống clean_text.py
    df["text"] = df["text"].map(fix_encoding).map(normalize_text)
    df["text"] = df["text"].map(lambda t: normalize_entities(t, ccfg["placeholders"], ccfg["drop_other_placeholders"]))
    df = df[df["text"].str.len() > 0]

    # trùng nội bộ
    n = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    print(f"[INFO] Trùng nội bộ: -{n - len(df)}")

    # rò rỉ với dữ liệu người (train/val/test)
    human_dir = Path(cfg["paths"]["processed_phishing_dir"])
    human = pd.concat([pd.read_csv(human_dir / f"{s}.csv", keep_default_na=False) for s in ("train", "val", "test")])
    h_text, h_tpl = set(human["text"]), set(human["text"].map(template_of))
    leak_exact = df["text"].isin(h_text)
    leak_tpl = df["text"].map(template_of).isin(h_tpl)
    print(f"[INFO] Trùng text với dữ liệu người: {int(leak_exact.sum())}; trùng khuôn: {int(leak_tpl.sum())} (sẽ loại)")
    df = df[~leak_tpl].copy()

    src_name = args.source_name or rcfg["source_name"]
    df["source"] = src_name
    df = df[["text", "label_binary", "source", "scam_type"]].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")
    print(f"[INFO] Ghi {len(df)} tin -> {out_dir / 'test.csv'}")

    # thống kê so với tập test của người
    h_test = pd.read_csv(human_dir / "test.csv", keep_default_na=False)

    def stats(d, name):
        g = d.assign(has_entity=d["text"].str.contains(ENTITY_PAT), n_chars=d["text"].str.len()).groupby("label_binary")
        out = pd.DataFrame({"so_tin": g.size(), "do_dai_trung_vi": g["n_chars"].median(),
                            "ti_le_co_thuc_the_%": (100 * g["has_entity"].mean()).round(1)})
        print(f"\n[{name}]\n{out.to_string()}")

    stats(df, f"AI ({src_name})")
    stats(h_test, "NGƯỜI (test)")
    print("\n[AI] số tin theo loại:\n" + df.groupby(["label_binary", "scam_type"]).size().to_string())


if __name__ == "__main__":
    main()
