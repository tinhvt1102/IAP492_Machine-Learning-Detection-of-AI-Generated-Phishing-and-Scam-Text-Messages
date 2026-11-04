"""predict_message.py - Chạy thử các mô hình đã train trên tin nhắn do bạn nhập (nhãn 1 = lừa đảo, 0 = hợp lệ).

Ba cách đưa tin vào:
  --text "tin 1" "tin 2"     một hoặc nhiều tin ngay trên dòng lệnh
  --file messages.txt         mỗi dòng một tin (dòng trống và dòng bắt đầu bằng # bị bỏ)
  --interactive               gõ từng tin, Enter để dự đoán; gõ dòng trống hoặc 'exit' để thoát
Tin được chuẩn hoá giống dữ liệu train (URL/số điện thoại/email -> <URL>/<PHONE_NUMBER>/<EMAIL_ADDRESS>)
trước khi đưa vào mô hình. Ngưỡng quyết định là 0,5 (như lúc đánh giá). Mô hình chỉ train trên tiếng Anh.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clean_text import fix_encoding, normalize_entities, normalize_text  # noqa: E402
from src.evaluate.evaluate_models import find_vectorizer  # noqa: E402

MODEL_NAMES = ["lr", "mnb", "roberta", "deberta"]
DISPLAY = {"lr": "LR", "mnb": "MNB", "roberta": "RoBERTa", "deberta": "DeBERTa"}
THRESHOLD = 0.5


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(text, ccfg):
    """Chuẩn hoá đúng như bước làm sạch dữ liệu train."""
    t = normalize_text(fix_encoding(text))
    return normalize_entities(t, ccfg["placeholders"], ccfg["drop_other_placeholders"])


class Predictor:
    """Nạp các mô hình được chọn một lần, rồi dự đoán xác suất lừa đảo cho nhiều tin."""

    def __init__(self, model_cfg, names):
        self.names, self.paths = [], model_cfg["paths"]
        self.max_length = model_cfg["transformer"]["max_length"]
        self.loaded = {}
        base = self.paths["baseline_dir"]
        for n in names:
            try:
                if n in ("lr", "mnb"):
                    import joblib
                    if "vec" not in self.loaded:
                        self.loaded["vec"] = joblib.load(find_vectorizer(base))
                    self.loaded[n] = joblib.load(Path(base) / f"{n}_model.joblib")
                else:
                    self.loaded[n] = self._load_transformer(Path(self.paths[f"{n}_dir"]) / "best")
                self.names.append(n)
            except Exception as e:  # thiếu file/thư viện: bỏ qua mô hình đó nhưng vẫn chạy các mô hình còn lại
                print(f"[WARN] Bỏ qua {DISPLAY[n]}: {type(e).__name__}: {e}")
        if not self.names:
            raise SystemExit("[ERROR] Không nạp được mô hình nào.")

    @staticmethod
    def _load_transformer(model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Không thấy thư mục {model_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).float().to(device).eval()
        return tok, model, device

    def proba(self, name, texts, batch_size=32):
        """Xác suất lớp 'lừa đảo' của từng tin."""
        if name in ("lr", "mnb"):
            return self.loaded[name].predict_proba(self.loaded["vec"].transform(texts))[:, 1].tolist()
        import torch
        tok, model, device = self.loaded[name]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt").to(device)
                out += torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().tolist()
        return out

    def run(self, raw_texts, ccfg):
        norm = [normalize(t, ccfg) for t in raw_texts]
        rows = [{"tin_goc": r, "sau_chuan_hoa": n} for r, n in zip(raw_texts, norm)]
        for name in self.names:
            for row, p in zip(rows, self.proba(name, norm)):
                row[f"{name}_p_lua_dao"] = p
                row[f"{name}_nhan"] = int(p >= THRESHOLD)
        return rows


def show(rows, names):
    for r in rows:
        print("-" * 78)
        print(f"Tin          : {r['tin_goc']}")
        if r["sau_chuan_hoa"] != r["tin_goc"]:
            print(f"Sau chuẩn hoá: {r['sau_chuan_hoa']}")
        for n in names:
            lab = "LỪA ĐẢO" if r[f"{n}_nhan"] else "hợp lệ"
            print(f"  {DISPLAY[n]:8s}: {lab:8s} (xác suất lừa đảo = {r[f'{n}_p_lua_dao']:.3f})")
        k = sum(r[f"{n}_nhan"] for n in names)
        print(f"  => {k}/{len(names)} mô hình báo lừa đảo")


def read_messages(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/model_config.yaml")
    ap.add_argument("--data_config", required=True, help="configs/data_config.yaml (lấy quy tắc chuẩn hoá)")
    ap.add_argument("--text", nargs="*", default=None, help="Một hoặc nhiều tin, đặt trong dấu nháy")
    ap.add_argument("--file", default=None, help="File txt, mỗi dòng một tin")
    ap.add_argument("--interactive", action="store_true", help="Nhập từng tin từ bàn phím")
    ap.add_argument("--models", nargs="*", choices=MODEL_NAMES, default=MODEL_NAMES,
                    help="Mô hình dùng (mặc định cả 4; bỏ roberta deberta nếu chưa cài torch)")
    ap.add_argument("--out", default=None, help="Ghi kết quả ra file csv (với --text/--file)")
    args = ap.parse_args()
    if not (args.text or args.file or args.interactive):
        ap.error("Cần một trong --text, --file hoặc --interactive")

    ccfg = load_yaml(args.data_config)["clean"]
    pred = Predictor(load_yaml(args.config), args.models)
    print(f"[INFO] Mô hình đã nạp: {[DISPLAY[n] for n in pred.names]} | ngưỡng {THRESHOLD}")

    texts = (args.text or []) + (read_messages(args.file) if args.file else [])
    if texts:
        rows = pred.run(texts, ccfg)
        show(rows, pred.names)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
            print(f"\n[INFO] Ghi {len(rows)} dòng -> {args.out}")
    if args.interactive:
        print("Nhập tin nhắn (dòng trống hoặc 'exit' để thoát):")
        while True:
            try:
                t = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not t or t.lower() == "exit":
                break
            show(pred.run([t], ccfg), pred.names)


if __name__ == "__main__":
    main()
