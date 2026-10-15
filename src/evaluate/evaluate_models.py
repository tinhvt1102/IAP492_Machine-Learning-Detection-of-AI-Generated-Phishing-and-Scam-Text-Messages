"""evaluate_models.py - Đánh giá LR, MNB, RoBERTa, DeBERTa trên tập test (lần đầu mở test).

Hai việc nối tiếp (làm trong cùng 1 script vì cùng phục vụ 1 mục đích: ra bảng kết quả cuối):
  (1) --models ...: chạy model trên test, ghi dự đoán ra <metrics_dir>/preds_<tag>_<model>.csv
  (2) luôn luôn  : đọc mọi file preds_<tag>_*.csv đang có, tính metric và ghi bảng kết quả.
Nhờ vậy có thể chạy LR/MNB trên máy mình, RoBERTa/DeBERTa trên Kaggle, rồi gom file preds về 1 chỗ để làm báo cáo.

KHÔNG chỉnh model/tham số dựa trên kết quả test. --test_csv/--tag cho phép dùng lại script với tập khác
(ví dụ tập tin nhắn do AI sinh ở bước robustness).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

try:  # chạy từ thư mục gốc project: python src/evaluate/evaluate_models.py
    from src.evaluate.metrics_utils import compute_metrics
except ModuleNotFoundError:  # chạy ở nơi khác: thêm thư mục gốc vào sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluate.metrics_utils import compute_metrics

MODEL_NAMES = ["lr", "mnb", "roberta", "deberta"]
DISPLAY = {"lr": "LR", "mnb": "MNB", "roberta": "RoBERTa", "deberta": "DeBERTa"}


# ---------------------------------------------------------------- dự đoán
def predict_baseline(name, texts, baseline_dir):
    """LR/MNB: dùng đúng TF-IDF đã fit trên train, rồi model.predict."""
    import joblib
    vec = joblib.load(Path(baseline_dir) / "tfidf_vectorizer.joblib")
    model = joblib.load(Path(baseline_dir) / f"{name}_model.joblib")
    return model.predict(vec.transform(texts)).astype(int).tolist()


def predict_transformer(model_dir, texts, max_length, batch_size):
    """RoBERTa/DeBERTa: nạp checkpoint tốt nhất (đã chọn bằng val) và dự đoán."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from src.features.tokenizer_utils import get_tokenizer, make_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(str(model_dir))  # tokenizer đã được lưu cùng model
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).float().to(device).eval()
    loader = make_loader(texts, [0] * len(texts), tok, max_length, batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            preds += logits.argmax(dim=-1).cpu().tolist()
    return preds


# ---------------------------------------------------------------- báo cáo
def safe_metrics(y_true, y_pred):
    """Metric an toàn cho tập con: tập chỉ có 1 lớp thì metric không xác định -> NaN (không bịa số)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = compute_metrics(y_true, y_pred)
    has_pos, has_neg = (y_true == 1).any(), (y_true == 0).any()
    if not has_neg:  # chỉ có tin lừa đảo: chỉ Recall (= Accuracy) có nghĩa
        m.update(precision=np.nan, f1=np.nan, fpr=np.nan)
    if not has_pos:  # chỉ có tin hợp lệ: chỉ FPR (= 1 - Accuracy) có nghĩa
        m.update(precision=np.nan, recall=np.nan, f1=np.nan)
    return m


def bootstrap_f1_ci(y_true, y_pred, n_boot=1000, seed=42):
    """Khoảng tin cậy 95% của F1 bằng bootstrap (lấy mẫu lại có hoàn lại)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng, n = np.random.RandomState(seed), len(y_true)
    f1s = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        f1s.append(f1_score(y_true[i], y_pred[i], zero_division=0))
    return np.percentile(f1s, [2.5, 97.5])


def build_reports(test_df, preds, label_col, source_col):
    overall, by_source = [], []
    y = test_df[label_col].astype(int).to_numpy()
    for name, p in preds.items():
        m = compute_metrics(y, p)
        lo, hi = bootstrap_f1_ci(y, p)
        overall.append({"model": DISPLAY[name], "n": len(y), **m, "f1_ci_low": lo, "f1_ci_high": hi})
        for src, g in test_df.groupby(source_col):
            idx = g.index.to_numpy()
            by_source.append({"model": DISPLAY[name], "source": src, "n": len(idx),
                              "n_label1": int((y[idx] == 1).sum()), **safe_metrics(y[idx], np.asarray(p)[idx])})
    return pd.DataFrame(overall), pd.DataFrame(by_source)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/model_config.yaml")
    ap.add_argument("--models", nargs="*", default=[], choices=MODEL_NAMES,
                    help="Model cần chạy dự đoán lần này (bỏ trống = chỉ làm báo cáo từ file preds có sẵn)")
    ap.add_argument("--test_csv", default=None, help="Mặc định: <processed_phishing_dir>/test.csv")
    ap.add_argument("--tag", default="test", help="Nhãn tập đánh giá, dùng đặt tên file output")
    ap.add_argument("--metrics_dir", default=None)
    ap.add_argument("--baseline_dir", default=None)
    ap.add_argument("--roberta_dir", default=None, help="Thư mục chứa best/ của RoBERTa (mặc định theo config)")
    ap.add_argument("--deberta_dir", default=None, help="Thư mục chứa best/ của DeBERTa (mặc định theo config)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths, tcfg = cfg["paths"], cfg["transformer"]
    metrics_dir = Path(args.metrics_dir or paths["metrics_dir"])
    if "raw" in metrics_dir.parts:
        raise ValueError("Không được ghi output vào thư mục raw")
    test_csv = Path(args.test_csv or Path(paths["processed_phishing_dir"]) / "test.csv")
    baseline_dir = args.baseline_dir or paths["baseline_dir"]
    tdirs = {"roberta": Path(args.roberta_dir or paths["roberta_dir"]) / "best",
             "deberta": Path(args.deberta_dir or paths["deberta_dir"]) / "best"}

    test_df = pd.read_csv(test_csv, keep_default_na=False)
    label_col, text_col = tcfg["label_column"], tcfg["text_column"]
    print(f"[INFO] Đọc tập '{args.tag}': {len(test_df)} bản ghi từ {test_csv}")
    print("[INFO] Lưu ý: không chỉnh model/tham số dựa trên kết quả này.")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # (1) chạy dự đoán cho các model được yêu cầu
    texts = test_df[text_col].astype(str).tolist()
    for name in args.models:
        print(f"[INFO] Dự đoán bằng {DISPLAY[name]} ...")
        if name in ("lr", "mnb"):
            p = predict_baseline(name, texts, baseline_dir)
        else:
            p = predict_transformer(tdirs[name], texts, tcfg["max_length"], tcfg["batch_size"] * 2)
        out = metrics_dir / f"preds_{args.tag}_{name}.csv"
        pd.DataFrame({"row_id": range(len(p)), "y_pred": p}).to_csv(out, index=False)
        print(f"[INFO] Ghi {len(p)} dự đoán -> {out}")

    # (2) báo cáo từ mọi file preds đang có
    preds = {}
    for name in MODEL_NAMES:
        f = metrics_dir / f"preds_{args.tag}_{name}.csv"
        if f.exists():
            d = pd.read_csv(f)
            if len(d) != len(test_df):
                raise ValueError(f"{f}: {len(d)} dòng, test có {len(test_df)} dòng (khác tập?)")
            preds[name] = d["y_pred"].astype(int).to_numpy()
    if not preds:
        raise SystemExit(f"[ERROR] Chưa có file preds_{args.tag}_*.csv nào trong {metrics_dir}. Chạy với --models trước.")
    print(f"[INFO] Gom báo cáo cho: {[DISPLAY[n] for n in preds]}")

    overall, by_source = build_reports(test_df, preds, label_col, "source")
    overall.to_csv(metrics_dir / f"eval_{args.tag}_overall.csv", index=False)
    by_source.to_csv(metrics_dir / f"eval_{args.tag}_by_source.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["model", "accuracy", "precision", "recall", "f1", "fpr", "f1_ci_low", "f1_ci_high"]
    print("\n=== TOÀN BỘ TẬP ===\n" + overall[cols].round(4).to_string(index=False))
    cols = ["model", "source", "n", "n_label1", "accuracy", "precision", "recall", "f1", "fpr"]
    print("\n=== THEO NGUỒN (imc25 chỉ có tin lừa đảo nên chỉ Recall/Accuracy có nghĩa) ===\n"
          + by_source[cols].round(4).to_string(index=False))
    print(f"\n[INFO] Ghi bảng -> {metrics_dir}/eval_{args.tag}_overall.csv, eval_{args.tag}_by_source.csv")


if __name__ == "__main__":
    main()
