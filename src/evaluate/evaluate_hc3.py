"""evaluate_hc3.py - Đánh giá RQ2 trên tập TEST của HC3 (người viết = 0, ChatGPT = 1 = positive).

Hai việc nối tiếp (giống evaluate_models.py của RQ1, nhưng là file riêng cho HC3):
  (1) --models ...: chạy dự đoán, ghi <metrics_dir>/hc3_preds_<tag>_<model>.csv
        roberta: model tốt nhất đã chọn bằng val (models/hc3_roberta/best)
        length : baseline CHỈ dùng độ dài ký tự (StandardScaler + LogisticRegression, fit trên train.csv)
                 -> cho biết riêng độ dài giải thích được bao nhiêu phần điểm số
  (2) luôn luôn: đọc mọi file hc3_preds_<tag>_*.csv đang có, tính Accuracy, Precision, Recall, F1, FPR,
      khoảng tin cậy 95% của F1 (bootstrap), ma trận nhầm lẫn; trên các tập con: toàn bộ, không có 'word.Word',
      có 'word.Word'; mỗi tập con: tổng + theo source. Luôn in kèm số mẫu mỗi nhãn.

Test chỉ chạy dự đoán MỘT lần cho mỗi model; KHÔNG chỉnh model/tham số dựa trên kết quả này.
--test_csv/--tag cho phép chạy thử script trên tập khác (vd val) mà không mở test.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

try:  # chạy từ thư mục gốc project: python src/evaluate/evaluate_hc3.py
    from src.evaluate.metrics_utils import compute_metrics
except ModuleNotFoundError:  # chạy ở nơi khác: thêm thư mục gốc vào sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluate.metrics_utils import compute_metrics
from src.data.prepare_hc3 import WORD_DOT_WORD_RE  # cùng định nghĩa 'word.Word' với lúc chuẩn bị dữ liệu

MODEL_NAMES = ["roberta", "length"]
DISPLAY = {"roberta": "RoBERTa (HC3)", "length": "LR độ dài ký tự"}
SUBSETS = ["all", "no_word.Word", "word.Word"]


# ---------------------------------------------------------------- dự đoán
def predict_roberta(model_dir, texts, max_length, batch_size):
    """Nạp checkpoint tốt nhất và dự đoán. Trả về (nhãn dự đoán, xác suất lớp 1)."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from src.features.tokenizer_utils import get_tokenizer, make_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(str(model_dir))  # tokenizer đã được lưu cùng model
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).float().to(device).eval()
    loader = make_loader(texts, [0] * len(texts), tok, max_length, batch_size, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            preds += logits.argmax(dim=-1).cpu().tolist()
            probs += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    return preds, probs


def predict_length(train_csv, texts, text_col, label_col, seed):
    """Baseline chỉ dùng độ dài ký tự: fit trên train, dự đoán trên tập đánh giá."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = pd.read_csv(train_csv, keep_default_na=False)
    x_tr = tr[text_col].astype(str).str.len().to_numpy().reshape(-1, 1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed))
    clf.fit(x_tr, tr[label_col].astype(int))
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    w, b = lr.coef_[0, 0], lr.intercept_[0]
    # dự đoán 1 khi w*z + b > 0 với z = (độ dài - mean)/std -> ngưỡng độ dài tương đương
    thr = sc.mean_[0] - b * sc.scale_[0] / w
    print(f"[INFO] Baseline độ dài: fit trên {len(tr)} mẫu train; dự đoán ChatGPT khi độ dài "
          f"{'>' if w > 0 else '<'} {thr:.0f} ký tự")
    x = np.array([len(t) for t in texts]).reshape(-1, 1)
    return clf.predict(x).astype(int).tolist(), clf.predict_proba(x)[:, 1].tolist()


# ---------------------------------------------------------------- báo cáo
def bootstrap_f1_ci(y_true, y_pred, n_boot, seed):
    """Khoảng tin cậy 95% của F1 bằng bootstrap (lấy mẫu lại có hoàn lại)."""
    rng, n = np.random.RandomState(seed), len(y_true)
    f1s = [f1_score(y_true[i], y_pred[i], zero_division=0) for i in (rng.randint(0, n, n) for _ in range(n_boot))]
    return np.percentile(f1s, [2.5, 97.5])


def metrics_row(y, p, n_boot, seed):
    """Metric cho 1 tập con; tập chỉ có 1 lớp thì metric không xác định -> NaN (không bịa số)."""
    m = compute_metrics(y, p)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    lo = hi = np.nan
    if n_pos and n_neg:
        lo, hi = bootstrap_f1_ci(y, p, n_boot, seed)
    if not n_neg:
        m.update(precision=np.nan, f1=np.nan, fpr=np.nan)
    if not n_pos:
        m.update(precision=np.nan, recall=np.nan, f1=np.nan)
    return {"n": len(y), "n_human": n_neg, "n_chatgpt": n_pos, **m, "f1_ci_low": lo, "f1_ci_high": hi}


def build_reports(df, preds, label_col, n_boot, seed):
    y_all = df[label_col].astype(int).to_numpy()
    has_wdw = df["text"].astype(str).str.contains(WORD_DOT_WORD_RE).to_numpy()
    masks = {"all": np.ones(len(df), bool), "no_word.Word": ~has_wdw, "word.Word": has_wdw}
    src = df["source"].to_numpy()
    rows = []
    for name, p in preds.items():
        for sub in SUBSETS:
            for s in ["ALL"] + sorted(df["source"].unique()):
                mask = masks[sub] & (True if s == "ALL" else src == s)
                if mask.sum() == 0:
                    continue
                rows.append({"model": DISPLAY[name], "subset": sub, "source": s,
                             **metrics_row(y_all[mask], p[mask], n_boot, seed)})
    out = pd.DataFrame(rows)
    return out[out["source"] == "ALL"].drop(columns="source"), out[out["source"] != "ALL"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/hc3_config.yaml")
    ap.add_argument("--models", nargs="*", default=[], choices=MODEL_NAMES,
                    help="Model cần chạy dự đoán lần này (bỏ trống = chỉ làm báo cáo từ file preds có sẵn)")
    ap.add_argument("--test_csv", default=None, help="Mặc định: <processed_hc3_dir>/test.csv")
    ap.add_argument("--tag", default="test", help="Nhãn tập đánh giá, dùng đặt tên file output")
    ap.add_argument("--model_dir", default=None, help="Thư mục best/ của RoBERTa (mặc định <hc3_roberta_dir>/best)")
    ap.add_argument("--metrics_dir", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths, tcfg, ecfg = cfg["paths"], cfg["transformer"], cfg["evaluate"]
    metrics_dir = Path(args.metrics_dir or paths["metrics_dir"])
    if "raw" in metrics_dir.parts:
        raise ValueError("Không được ghi output vào thư mục raw")
    data_dir = Path(paths["processed_hc3_dir"])
    test_csv = Path(args.test_csv or data_dir / "test.csv")
    model_dir = Path(args.model_dir or Path(paths["hc3_roberta_dir"]) / "best")
    label_col, text_col = tcfg["label_column"], tcfg["text_column"]

    df = pd.read_csv(test_csv, keep_default_na=False)
    print(f"[INFO] Đọc tập '{args.tag}': {len(df)} bản ghi từ {test_csv}")
    print("[INFO] Lưu ý: không chỉnh model/tham số dựa trên kết quả này.")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # (1) chạy dự đoán cho các model được yêu cầu
    texts = df[text_col].astype(str).tolist()
    for name in args.models:
        print(f"[INFO] Dự đoán bằng {DISPLAY[name]} ...")
        if name == "roberta":
            p, prob = predict_roberta(model_dir, texts, tcfg["max_length"], tcfg["batch_size"] * 2)
        else:
            p, prob = predict_length(data_dir / "train.csv", texts, text_col, label_col, ecfg["length_baseline_seed"])
        out = metrics_dir / f"hc3_preds_{args.tag}_{name}.csv"
        pd.DataFrame({"row_id": range(len(p)), "y_pred": p, "prob_chatgpt": prob}).to_csv(out, index=False)
        print(f"[INFO] Ghi {len(p)} dự đoán -> {out}")

    # (2) báo cáo từ mọi file preds đang có
    preds = {}
    for name in MODEL_NAMES:
        f = metrics_dir / f"hc3_preds_{args.tag}_{name}.csv"
        if f.exists():
            d = pd.read_csv(f)
            if len(d) != len(df):
                raise ValueError(f"{f}: {len(d)} dòng, tập đánh giá có {len(df)} dòng (khác tập?)")
            preds[name] = d["y_pred"].astype(int).to_numpy()
    if not preds:
        raise SystemExit(f"[ERROR] Chưa có file hc3_preds_{args.tag}_*.csv nào trong {metrics_dir}. Chạy với --models trước.")

    overall, by_source = build_reports(df, preds, label_col, ecfg["n_bootstrap"], ecfg["bootstrap_seed"])
    f_all = metrics_dir / f"hc3_eval_{args.tag}_overall.csv"
    f_src = metrics_dir / f"hc3_eval_{args.tag}_by_source.csv"
    overall.to_csv(f_all, index=False)
    by_source.to_csv(f_src, index=False)

    pd.set_option("display.width", 250)
    cols = ["model", "subset", "n", "n_human", "n_chatgpt", "accuracy", "precision", "recall", "f1",
            "f1_ci_low", "f1_ci_high", "fpr"]
    print("\n=== TOÀN BỘ TẬP (và tập con theo 'word.Word') ===\n" + overall[cols].round(4).to_string(index=False))
    print("\n=== THEO NGUỒN ===\n" + by_source[cols[:2] + ["source"] + cols[2:]].round(4).to_string(index=False))
    print("\n=== MA TRẬN NHẦM LẪN (toàn bộ tập; hàng = nhãn thật, cột = dự đoán) ===")
    for _, r in overall[overall["subset"] == "all"].iterrows():
        print(f"{r['model']}:\n              pred human  pred chatgpt\n"
              f"  thật human   {r['tn']:>10}  {r['fp']:>12}\n  thật chatgpt {r['fn']:>10}  {r['tp']:>12}")
    print(f"\n[INFO] Ghi bảng -> {f_all}, {f_src}")


if __name__ == "__main__":
    main()
