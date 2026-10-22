"""compare_robustness.py - Phân tích mức giảm điểm (performance drop): tập test của người vs tập tin nhắn do AI viết.

Input : <metrics_dir>/eval_test_overall.csv   (người)       <metrics_dir>/eval_<tag>_overall.csv (AI)
        <metrics_dir>/preds_<tag>_<model>.csv + --ai_csv (để tách theo loại lừa đảo và liệt kê tin bị sai)
Output: <metrics_dir>/robustness_comparison.csv, robustness_by_scam_type.csv, robustness_by_entity.csv, robustness_errors.csv
Delta = AI - người (âm = điểm giảm).
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

MODELS = {"lr": "LR", "mnb": "MNB", "roberta": "RoBERTa", "deberta": "DeBERTa"}
ENTITY_PAT = r"<(?:URL|EMAIL_ADDRESS|PHONE_NUMBER)>"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/model_config.yaml")
    ap.add_argument("--ai_csv", required=True, help="Tập AI đã chuẩn hoá (data/processed/ai_generated/test.csv)")
    ap.add_argument("--tag", default="ai")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mdir = Path(cfg["paths"]["metrics_dir"])
    human_test = Path(cfg["paths"]["processed_phishing_dir"]) / "test.csv"
    ai = pd.read_csv(args.ai_csv, keep_default_na=False)
    pd.set_option("display.width", 220)

    # --- so sánh điểm toàn bộ
    h = pd.read_csv(mdir / "eval_test_overall.csv")
    a = pd.read_csv(mdir / f"eval_{args.tag}_overall.csv")
    k = ["accuracy", "precision", "recall", "f1", "fpr"]
    m = h[["model"] + k].merge(a[["model"] + k + ["fp", "fn", "f1_ci_low", "f1_ci_high"]], on="model", suffixes=("_nguoi", "_ai"))
    for c in k:
        m[f"{c}_delta"] = m[f"{c}_ai"] - m[f"{c}_nguoi"]
    cols = ["model", "f1_nguoi", "f1_ai", "f1_delta", "f1_ci_low", "f1_ci_high",
            "recall_nguoi", "recall_ai", "recall_delta", "fpr_nguoi", "fpr_ai", "fpr_delta", "fp", "fn"]
    print(f"=== NGƯỜI (test) vs AI ({len(ai)} tin) ===")
    print(m[cols].round(4).to_string(index=False))
    m.to_csv(mdir / "robustness_comparison.csv", index=False)

    # --- recall theo loại lừa đảo + danh sách tin sai
    y = ai["label_binary"].astype(int).to_numpy()
    preds = {}
    for key, name in MODELS.items():
        f = mdir / f"preds_{args.tag}_{key}.csv"
        if f.exists():
            p = pd.read_csv(f)["y_pred"].astype(int).to_numpy()
            if len(p) != len(ai):
                raise ValueError(f"{f}: {len(p)} dòng, tập AI có {len(ai)} dòng")
            preds[name] = p
    rows = []
    for name, p in preds.items():
        d = ai.assign(pred=p)
        for t, g in d.groupby("scam_type"):
            ok = (g["pred"] == g["label_binary"]).mean()
            rows.append({"model": name, "scam_type": t, "label_binary": int(g["label_binary"].iloc[0]),
                         "n": len(g), "ti_le_dung": ok})
    bt = pd.DataFrame(rows)
    piv = bt.pivot_table(index=["label_binary", "scam_type", "n"], columns="model", values="ti_le_dung").round(3)
    print("\n=== TỈ LỆ ĐOÁN ĐÚNG THEO LOẠI (nhãn 1 = Recall của loại đó; nhãn 0 = 1 - FPR) ===")
    print(piv.to_string())
    bt.to_csv(mdir / "robustness_by_scam_type.csv", index=False)

    # --- Recall (tin lừa đảo) và FPR (tin hợp lệ) theo có/không thực thể (URL/số điện thoại/email), người vs AI
    def recall_by_entity(df, tag, label):
        out = []
        for key, name in MODELS.items():
            f = mdir / f"preds_{tag}_{key}.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)["y_pred"].astype(int).to_numpy()
            if len(p) != len(df):
                raise ValueError(f"{f}: {len(p)} dòng, tập {label} có {len(df)} dòng")
            d = df.assign(pred=p, co_thuc_the=df["text"].str.contains(ENTITY_PAT))
            for lab, kind, fn in ((1, "lua_dao: Recall", lambda g: (g["pred"] == 1).mean()),
                                  (0, "hop_le: FPR", lambda g: (g["pred"] == 1).mean())):
                for flag, g in d[d["label_binary"] == lab].groupby("co_thuc_the"):
                    out.append({"tap": label, "loai_tin": kind, "co_thuc_the": bool(flag), "n": len(g),
                                "model": name, "gia_tri": fn(g)})
        return out

    hdf = pd.read_csv(human_test, keep_default_na=False)
    ent = pd.DataFrame(recall_by_entity(hdf, "test", "nguoi") + recall_by_entity(ai, args.tag, "ai"))
    if len(ent):
        pe = ent.pivot_table(index=["loai_tin", "tap", "co_thuc_the", "n"], columns="model", values="gia_tri").round(3)
        print("\n=== THEO CÓ/KHÔNG CÓ URL/SĐT/EMAIL (người vs AI): Recall của tin lừa đảo và FPR của tin hợp lệ ===")
        print(pe.to_string())
        ent.to_csv(mdir / "robustness_by_entity.csv", index=False)

    err = ai.copy()
    for name, p in preds.items():
        err[f"pred_{name}"] = p
    wrong = (err[[f"pred_{n}" for n in preds]].ne(err["label_binary"], axis=0)).any(axis=1)
    err = err[wrong]
    err.to_csv(mdir / "robustness_errors.csv", index=False)
    print(f"\n[INFO] Số tin bị ít nhất 1 mô hình đoán sai: {len(err)}/{len(ai)}")
    print(f"[INFO] Ghi -> {mdir}/robustness_comparison.csv, robustness_by_scam_type.csv, robustness_errors.csv")


if __name__ == "__main__":
    main()
