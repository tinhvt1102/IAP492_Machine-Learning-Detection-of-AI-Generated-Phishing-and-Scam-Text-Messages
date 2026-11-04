"""Danh gia nhanh (n nho) mo hinh HC3 RoBERTa tren bo test doc lap tu clean_ai_answers.py.

Chi doc file --data, khong dung data/processed/hc3/test.csv. Khong tune gi tu ket qua nay.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")


def predict(df, model_dir, max_length, batch_size):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    texts = df["text"].tolist()
    probs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs), device


def report(d, title):
    print(f"\n=== {title} ===")
    if d.empty:
        print("  (no rows)")
        return
    ok = d["pred"] == d["label_binary"]
    print(f"Overall accuracy: {ok.mean():.4f} ({int(ok.sum())}/{len(d)})")
    cm = pd.crosstab(d["label_binary"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    print("Confusion matrix (rows true 0/1, columns predicted 0/1):")
    print(cm.to_string())
    h, a = d[d.label_binary == 0], d[d.label_binary == 1]
    print(f"Human correctly kept as human: {int((h.pred == 0).sum())}/{len(h)}")
    print(f"AI detected: {int((a.pred == 1).sum())}/{len(a)}")
    print("AI detected per llm:")
    for l, g in a.groupby("llm"):
        print(f"  {l}: {int((g.pred == 1).sum())}/{len(g)}")
    print("Per site:")
    for s, g in d.groupby("site"):
        gh, ga = g[g.label_binary == 0], g[g.label_binary == 1]
        print(f"  {s}: human kept {int((gh.pred == 0).sum())}/{len(gh)}, "
              f"AI detected {int((ga.pred == 1).sum())}/{len(ga)}")
    print(f"Mean prob_ai: label 0 = {h.prob_ai.mean():.4f}, label 1 = {a.prob_ai.mean():.4f}"
          if len(h) and len(a) else "Mean prob_ai: (one class missing)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_dir", default="models/hc3_roberta/best")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="results/metrics")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data, dtype={"question_id": str, "llm": str}, keep_default_na=False)
    df["flag_followup"] = df["flag_followup"].astype(str) == "True"
    df["flag_has_markdown"] = df["flag_has_markdown"].astype(str) == "True"
    df["label_binary"] = df["label_binary"].astype(int)

    pred, prob_ai, device = predict(df, args.model_dir, args.max_length, args.batch_size)
    df["prob_ai"] = prob_ai
    df["pred"] = pred
    print(f"Device: {device}; rows: {len(df)}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"hc3_independent_quick_{args.tag}_preds.csv")
    df[["question_id", "source", "llm", "label_binary", "pred", "prob_ai",
        "flag_followup", "flag_has_markdown"]].to_csv(out, index=False, encoding="utf-8")
    print(f"Saved predictions to {out}")

    report(df, "ALL ROWS")
    for title, flagged in (("EXCLUDING flag_followup AI rows and their paired human rows",
                            df.flag_followup),
                           ("EXCLUDING any flagged AI rows (followup or markdown) and their paired human rows",
                            df.flag_followup | df.flag_has_markdown)):
        bad = set(df.loc[(df.label_binary == 1) & flagged, "question_id"])
        report(df[~df.question_id.isin(bad)], title)

    print("\nn is tiny: these numbers are an early signal, not results for the report.")


if __name__ == "__main__":
    main()
