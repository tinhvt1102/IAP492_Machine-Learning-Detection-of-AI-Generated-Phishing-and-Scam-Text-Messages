"""Lam sach cau tra loi AI (dien tay) va ghep cap voi cau tra loi nguoi cho bo test doc lap nhanh.

Hai phien ban: "minimal" (chi bo artifact cua giao dien web) va "plain" (bo them ky hieu markdown).
Khong bao gio cat ngan cau tra loi. Khong doc data/processed/hc3/test.csv.
"""
import argparse
import csv
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BLOCK_RE = re.compile(r"^### (\S+) \| (\S+)\s*$")
LINK_LINE_RE = re.compile(r"^\s*\[[^\]]*\]\(\s*https?://[^)\s]+\s*\)\s*$")
BADGE_RE = re.compile(r"^\s*[A-Za-z0-9 .&'-]{1,60}\+\d+\s*$")
INLINE_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
LIST_MARK_RE = re.compile(r"^\s*(?:\d+\.\s+|[-*•]\s+|#+\s*)")
MD_LINE_RES = [re.compile(r"^\s*#"), re.compile(r"^\s*\d+\.\s"), re.compile(r"^\s*[-*]\s")]
EMPH_RES = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])"), r"\1"),
]

OUT_COLS = ["question_id", "site", "source", "llm", "text", "label_binary", "char_len",
            "flag_followup", "flag_has_markdown", "removed_lines"]


def collapse(s):
    return " ".join(s.split())


def parse_blocks(path):
    """Tra ve list (question_id, llm, answer_raw); bo qua header '#' o dau file."""
    blocks = []
    cur = None
    in_answer = False
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            m = BLOCK_RE.match(line)
            if m:
                cur = {"qid": m.group(1), "llm": m.group(2), "answer": []}
                blocks.append(cur)
                in_answer = False
                continue
            if cur is None:
                continue  # header
            if not in_answer:
                if line.startswith("ANSWER:"):
                    in_answer = True
                    cur["answer"].append(line[len("ANSWER:"):])
                continue
            cur["answer"].append(line)
    return [(b["qid"], b["llm"], "\n".join(b["answer"])) for b in blocks]


def clean_answer(qid, llm, raw, removed_log):
    lines = raw.splitlines()
    kept = []
    n_removed = 0
    for line in lines:
        if LINK_LINE_RE.match(line):
            removed_log.append((qid, llm, "A1_link_only_line", line))
            n_removed += 1
        elif BADGE_RE.match(line):
            removed_log.append((qid, llm, "A2_source_badge", line))
            n_removed += 1
        else:
            kept.append(line)

    # flag_followup: 15 dong khong rong cuoi (sau A.1-A.2) ket thuc bang "?"
    nonempty = [l for l in kept if l.strip()]
    flag_followup = any(l.rstrip().endswith("?") for l in nonempty[-15:])
    flag_md = ("**" in raw) or any(r.match(l) for l in lines for r in MD_LINE_RES)

    kept = [INLINE_LINK_RE.sub(r"\1", l) for l in kept]  # A.3
    minimal = collapse("\n".join(kept))                   # A.5

    plain_lines = []
    for l in kept:
        l = LIST_MARK_RE.sub("", l)
        for rx, rep in EMPH_RES:
            l = rx.sub(rep, l)
        plain_lines.append(l.replace("`", ""))
    plain = collapse("\n".join(plain_lines))
    return minimal, plain, flag_followup, flag_md, n_removed


def load_texts(path):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, encoding="utf-8", newline="") as f:
        return {collapse(r["text"]) for r in csv.DictReader(f)}


def summ(vals):
    return (f"n={len(vals)}, median={statistics.median(vals):.0f}, "
            f"mean={statistics.mean(vals):.1f}, min={min(vals)}, max={max(vals)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", default="data/processed/se_human/human_sample_28.csv")
    ap.add_argument("--answers", default="data/processed/se_human/ai_answers_template.txt")
    ap.add_argument("--out_dir", default="data/processed/se_human")
    ap.add_argument("--hc3_dir", default="data/processed/hc3")
    args = ap.parse_args()

    human = pd.read_csv(args.human, dtype=str, keep_default_na=False)
    blocks = parse_blocks(args.answers)
    seen = Counter(b[0] for b in blocks)
    dups = [q for q, c in seen.items() if c > 1]
    if dups:
        sys.exit(f"ERROR: duplicate blocks for question_id: {dups}")
    human_ids = set(human["question_id"])
    for qid, _, _ in blocks:
        if qid not in human_ids:
            print(f"WARNING: block {qid} has no matching human row; skipped")

    by_qid = {b[0]: b for b in blocks}
    filled, blank = Counter(), Counter()
    removed_log = []
    rows_min, rows_plain = [], []
    for _, h in human.iterrows():
        qid = h["question_id"]
        if qid not in by_qid:
            print(f"WARNING: human row {qid} has no block in the answers file; skipped")
            continue
        _, llm, raw = by_qid[qid]
        if not raw.strip():
            blank[llm] += 1
            print(f"WARNING: block {qid} | {llm} has an empty ANSWER; skipped with its human row")
            continue
        filled[llm] += 1
        minimal, plain, f_fu, f_md, n_rem = clean_answer(qid, llm, raw, removed_log)
        htext = collapse(h["human_answer"])
        for rows, ai_text in ((rows_min, minimal), (rows_plain, plain)):
            rows.append({"question_id": qid, "site": h["site"], "source": "human", "llm": "",
                         "text": htext, "label_binary": 0, "char_len": len(htext),
                         "flag_followup": False, "flag_has_markdown": False, "removed_lines": 0})
            rows.append({"question_id": qid, "site": h["site"], "source": "ai", "llm": llm,
                         "text": ai_text, "label_binary": 1, "char_len": len(ai_text),
                         "flag_followup": f_fu, "flag_has_markdown": f_md, "removed_lines": n_rem})

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("minimal", rows_min), ("plain", rows_plain)):
        pd.DataFrame(rows, columns=OUT_COLS).to_csv(
            os.path.join(args.out_dir, f"independent_quick_{name}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(args.out_dir, "ai_clean_removed_lines.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "llm", "rule", "line"])
        w.writerows(removed_log)

    # ---- bao cao ----
    llms = sorted(set(filled) | set(blank))
    print("\nBlocks per llm (filled / blank):")
    for l in llms:
        print(f"  {l}: filled={filled[l]}, blank={blank[l]}")
    print("\nDropped lines per rule:")
    rc = Counter(r[2] for r in removed_log)
    for rule in ("A1_link_only_line", "A2_source_badge"):
        print(f"  {rule}: {rc[rule]}")

    df = pd.DataFrame(rows_min, columns=OUT_COLS)
    if df.empty:
        print("\nNo filled pairs; nothing more to report.")
        return
    low = df["text"].str.lower()
    print(f"\nExact duplicates among all texts (case-insensitive): {int(low.duplicated(keep=False).sum())} rows involved")
    print("\nExact matches against HC3 train.csv and val.csv (test.csv deliberately NOT read):")
    for split in ("train", "val"):
        pool = load_texts(os.path.join(args.hc3_dir, f"{split}.csv"))
        print(f"  {split}: {int(df['text'].isin(pool).sum())} matches")

    print("\nChar length (minimal text) per label:")
    for lab, name in ((0, "human"), (1, "ai")):
        print(f"  {name}: {summ(df.loc[df.label_binary == lab, 'char_len'].tolist())}")
    print("Char length per llm:")
    for l in llms:
        v = df.loc[df.llm == l, "char_len"].tolist()
        if v:
            print(f"  {l}: {summ(v)}")

    print("\nShare of texts containing marker (minimal text):")
    for lab, name in ((0, "human"), (1, "ai")):
        sub = df[df.label_binary == lab]["text"]
        shares = ", ".join(f'"{m}"={sub.str.contains(m, regex=False).mean():.0%}'
                           for m in ("http", "...", "*", "#"))
        print(f"  {name}: {shares}")

    print("\nRows with flag_followup:")
    fl = df[df.flag_followup]
    if fl.empty:
        print("  (none)")
    for _, r in fl.iterrows():
        print(f"  {r['question_id']} | {r['llm']}")


if __name__ == "__main__":
    main()
