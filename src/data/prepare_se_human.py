"""Chon cau tra loi cua NGUOI tu Stack Exchange dump cho tap test doc lap cua nhanh HC3.

Doc stream data/raw/stackexchange/<site>/Posts.xml (ET.iterparse), loc theo cac buoc
co dinh, lay mau per_site cau hoi/site va ghi ra out_dir. Chi dung thu vien chuan.
"""
import argparse
import csv
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

sys.stdout.reconfigure(encoding="utf-8")

CODE_RE = re.compile(r"<\s*(pre|code|img|table)\b", re.IGNORECASE)
BLOCK_TAGS = {"p", "br", "li", "ul", "ol", "div", "blockquote", "hr", "h1", "h2", "h3", "h4"}

REPORT_COLS = [
    "site", "questions_total", "answers_total", "answers_date_ok", "answers_score_ok",
    "answers_no_code", "answers_length_ok", "answers_after_dedup",
    "questions_with_eligible_answer", "questions_final_pool", "sampled",
]
SAMPLE_COLS = [
    "question_id", "site", "question_title", "question_body", "answer_id", "answer_created",
    "answer_score", "license", "human_answer", "char_len",
]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html):
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join("".join(parser.parts).split())


def has_code(html):
    return CODE_RE.search(html) is not None


def percentile_index(n, q):
    return round((n - 1) * q)


def hc3_human_length_range(train_path):
    # Chi doc train.csv (khong bao gio doc test.csv)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    lengths = []
    with open(train_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["label_binary"] == "0":
                lengths.append(len(row["text"]))
    if not lengths:
        sys.exit(f"ERROR: no rows with label_binary == '0' in {train_path}")
    lengths.sort()
    n = len(lengths)
    return lengths[percentile_index(n, 0.10)], lengths[percentile_index(n, 0.90)], n


def stream_site(path, cutoff, min_score):
    """Stream Posts.xml, tra ve (questions, answers, questions_total, answers_total).

    questions: {id: (date_ok, title, body_html)}; answers: list theo thu tu file, chi giu
    cac answer co ngay cua chinh no hop le (body chi giu neu score >= min_score).
    """
    questions = {}
    answers = []
    q_total = a_total = 0
    root = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if root is None:
                root = elem
            continue
        if elem.tag != "row":
            continue
        a = elem.attrib
        ptype = a.get("PostTypeId")
        created = a.get("CreationDate", "")
        edited = a.get("LastEditDate")
        own_ok = bool(created) and created < cutoff and (edited is None or edited < cutoff)
        if ptype == "1":
            q_total += 1
            if own_ok:
                questions[a["Id"]] = (a.get("Title", ""), a.get("Body", ""))
        elif ptype == "2":
            a_total += 1
            if "ParentId" not in a:
                sys.exit(f"ERROR: answer row Id={a.get('Id')} in {path} has no ParentId")
            if own_ok:
                score = int(a.get("Score", "0"))
                body = a.get("Body", "") if score >= min_score else None
                answers.append((a["Id"], a["ParentId"], created, score,
                                a.get("ContentLicense", ""), body))
        elem.clear()
        root.clear()
    return questions, answers, q_total, a_total


def process_site(site, path, args, min_chars, max_chars, rng):
    questions, answers, q_total, a_total = stream_site(path, args.cutoff, args.min_score)
    rep = {"site": site, "questions_total": q_total, "answers_total": a_total}

    # 1. ngay: answer va question deu < cutoff
    step1 = [x for x in answers if x[1] in questions]
    rep["answers_date_ok"] = len(step1)
    # 2. score
    step2 = [x for x in step1 if x[3] >= args.min_score]
    rep["answers_score_ok"] = len(step2)
    # 3. khong co code/img/table o ca answer lan question
    step3 = [x for x in step2 if not has_code(x[5]) and not has_code(questions[x[1]][1])]
    rep["answers_no_code"] = len(step3)

    # 4-5. doi sang text thuan va loc do dai answer
    q_text = {}
    step5 = []
    for aid, qid, created, score, lic, body in step3:
        text = html_to_text(body)
        if min_chars <= len(text) <= max_chars:
            step5.append((aid, qid, created, score, lic, text))
    rep["answers_length_ok"] = len(step5)

    # 6. do dai question va title khong rong
    step6 = []
    for item in step5:
        qid = item[1]
        if qid not in q_text:
            title, qbody = questions[qid]
            title = html_to_text(title)
            body_txt = html_to_text(qbody)
            q_text[qid] = (title, body_txt, len((title + " " + body_txt)))
        title, body_txt, qlen = q_text[qid]
        if title and qlen <= args.max_q_chars:
            step6.append(item)

    # 7. bo answer trung lap (giu cai dau tien theo thu tu file)
    seen = set()
    step7 = []
    for item in step6:
        key = item[5].lower()
        if key not in seen:
            seen.add(key)
            step7.append(item)
    rep["answers_after_dedup"] = len(step7)
    rep["questions_with_eligible_answer"] = len({x[1] for x in step7})

    # 8. moi question 1 answer: score cao nhat, hoa thi Id nho nhat
    best = {}
    for item in step7:
        cur = best.get(item[1])
        if cur is None or (-item[3], int(item[0])) < (-cur[3], int(cur[0])):
            best[item[1]] = item
    pool = sorted(best.values(), key=lambda x: f"{site}-{x[1]}")
    rep["questions_final_pool"] = len(pool)

    rng.shuffle(pool)
    if len(pool) < args.per_site:
        print(f"WARNING: site '{site}' has only {len(pool)} eligible questions "
              f"(requested {args.per_site})")
    chosen = pool[: args.per_site]
    rep["sampled"] = len(chosen)

    rows = []
    for aid, qid, created, score, lic, text in chosen:
        title, body_txt, _ = q_text[qid]
        rows.append({
            "question_id": f"{site}-{qid}", "site": site, "question_title": title,
            "question_body": body_txt, "answer_id": aid, "answer_created": created,
            "answer_score": score, "license": lic, "human_answer": text, "char_len": len(text),
        })
    return rep, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/stackexchange")
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--out_dir", default="data/processed/se_human")
    ap.add_argument("--cutoff", default="2022-12-01")
    ap.add_argument("--min_score", type=int, default=1)
    ap.add_argument("--hc3_train", default="data/processed/hc3/train.csv")
    ap.add_argument("--min_chars", type=int, default=None)
    ap.add_argument("--max_chars", type=int, default=None)
    ap.add_argument("--max_q_chars", type=int, default=1500)
    ap.add_argument("--per_site", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.min_chars is None or args.max_chars is None:
        lo, hi, n = hc3_human_length_range(args.hc3_train)
        min_chars = lo if args.min_chars is None else args.min_chars
        max_chars = hi if args.max_chars is None else args.max_chars
        print(f"Length range from HC3 train human answers (n={n}): "
              f"min_chars={min_chars}, max_chars={max_chars} (P10-P90)")
    else:
        min_chars, max_chars = args.min_chars, args.max_chars
        print(f"Length range given on command line: min_chars={min_chars}, max_chars={max_chars}")

    rng_seed = args.seed
    reports, all_rows = [], []
    for site in args.sites:
        path = os.path.join(args.root, site, "Posts.xml")
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {path} not found")
        print(f"Processing {site} ...")
        rep, rows = process_site(site, path, args, min_chars, max_chars, random.Random(rng_seed))
        reports.append(rep)
        all_rows.extend(rows)
        print("  " + ", ".join(f"{k}={rep[k]}" for k in REPORT_COLS[1:]))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "human_sample.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_COLS)
        w.writeheader()
        w.writerows(all_rows)
    with open(os.path.join(args.out_dir, "questions_for_llm.txt"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(f"[{r['question_id']}] {r['question_title']}\n")
            if r["question_body"]:
                f.write(r["question_body"] + "\n")
            f.write("\n")
    with open(os.path.join(args.out_dir, "filter_report.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLS)
        w.writeheader()
        w.writerows(reports)
    print(f"Wrote {len(all_rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
