"""prepare_hc3.py - Chuẩn bị dữ liệu HC3 cho RQ2 (người viết = 0, ChatGPT viết = 1). Độc lập với RQ1.

Input : data/raw/hc3/HC3.jsonl (chỉ đọc)
Output: data/processed/hc3/{train,val,test}.csv, cột text,label_binary,source,question_id
        (chỉ ghi khi split.write_files = true trong config)

Các bước:
  (0) đọc jsonl (dòng lỗi: thử bỏ \\x00; vẫn lỗi -> log số dòng, không bỏ âm thầm), trải phẳng câu trả lời
  (1) thống kê dữ liệu thô
  (2) bỏ câu trả lời rỗng và chuỗi lỗi/giao diện ChatGPT (theo clean.error_patterns)
  (3) chuẩn hoá định dạng (b), áp dụng GIỐNG NHAU cho cả 2 nhãn
  (4) bỏ trùng lặp chính xác (sau chuẩn hoá), giữ lần xuất hiện đầu; text ở cả 2 nhãn -> bỏ hết
  (5) cân bằng 50/50 trong từng source (lấy mẫu bớt nhãn đông hơn, seed cố định)
  (6) chia theo CÂU HỎI (question_id = 1 mã cho mỗi text câu hỏi khác nhau), cân theo (source, label)

LƯU Ý: các quy tắc chuẩn hoá (b) (kể cả phần mở rộng: ngoặc, "x ' s", xuống dòng, khoảng trắng, ghép cặp nháy) và tỉ lệ
chia 70/15/15 là ĐỀ XUẤT, nhóm CHƯA xác nhận. Cố ý KHÔNG sửa: '...', '*', chữ thường đầu câu, thiếu dấu
câu cuối, nháy cong, 'word.Word' (là văn phong thật hoặc sửa dễ làm hỏng viết tắt/số thập phân).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NAME = {0: "human", 1: "chatgpt"}

# Dấu vết định dạng dùng để thống kê (trước và sau chuẩn hoá)
URL_PH_RE = re.compile(r"\bURL_\d+\b")                 # placeholder kiểu URL_0
SPACE_PUNCT_RE = re.compile(r"\w (?:[,.;:!?])(?=\s|$)")  # khoảng trắng trước dấu câu: "word , next"
SPLIT_NT_RE = re.compile(r"\w n't\b")                   # "n't" bị tách: "do n't"
SPLIT_CONTR_RE = re.compile(r"(?i)\w (?:n't|'s|'re|'ve|'ll|'d|'m)\b")  # mọi dạng rút gọn bị tách

WORD_DOT_WORD_RE = re.compile(r"[a-z]\.[A-Z]")         # dính câu 'word.Word' (không sửa; evaluate_hc3.py dùng lại)

# Quy tắc chuẩn hoá (b) - ĐỀ XUẤT, chưa được nhóm xác nhận. Áp dụng theo đúng thứ tự, giống nhau cho 2 nhãn.
NORM_RULES = [
    (re.compile(r"\bURL_\d+\b"), "<URL>"),                                  # mọi URL_n -> 1 placeholder
    (re.compile(r"\r\n|\r|\n|\\n"), " "),                                   # xuống dòng thật và chuỗi '\n' dạng chữ -> 1 khoảng trắng
    (re.compile(r"\( +"), "("),                                             # "( x" -> "(x"
    (re.compile(r" +\)"), ")"),                                             # "x )" -> "x)"
    (re.compile(r"(?i)(?<=\w) ' (?=(?:s|t|re|ve|ll|d|m)\b)"), "'"),         # "it ' s" -> "it's" (chỉ dạng rút gọn/sở hữu)
    (re.compile(r"(?<=\S) +([,.;:!?])(?=\s|$)"), r"\1"),                    # bỏ khoảng trắng trước dấu câu
    (re.compile(r"(?i)(?<=\w) (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),       # nối lại rút gọn: "do n't" -> "don't"
]
# Sau NORM_RULES: ghép cặp nháy (fix_quotes), rồi gộp khoảng trắng và bỏ khoảng trắng 2 đầu
MULTI_SPACE_RE = re.compile(r" {2,}")
NEEDS_DQ_FIX_RE = re.compile(r'(?<![^\s(\[])" |(?<=\s)"(?=$|[\s,.;:!?)\]])')  # còn nháy kép có khoảng trắng phía trong
OPEN_BEFORE = set(" ([")          # ký tự được phép đứng ngay trước nháy MỞ (ngoài đầu chuỗi)
CLOSE_AFTER = set(" ,.;:!?)]")    # ký tự được phép đứng ngay sau nháy ĐÓNG (ngoài cuối chuỗi)

# Các khác biệt định dạng KHÁC (chỉ thống kê, không sửa) để nhóm xem xét
EXTRA_FEATURES = {
    "khoảng trắng trong ngoặc '( x' / 'x )'": r"\( | \)",
    "khoảng trắng quanh nháy \"x ' s\"": r"\w ' \w",
    "cặp nháy đơn có khoảng trắng ' x '": r"(?:^|\s)' [^']{1,80}? '(?=\s|[,.;:!?]|$)",
    "cặp nháy kép có khoảng trắng \" x \"": r'(?:^|\s)" [^"]{1,80}? "(?=\s|[,.;:!?]|$)',
    "nháy đơn đứng riêng (2 bên là khoảng trắng/đầu/cuối)": r"(?:^|\s)'(?=\s|$)",
    "nháy kép còn khoảng trắng phía trong": NEEDS_DQ_FIX_RE.pattern,
    "2 khoảng trắng liền nhau": r"  ",
    "có xuống dòng thật (\\n)": r"\n",
    "có \\r": r"\r",
    "có chuỗi '\\n' dạng chữ (backslash + n)": r"\\n",
    "mở đầu bằng khoảng trắng": r"^\s",
    "kết thúc bằng khoảng trắng": r"\s$",
    "mở đầu bằng '!'": r"^!",
    "mở đầu kiểu '2 / 2'": r"^\s*(?:\\n)*\s*\d+ / \d+",
    "dính câu 'word.Word' (thiếu khoảng trắng)": r"[a-z]\.[A-Z]",
    "dấu ba chấm '...'": r"\.\.\.",
    "dấu '*' (markdown)": r"\*",
    "nháy cong ‘ ’ “ ”": r"[‘’“”]",
    "ký tự ngoài ASCII": r"[^\x00-\x7f]",
    "mở đầu bằng chữ thường": r"^\s*[a-z]",
    "không kết thúc bằng . ! ?": r"[^.!?\"')\s]\s*$",
    "có link http(s)": r"https?://",
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- (0) đọc + trải phẳng
def read_jsonl(path):
    """Đọc từng dòng; dòng lỗi thì thử bỏ \\x00 rồi parse lại. Dòng vẫn lỗi -> ghi log, KHÔNG bỏ âm thầm."""
    records, fixed, skipped = [], [], []
    with open(path, "rb") as f:
        for line_no, raw in enumerate(f, 1):
            s = raw.decode("utf-8")
            try:
                records.append((line_no, json.loads(s)))
                continue
            except json.JSONDecodeError:
                pass
            try:
                records.append((line_no, json.loads(s.replace("\x00", ""))))
                fixed.append(line_no)
            except json.JSONDecodeError as e:
                skipped.append((line_no, s.count("\x00"), str(e)))
    n_lines = line_no
    print(f"[INFO] {path}: {n_lines} dòng | parse được {len(records)} "
          f"(trong đó sửa bằng cách bỏ \\x00: {len(fixed)}) | bỏ qua: {len(skipped)}")
    for ln in fixed:
        print(f"[INFO]   dòng {ln}: đã sửa (bỏ \\x00)")
    for ln, n_nul, err in skipped:
        print(f"[WARN]   dòng {ln}: KHÔNG parse được dù đã bỏ {n_nul} ký tự \\x00 -> bỏ qua ({err})")
    return records


def flatten(records):
    """Mỗi câu trả lời -> 1 dòng. line_no, question lấy từ dòng json (không lấy từ câu trả lời)."""
    rows = []
    for line_no, d in records:
        for key, label in (("human_answers", 0), ("chatgpt_answers", 1)):
            for ans in d[key]:
                rows.append({"text": ans, "label_binary": label, "source": d["source"],
                             "line_no": line_no, "question": d["question"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- báo cáo
def print_counts(df, step):
    """In số mẫu theo source x nhãn (và tổng) sau mỗi bước."""
    t = pd.crosstab(df["source"], df["label_binary"].map(NAME), margins=True, margins_name="TONG")
    for c in ("human", "chatgpt"):
        if c not in t:
            t[c] = 0
    t = t[["human", "chatgpt", "TONG"]]
    t["%chatgpt"] = (100 * t["chatgpt"] / t["TONG"]).round(1)
    print(f"\n--- Số mẫu: {step} ---\n{t.to_string()}")


def pct_table(values_by_label):
    out = []
    for lab, v in values_by_label.items():
        v = np.asarray(v)
        out.append({"label": lab, "n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                    "p99": float(np.percentile(v, 99)), "max": int(v.max())})
    return pd.DataFrame(out)


def count_tokens(texts, tokenizer_name):
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # tắt cảnh báo "dài hơn 512" (ở đây chỉ đếm, không đưa vào model)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return np.array([len(x) for x in tok(list(texts), add_special_tokens=True)["input_ids"]])


def artifact_report(df, title):
    """Tỉ lệ các dấu vết định dạng chính theo nhãn."""
    lab = df["label_binary"].map(NAME)
    flags = pd.DataFrame({
        "URL_n placeholder": df["text"].str.contains(URL_PH_RE),
        "khoảng trắng trước dấu câu": df["text"].str.contains(SPACE_PUNCT_RE),
        "n't bị tách": df["text"].str.contains(SPLIT_NT_RE),
        "rút gọn bị tách (n't 's 're 've 'll 'd 'm)": df["text"].str.contains(SPLIT_CONTR_RE),
    })
    flags["có ít nhất 1 dấu vết"] = flags.any(axis=1)
    print(f"\n=== DẤU VẾT ĐỊNH DẠNG {title} (số dòng, % trong nhãn) ===")
    for col in flags:
        s = flags[col].groupby(lab).agg(["sum", "mean"])
        print(f"{col:45s} " + " | ".join(f"{k}: {int(r['sum'])} ({100 * r['mean']:.1f}%)" for k, r in s.iterrows()))
    print("Theo nguồn (% có ít nhất 1 dấu vết):")
    print((100 * flags["có ít nhất 1 dấu vết"].groupby([df["source"], lab]).mean()).round(1).unstack().to_string())


def extra_format_report(df):
    """Các khác biệt định dạng khác giữa 2 nhãn (chỉ thống kê)."""
    lab = df["label_binary"]
    rows = []
    for name, pat in EXTRA_FEATURES.items():
        m = df["text"].str.contains(pat, regex=True)
        rows.append({"đặc điểm": name, "human%": round(100 * m[lab == 0].mean(), 1),
                     "chatgpt%": round(100 * m[lab == 1].mean(), 1)})
    print("\n=== KHÁC BIỆT ĐỊNH DẠNG KHÁC (sau chuẩn hoá, chưa sửa) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    m = df["text"].str.contains(WORD_DOT_WORD_RE)
    print("'word.Word' theo nguồn (% trong nhóm):")
    print((100 * m.groupby([df["source"], lab.map(NAME)]).mean()).round(1).unstack().to_string())


def raw_report(df, tokenizer_name):
    lab = df["label_binary"].map(NAME)
    print_counts(df, "dữ liệu thô")

    q_lines = df.groupby("question")["line_no"].nunique()
    print(f"\nSố dòng json có câu trả lời: {df['line_no'].nunique()} | số câu hỏi (text) khác nhau: {len(q_lines)} | "
          f"câu hỏi xuất hiện ở >1 dòng: {(q_lines > 1).sum()}")

    empty = df["text"].str.strip() == ""
    print("Rỗng hoặc chỉ khoảng trắng:", empty.groupby(lab).sum().to_dict())
    cnt = df["text"].value_counts()
    dup_rows = df["text"].map(cnt) > 1
    print(f"Text trùng chính xác: {(cnt > 1).sum()} text khác nhau, xuất hiện ở {dup_rows.sum()} dòng; "
          f"theo nhãn (số dòng): {dup_rows.groupby(lab).sum().to_dict()}")
    labs_per_text = df.groupby("text")["label_binary"].nunique()
    print(f"Text xuất hiện ở CẢ 2 nhãn: {(labs_per_text > 1).sum()}")

    print("\n=== ĐỘ DÀI (ký tự) ===")
    print(pct_table({NAME[k]: g.str.len() for k, g in df.groupby("label_binary")["text"]}).to_string(index=False))
    n_tok = count_tokens(df["text"], tokenizer_name)
    print(f"\n=== ĐỘ DÀI (token {tokenizer_name}, gồm <s></s>) ===")
    print(pct_table({NAME[k]: n_tok[(df["label_binary"] == k).to_numpy()] for k in (0, 1)}).to_string(index=False))
    artifact_report(df, "TRƯỚC chuẩn hoá")


# ---------------------------------------------------------------- (2)-(5) làm sạch
def drop_empty_and_errors(df, error_patterns, n_examples, seed):
    """Bỏ câu trả lời rỗng và câu trả lời chứa chuỗi lỗi/giao diện ChatGPT. In quy tắc + ví dụ ngẫu nhiên."""
    empty = df["text"].str.strip() == ""
    err_re = re.compile("|".join(f"(?:{p})" for p in error_patterns))
    err = df["text"].str.contains(err_re) & ~empty
    print("\n=== CHUỖI LỖI / GIAO DIỆN CHATGPT ===")
    print("Quy tắc: bỏ câu trả lời nếu text khớp (re.search) BẤT KỲ mẫu nào dưới đây:")
    for p in error_patterns:
        m = df["text"].str.contains(p, regex=True) & ~empty
        print(f"  {p!r:70s} khớp {int(m.sum())} dòng {m.groupby(df['label_binary'].map(NAME)).sum().to_dict()}")
    print(f"Tổng bị bỏ do lỗi/giao diện: {int(err.sum())} dòng "
          f"{err.groupby(df['label_binary'].map(NAME)).sum().to_dict()} | rỗng: {int(empty.sum())}")
    ex = df[err].sample(n=min(n_examples, int(err.sum())), random_state=seed)
    print(f"{len(ex)} ví dụ ngẫu nhiên (seed {seed}); in 150 ký tự đầu ... 150 ký tự cuối, độ dài thật trong ngoặc:")
    for i, (_, r) in enumerate(ex.iterrows(), 1):
        t = r["text"]
        body = repr(t) if len(t) <= 300 else f"{t[:150]!r} ... {t[-150:]!r}"
        print(f"  [{i}] {r['source']}/{NAME[r['label_binary']]} ({len(t)} ký tự): {body}")
    return df[~empty & ~err].copy()


def _pair_quotes(text, positions):
    """Ghép tuần tự các vị trí nháy thành cặp (1-2, 3-4, ...) và bỏ khoảng trắng NGAY PHÍA TRONG mỗi cặp.

    Thận trọng: chỉ sửa khi ghép được TẤT CẢ: số nháy chẵn, mỗi nháy mở đứng sau đầu chuỗi/khoảng trắng/( [,
    mỗi nháy đóng đứng trước cuối chuỗi/khoảng trắng/dấu câu/) ], nội dung giữa cặp không rỗng.
    Không ghép được -> giữ nguyên câu trả lời. Trả về (text, trạng thái: none | paired | unpaired).
    """
    if not positions:
        return text, "none"
    if len(positions) % 2:
        return text, "unpaired"
    drop = set()
    for o, c in zip(positions[0::2], positions[1::2]):
        ok_open = o == 0 or text[o - 1] in OPEN_BEFORE
        ok_close = c == len(text) - 1 or text[c + 1] in CLOSE_AFTER
        if not (ok_open and ok_close) or text[o + 1:c].strip() == "":
            return text, "unpaired"
        if text[o + 1] == " ":
            drop.add(o + 1)
        if text[c - 1] == " ":
            drop.add(c - 1)
    if not drop:
        return text, "none"
    return "".join(ch for i, ch in enumerate(text) if i not in drop), "paired"


def fix_quotes(text):
    """Bỏ khoảng trắng phía trong cặp nháy: '" x "' -> '"x"', "' x '" -> "'x'".

    Nháy kép: xét mọi dấu ". Nháy đơn: CHỈ xét dấu ' đứng riêng (2 bên là khoảng trắng/đầu/cuối chuỗi, hoặc
    trước là ( [, sau là dấu câu) -> dấu ' trong từ (don't, it's, John's) và ' dính vào từ không bị đụng tới.
    Trả về (text, trạng thái nháy kép, trạng thái nháy đơn); 'unpaired' chỉ tính khi câu thật sự cần sửa.
    """
    dq_pos = [i for i, ch in enumerate(text) if ch == '"']
    text, dq = _pair_quotes(text, dq_pos)
    if dq == "unpaired" and not NEEDS_DQ_FIX_RE.search(text):
        dq = "none"   # có nháy lẻ (vd 5" = inch) nhưng không có khoảng trắng phía trong -> không cần sửa
    sq_pos = [i for i, ch in enumerate(text) if ch == "'"
              and (i == 0 or text[i - 1] in OPEN_BEFORE)
              and (i == len(text) - 1 or text[i + 1] in CLOSE_AFTER)]
    text, sq = _pair_quotes(text, sq_pos)
    return text, dq, sq


def normalize(text):
    """Chuẩn hoá (b). Trả về (text, trạng thái nháy kép, trạng thái nháy đơn)."""
    for rx, rep in NORM_RULES:
        text = rx.sub(rep, text)
    text, dq, sq = fix_quotes(text)
    # bỏ khoảng trắng ở 2 đầu: nếu không, xuống dòng đầu/cuối biến thành khoảng trắng đầu/cuối
    # (tokenizer RoBERTa phân biệt "The" và " The" -> tạo ra dấu vết mới)
    return MULTI_SPACE_RE.sub(" ", text).strip(), dq, sq


def quote_report(before, df, dq, sq, n_examples, seed):
    """Số câu trả lời đã ghép cặp / không ghép được theo nhãn + vài ví dụ trước/sau để kiểm tra bằng mắt."""
    lab = df["label_binary"].map(NAME)
    print("\n=== GHÉP CẶP NHÁY (số câu trả lời theo nhãn) ===")
    for name, st in (("nháy kép", dq), ("nháy đơn", sq)):
        t = pd.crosstab(st, lab)
        print(f"{name}:\n{t.to_string()}")
    rng = np.random.RandomState(seed)
    for name, st in (("nháy đơn", sq), ("nháy kép", dq)):
        for status in ("paired", "unpaired"):
            idx = np.flatnonzero(st.to_numpy() == status)
            if not len(idx):
                continue
            pick = rng.choice(idx, size=min(n_examples, len(idx)), replace=False)
            print(f"\n{len(pick)} ví dụ ngẫu nhiên {name} = {status} (seed {seed}), 120 ký tự quanh dấu nháy cần xét:")
            q = "'" if name == "nháy đơn" else '"'
            spaced = re.compile(f"(?:^|\\s){q}|{q}(?:\\s|$)")  # nháy có khoảng trắng bên cạnh (bỏ qua ' trong từ)
            for i in pick:
                b, a = before.iloc[i], df["text"].iloc[i]
                ma, mb = spaced.search(a), spaced.search(b)
                k, kb = max(0, (ma.start() if ma else 0) - 40), max(0, (mb.start() if mb else 0) - 40)
                if status == "paired":
                    print(f"  [{NAME[df['label_binary'].iloc[i]]}] TRƯỚC: {b[kb:kb + 120]!r}")
                    print(f"            SAU  : {a[k:k + 120]!r}")
                else:
                    print(f"  [{NAME[df['label_binary'].iloc[i]]}] GIỮ NGUYÊN: {a[k:k + 120]!r}")


def length_report(df, tokenizer_name, title):
    """Độ dài ký tự và token theo nhãn: toàn bộ và theo từng source."""
    n_chr = df["text"].str.len().to_numpy()
    n_tok = count_tokens(df["text"], tokenizer_name)
    lab, src = df["label_binary"].to_numpy(), df["source"].to_numpy()
    for unit, v in (("ký tự", n_chr), (f"token {tokenizer_name}, gồm <s></s>", n_tok)):
        rows = []
        for s in ["TẤT CẢ"] + sorted(df["source"].unique()):
            in_s = np.ones(len(df), bool) if s == "TẤT CẢ" else src == s
            t = pct_table({NAME[k]: v[in_s & (lab == k)] for k in (0, 1)})
            rows.append(t.assign(source=s))
        print(f"\n=== ĐỘ DÀI ({unit}) {title} ===")
        print(pd.concat(rows)[["source", "label", "n", "median", "p90", "p99", "max"]].to_string(index=False))


def dedup(df):
    """Bỏ trùng chính xác (sau chuẩn hoá), giữ lần đầu theo line_no; text có ở cả 2 nhãn -> bỏ hết."""
    labs = df.groupby("text")["label_binary"].nunique()
    both = labs[labs > 1].index
    if len(both):
        print(f"[WARN] {len(both)} text xuất hiện ở cả 2 nhãn sau chuẩn hoá -> bỏ hết các dòng đó")
    df = df[~df["text"].isin(both)]
    before = len(df)
    df = df.sort_values("line_no", kind="stable").drop_duplicates("text", keep="first")
    print(f"[INFO] Bỏ trùng: {before - len(df)} dòng bị bỏ")
    return df


def balance_per_source(df, seed):
    """Trong mỗi source: lấy mẫu bớt nhãn đông hơn cho bằng nhãn ít hơn."""
    parts = []
    for _, g in df.groupby("source"):
        n = g["label_binary"].value_counts().min()
        for _, gl in g.groupby("label_binary"):
            parts.append(gl.sample(n=n, random_state=seed) if len(gl) > n else gl)
    return pd.concat(parts).sort_values("line_no", kind="stable")


# ---------------------------------------------------------------- (6) chia theo câu hỏi
def assign_by_question(df, split_names, ratios, strata_cols, seed):
    """Xếp nguyên mỗi question_id vào 1 tập sao cho từng stratum gần tỉ lệ mục tiêu (greedy, như split_data.py)."""
    stratum = df[strata_cols].astype(str).agg("|".join, axis=1)
    strata = sorted(stratum.unique())
    s_idx = {s: i for i, s in enumerate(strata)}
    gcount = {}
    for qid, st in zip(df["question_id"], stratum):
        gcount.setdefault(qid, np.zeros(len(strata)))[s_idx[st]] += 1
    qids = list(gcount)
    rng = np.random.RandomState(seed)
    # Xếp theo thứ tự ngẫu nhiên, KHÔNG xếp câu hỏi lớn trước như split_data.py: ở HC3 câu hỏi lớn
    # thường nhiều câu trả lời người viết -> dồn hết vào train, val/test lệch nhãn (đã thử: 74/13/13, 55.6% chatgpt)
    rng.shuffle(qids)

    total = np.sum(list(gcount.values()), axis=0)
    target = {n: total * r for n, r in zip(split_names, ratios)}
    cur = {n: np.zeros(len(strata)) for n in split_names}
    owner = {}
    for q in qids:
        best, best_cost = None, None
        for n in split_names:
            new = cur[n] + gcount[q]
            cost = float(np.sum(((new - target[n]) ** 2 - (cur[n] - target[n]) ** 2) / np.maximum(total, 1)))
            if best_cost is None or cost < best_cost:
                best, best_cost = n, cost
        owner[q] = best
        cur[best] += gcount[q]
    return df["question_id"].map(owner)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/hc3_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths, ccfg, scfg = cfg["paths"], cfg["clean"], cfg["split"]
    tok_name = cfg["data"]["tokenizer_name"]
    out_dir = Path(paths["processed_hc3_dir"])
    if "raw" in out_dir.parts:
        raise ValueError("Không được ghi output vào thư mục raw")
    ratios = [scfg["train_ratio"], scfg["val_ratio"], scfg["test_ratio"]]
    if abs(sum(ratios) - 1) > 1e-9:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1, hiện là {sum(ratios)}")

    # (0)-(1) đọc + thống kê thô
    df = flatten(read_jsonl(Path(paths["raw_hc3"])))
    raw_report(df, tok_name)

    # (2) bỏ rỗng + lỗi/giao diện
    df = drop_empty_and_errors(df, ccfg["error_patterns"], ccfg["n_error_examples"], ccfg["seed"])
    print_counts(df, "sau khi bỏ rỗng + lỗi/giao diện")

    # (3) chuẩn hoá (b), giống nhau cho cả 2 nhãn
    before = df["text"].copy()
    res = [normalize(t) for t in df["text"]]
    df["text"] = [r[0] for r in res]
    dq = pd.Series([r[1] for r in res], index=df.index)
    sq = pd.Series([r[2] for r in res], index=df.index)
    quote_report(before, df, dq, sq, ccfg["n_quote_examples"], ccfg["seed"])
    empty = df["text"] == ""
    if empty.any():
        print(f"[WARN] {int(empty.sum())} câu trả lời rỗng sau chuẩn hoá -> bỏ")
        df = df[~empty]
    print_counts(df, "sau chuẩn hoá")
    length_report(df, tok_name, "SAU chuẩn hoá")
    artifact_report(df, "SAU chuẩn hoá")
    extra_format_report(df)

    # (4) bỏ trùng
    df = dedup(df)
    print_counts(df, "sau bỏ trùng")

    # (5) cân bằng trong từng source
    df = balance_per_source(df, ccfg["seed"])
    print_counts(df, "sau cân bằng 50/50 trong từng source")

    # (6) question_id = 1 mã cho mỗi text câu hỏi (câu hỏi lặp ở 2 dòng json dùng chung mã)
    qcodes = pd.factorize(df["question"])[0]
    df["question_id"] = [f"hc3_q{c:05d}" for c in qcodes]
    names = ["train", "val", "test"]
    df["_split"] = assign_by_question(df, names, ratios, scfg["stratify_columns"], scfg["random_seed"])
    parts = {n: df[df["_split"] == n] for n in names}

    # Kiểm tra không rò rỉ: không question_id và không text nào ở 2 tập
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(parts[a]["question_id"]) & set(parts[b]["question_id"]):
                raise ValueError(f"Có question_id nằm ở cả {a} và {b}")
            if set(parts[a]["text"]) & set(parts[b]["text"]):
                raise ValueError(f"Có text nằm ở cả {a} và {b}")
    print("\n[INFO] Kiểm tra rò rỉ: OK (không question_id/text nào nằm ở 2 tập)")
    for n in names:
        p = parts[n]
        print_counts(p, f"tập {n}: {len(p)} mẫu ({100 * len(p) / len(df):.1f}%), "
                        f"{p['question_id'].nunique()} câu hỏi, %chatgpt={100 * p['label_binary'].mean():.1f}")

    if not scfg.get("write_files", False):
        print("\n[INFO] split.write_files = false -> CHƯA ghi file (chờ nhóm xác nhận).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["text", "label_binary", "source", "question_id"]
    for n in names:
        p = out_dir / f"{n}.csv"
        parts[n][keep].sample(frac=1, random_state=scfg["random_seed"]).to_csv(p, index=False, encoding="utf-8")
        print(f"[INFO] Ghi {len(parts[n])} bản ghi -> {p}")

    # Kiểm tra lại rò rỉ trên chính file đã ghi (đọc lại từ đĩa)
    disk = {n: pd.read_csv(out_dir / f"{n}.csv", keep_default_na=False) for n in names}
    for n in names:
        if len(disk[n]) != len(parts[n]) or set(disk[n]["text"]) != set(parts[n]["text"]):
            raise ValueError(f"{n}.csv đọc lại không khớp dữ liệu đã ghi")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(disk[a]["question_id"]) & set(disk[b]["question_id"]) or set(disk[a]["text"]) & set(disk[b]["text"]):
                raise ValueError(f"File đã ghi: có question_id/text nằm ở cả {a} và {b}")
    print("[INFO] Kiểm tra lại trên file đã ghi: OK (khớp dữ liệu, không rò rỉ)")
    for n in names:
        print_counts(disk[n], f"file {n}.csv ({len(disk[n])} mẫu)")

    # Độ dài token trên train: % bị cắt ở các max_length ứng viên (để chọn max_length cho config)
    n_tok = count_tokens(parts["train"]["text"], tok_name)
    lab = parts["train"]["label_binary"].to_numpy()
    print(f"\n=== TRAIN: % mẫu bị cắt (số token {tok_name} > max_length) ===")
    for L in scfg["max_length_candidates"]:
        print(f"max_length={L}: tất cả {100 * (n_tok > L).mean():.2f}% | human {100 * (n_tok[lab == 0] > L).mean():.2f}% "
              f"| chatgpt {100 * (n_tok[lab == 1] > L).mean():.2f}%")


if __name__ == "__main__":
    main()
