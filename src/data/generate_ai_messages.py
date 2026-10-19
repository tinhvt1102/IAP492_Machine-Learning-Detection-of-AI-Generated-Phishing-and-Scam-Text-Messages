"""generate_ai_messages.py - Công cụ generator: dùng LLM sinh hàng loạt tin nhắn (lừa đảo + hợp lệ) cho bước robustness.

Ba giai đoạn (--stage):
  prompts : tạo prompt cho từng lần gọi (chủ đề lấy từ scam_type của IMC25 + nhóm trúng thưởng của Mishra & Soni;
            kèm vài tin mẫu của NGƯỜI lấy từ tập train để mô phỏng văn phong) -> <work_dir>/prompts/prompt_XXXX.txt + manifest.jsonl
  run     : (chỉ backend gemini) gọi API cho từng prompt, lưu phản hồi -> <work_dir>/responses/response_XXXX.txt
            Không có khoá API: bỏ qua giai đoạn này, tự dán từng prompt vào một chat LLM (ChatGPT/Gemini/Claude...) và
            lưu câu trả lời thành response_XXXX.txt trong thư mục responses/ (cùng số XXXX).
  parse   : đọc phản hồi, lọc (trùng, độ dài, giống tin mẫu), cắt theo số lượng mục tiêu, ghi scam_messages.txt và
            ham_messages.txt (định dạng "loai|noi dung") + generation_report.json (mô tả công cụ/số lượng/prompt cho báo cáo).
Sau đó: prepare_ai_testset.py chuẩn hoá về <URL>/<PHONE_NUMBER>/<EMAIL_ADDRESS> và loại tin trùng với dữ liệu người.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

TOKEN_RE = re.compile(r"[a-z0-9']+")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def jaccard(a, b):
    ta, tb = set(TOKEN_RE.findall(a.lower())), set(TOKEN_RE.findall(b.lower()))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def pick_seeds(train, flt, k, rng):
    """Chọn k tin mẫu của người từ tập train theo bộ lọc (source, scam_type, label_binary)."""
    d = train
    for col, val in flt.items():
        d = d[d[col] == val]
    if len(d) == 0:
        raise ValueError(f"Không có tin train khớp bộ lọc {flt}")
    d = d[d["text"].str.len().between(20, 300)] if (d["text"].str.len().between(20, 300)).any() else d
    idx = rng.sample(range(len(d)), min(k, len(d)))
    return d["text"].iloc[idx].tolist()


# ------------------------------------------------------------------ giai đoạn 1: prompts
def build_calls(cfg, train):
    g, plan = cfg["generator"], cfg["plan"]
    rng = random.Random(g["seed"])
    calls, cid = [], 1
    for label, topics in plan.items():
        for topic, t in topics.items():
            need = math.ceil(t["n"] * g["oversample"])
            n_calls = math.ceil(need / g["messages_per_call"])
            per = math.ceil(need / n_calls)
            for i in range(n_calls):
                seeds = pick_seeds(train, t["seed_filter"], g["seed_examples_per_call"], rng)
                situation = t["situations"][i % len(t["situations"])] if "situations" in t else ""
                prompt = cfg["prompts"][label].format(
                    n=per, topic=topic.replace("_", " "), description=t["description"], situation=situation,
                    entity_pct=cfg["entity_pct"][label], examples="\n".join(f"- {s}" for s in seeds))
                calls.append({"call_id": f"{cid:04d}", "label": label, "topic": topic, "n_requested": per,
                              "situation": situation, "seeds": seeds, "prompt": prompt})
                cid += 1
    return calls


def stage_prompts(cfg):
    train = pd.read_csv(cfg["paths"]["train_csv"], keep_default_na=False)
    wd = Path(cfg["paths"]["work_dir"])
    (wd / "prompts").mkdir(parents=True, exist_ok=True)
    (wd / "responses").mkdir(parents=True, exist_ok=True)
    calls = build_calls(cfg, train)
    with open(wd / "manifest.jsonl", "w", encoding="utf-8") as f:
        for c in calls:
            (wd / "prompts" / f"prompt_{c['call_id']}.txt").write_text(c["prompt"], encoding="utf-8")
            f.write(json.dumps({k: v for k, v in c.items() if k != "prompt"}, ensure_ascii=False) + "\n")
    by = pd.DataFrame(calls).groupby(["label", "topic"])["n_requested"].agg(["count", "sum"])
    print(f"[INFO] Tạo {len(calls)} prompt -> {wd / 'prompts'} (manifest: {wd / 'manifest.jsonl'})")
    print(by.rename(columns={"count": "so_lan_goi", "sum": "so_tin_yeu_cau"}).to_string())
    print(f"[INFO] Mỗi prompt yêu cầu LLM trả về JSON array. Lưu phản hồi thành {wd / 'responses'}/response_XXXX.txt "
          f"(XXXX trùng số của prompt), rồi chạy --stage parse.")


# ------------------------------------------------------------------ giai đoạn 2: run (gemini)
def call_gemini(prompt, model, temperature, key, retries):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}).encode()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            cands = data.get("candidates") or []
            if not cands or "content" not in cands[0]:
                reason = data.get("promptFeedback", {}).get("blockReason") or (cands[0].get("finishReason") if cands else "no_candidates")
                return None, f"blocked:{reason}"
            return "".join(p.get("text", "") for p in cands[0]["content"]["parts"]), "ok"
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries:
                wait = 20 * attempt
                print(f"[WARN] HTTP {e.code}, thử lại sau {wait}s (lần {attempt}/{retries})")
                time.sleep(wait)
                continue
            return None, f"http_{e.code}"
        except Exception as e:  # lỗi mạng
            if attempt < retries:
                time.sleep(10 * attempt)
                continue
            return None, f"error:{type(e).__name__}"
    return None, "failed"


def stage_run(cfg, limit=None):
    g, wd = cfg["generator"], Path(cfg["paths"]["work_dir"])
    key = os.environ.get(g["api_key_env"])
    if not key:
        raise SystemExit(f"[ERROR] Chưa đặt biến môi trường {g['api_key_env']} (khoá API). "
                         f"Không có khoá thì dùng chế độ thủ công: dán prompt vào chat LLM rồi chạy --stage parse.")
    calls = [json.loads(l) for l in (wd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    log = open(wd / "generation_log.jsonl", "a", encoding="utf-8")
    rounds, done = g.get("rounds", 3), 0
    for rnd in range(1, rounds + 1):
        todo = [c for c in calls if not (wd / "responses" / f"response_{c['call_id']}.txt").exists()]
        if not todo:
            break
        print(f"[INFO] Vòng {rnd}/{rounds}: còn {len(todo)}/{len(calls)} lần gọi chưa có phản hồi")
        for c in todo:
            if limit is not None and done >= limit:
                print(f"[INFO] Đã gọi {done} lần theo --limit; dừng (chạy lại không có --limit để gọi phần còn lại)")
                log.close()
                return
            prompt = (wd / "prompts" / f"prompt_{c['call_id']}.txt").read_text(encoding="utf-8")
            done += 1
            text, status = call_gemini(prompt, g["model"], g["temperature"], key, g["max_retries"])
            log.write(json.dumps({"call_id": c["call_id"], "backend": "gemini", "model": g["model"], "status": status,
                                  "round": rnd, "time": datetime.now(timezone.utc).isoformat(), "prompt_chars": len(prompt)}) + "\n")
            log.flush()
            print(f"[INFO] {c['call_id']} ({c['label']}/{c['topic']}): {status}")
            if text:
                (wd / "responses" / f"response_{c['call_id']}.txt").write_text(text, encoding="utf-8")
            time.sleep(g["sleep_seconds"])
        left = [c for c in calls if not (wd / "responses" / f"response_{c['call_id']}.txt").exists()]
        if left and rnd < rounds:
            print(f"[INFO] Còn {len(left)} lần gọi thất bại; nghỉ {g.get('round_wait_seconds', 90)}s rồi thử lại")
            time.sleep(g.get("round_wait_seconds", 90))
    left = [c["call_id"] for c in calls if not (wd / "responses" / f"response_{c['call_id']}.txt").exists()]
    log.close()
    print(f"[INFO] Xong. Còn thiếu {len(left)} lần gọi: {left}" if left else "[INFO] Xong: đủ phản hồi cho mọi prompt")


# ------------------------------------------------------------------ giai đoạn 3: parse
def parse_response(text):
    """Lấy danh sách chuỗi từ phản hồi: ưu tiên JSON array; nếu hỏng thì đọc theo từng dòng."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    s, e = t.find("["), t.rfind("]")
    if s != -1 and e > s:
        try:
            arr = json.loads(t[s:e + 1])
            return [str(x).strip() for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            pass
    items = []
    for line in t.splitlines():
        line = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', "", line).strip().strip('",')
        if line:
            items.append(line)
    return items


def stage_parse(cfg, llm_name):
    g, wd = cfg["generator"], Path(cfg["paths"]["work_dir"])
    out_dir = Path(cfg["paths"]["raw_out_dir"])
    if "raw" in Path(cfg["paths"]["work_dir"]).parts:
        raise ValueError("work_dir không được nằm trong thư mục raw")
    calls = [json.loads(l) for l in (wd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    kept = {}                  # (label, topic) -> list
    stats = {}
    seen = set()
    n_missing = 0
    for c in calls:
        f = wd / "responses" / f"response_{c['call_id']}.txt"
        key = (c["label"], c["topic"])
        st = stats.setdefault(key, {"yeu_cau": 0, "nhan_ve": 0, "loai_do_dai": 0, "loai_trung": 0, "loai_giong_mau": 0, "giu": 0})
        st["yeu_cau"] += c["n_requested"]
        if not f.exists():
            n_missing += 1
            continue
        msgs = parse_response(f.read_text(encoding="utf-8"))
        st["nhan_ve"] += len(msgs)
        for m in msgs:
            m = re.sub(r"\s+", " ", m).strip()
            if not (g["min_chars"] <= len(m) <= g["max_chars"]):
                st["loai_do_dai"] += 1
                continue
            norm = m.lower()
            if norm in seen:
                st["loai_trung"] += 1
                continue
            if any(jaccard(m, s) >= g["max_seed_overlap"] for s in c["seeds"]):
                st["loai_giong_mau"] += 1
                continue
            seen.add(norm)
            kept.setdefault(key, []).append(m)
    if n_missing:
        print(f"[WARN] Thiếu phản hồi cho {n_missing}/{len(calls)} lần gọi")

    # cắt theo số lượng mục tiêu mỗi chủ đề (giữ thứ tự sinh ra)
    targets = {(lab, tp): t["n"] for lab, tps in cfg["plan"].items() for tp, t in tps.items()}
    rows = {"scam": [], "ham": []}
    for (lab, tp), n in targets.items():
        got = kept.get((lab, tp), [])
        use = got[:n]
        stats.setdefault((lab, tp), {})["giu"] = len(use)
        stats[(lab, tp)]["muc_tieu"] = n
        if len(use) < n:
            print(f"[WARN] {lab}/{tp}: chỉ có {len(use)}/{n} tin sau khi lọc")
        rows[lab] += [(("ham_personal" if lab == "ham" else tp), m) for m in use]

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for lab, fname in (("scam", "scam_messages.txt"), ("ham", "ham_messages.txt")):
        head = f"# Sinh bởi generate_ai_messages.py | LLM: {llm_name} | ngày parse: {stamp} | {len(rows[lab])} tin\n"
        (out_dir / fname).write_text(head + "\n".join(f"{k}|{m}" for k, m in rows[lab]) + "\n", encoding="utf-8")
        print(f"[INFO] Ghi {len(rows[lab])} tin -> {out_dir / fname}")

    prompt_hash = hashlib.sha256(json.dumps(cfg["prompts"], sort_keys=True).encode()).hexdigest()[:12]
    report = {
        "cong_cu": "src/data/generate_ai_messages.py", "llm": llm_name, "ngay_parse": stamp,
        "so_lan_goi": len(calls), "so_lan_goi_thieu_phan_hoi": n_missing,
        "tham_so": {k: g[k] for k in ("seed", "oversample", "messages_per_call", "seed_examples_per_call",
                                       "min_chars", "max_chars", "max_seed_overlap")},
        "nguon_chu_de": "scam_type của IMC25 (banking, delivery, government, telecom, others, hey mum/dad) + nhóm trúng thưởng của Mishra & Soni; "
                        "tin hợp lệ: tin nhắn cá nhân (theo tin ham của Mishra & Soni). Tin mẫu lấy từ tập TRAIN.",
        "prompt_hash": prompt_hash, "prompts": cfg["prompts"],
        "theo_chu_de": {f"{k[0]}/{k[1]}": v for k, v in stats.items()},
        "tong": {"scam": len(rows["scam"]), "ham": len(rows["ham"])},
    }
    (wd / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + pd.DataFrame({f"{k[0]}/{k[1]}": v for k, v in stats.items()}).T.to_string())
    print(f"\n[INFO] Tổng: lừa đảo {len(rows['scam'])}, hợp lệ {len(rows['ham'])} | báo cáo: {wd / 'generation_report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/generator_config.yaml")
    ap.add_argument("--stage", required=True, choices=["prompts", "run", "parse"])
    ap.add_argument("--limit", type=int, default=None, help="--stage run: chỉ gọi tối đa N lần (để thử trước, ví dụ 1)")
    ap.add_argument("--llm_name", default=None, help="Tên LLM đã dùng (bắt buộc ở --stage parse), ví dụ 'ChatGPT web (GPT-x), 2026-10'")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage == "prompts":
        stage_prompts(cfg)
    elif args.stage == "run":
        stage_run(cfg, args.limit)
    else:
        if not args.llm_name:
            raise SystemExit("[ERROR] --stage parse cần --llm_name để ghi vào báo cáo (LLM nào đã sinh dữ liệu)")
        stage_parse(cfg, args.llm_name)


if __name__ == "__main__":
    main()
