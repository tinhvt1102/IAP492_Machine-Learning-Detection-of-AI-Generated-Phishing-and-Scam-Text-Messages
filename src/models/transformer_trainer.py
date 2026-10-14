"""transformer_trainer.py - Vòng lặp fine-tune dùng chung cho RoBERTa và DeBERTa-v3.

Mỗi epoch: train trên train.csv -> đánh giá trên val.csv -> nếu F1 val tốt nhất từ trước tới giờ
thì lưu model vào <output_dir>/best. KHÔNG đọc test.csv (test chỉ mở ở evaluate_models.py).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.evaluate.metrics_utils import compute_metrics
from src.features.tokenizer_utils import get_tokenizer, load_split, make_loader


def set_seed(seed):
    """Cố định seed để chạy lại ra kết quả gần giống nhau."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, use_fp16):
    """Chạy model trên 1 tập, trả về dict metric."""
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        y_pred += logits.argmax(dim=-1).cpu().tolist()
        y_true += batch["labels"].cpu().tolist()
    return compute_metrics(y_true, y_pred)


def main(model_key, out_key, tag):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/model_config.yaml")
    ap.add_argument("--data_dir", default=None, help="Thư mục chứa train.csv, val.csv (mặc định lấy từ config)")
    ap.add_argument("--output_dir", default=None, help="Nơi lưu model (mặc định lấy từ config)")
    ap.add_argument("--metrics_dir", default=None, help="Nơi lưu lịch sử metric (mặc định lấy từ config)")
    ap.add_argument("--smoke_test", action="store_true", help="Chạy thử rất nhỏ (128 tin, 1 epoch) để kiểm tra code")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg, paths = cfg["transformer"], cfg["paths"]
    model_name = cfg[model_key]["model_name"]
    data_dir = Path(args.data_dir or paths["processed_phishing_dir"])
    output_dir = Path(args.output_dir or paths[out_key])
    metrics_dir = Path(args.metrics_dir or paths["metrics_dir"])
    for p in (output_dir, metrics_dir):
        if "raw" in p.parts:
            raise ValueError("Không được ghi output vào thư mục raw")

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(tcfg["fp16"]) and device.type == "cuda"
    print(f"[INFO] Model: {model_name} | thiết bị: {device} | fp16: {use_fp16}")
    if device.type == "cpu":
        print("[WARN] Không thấy GPU, sẽ rất chậm. Trên Kaggle: Session options -> Accelerator -> GPU")

    # (1) đọc dữ liệu (chỉ train + val)
    tr_x, tr_y = load_split(data_dir / "train.csv", tcfg["text_column"], tcfg["label_column"])
    va_x, va_y = load_split(data_dir / "val.csv", tcfg["text_column"], tcfg["label_column"])
    epochs = tcfg["epochs"]
    if args.smoke_test:
        tr_x, tr_y, va_x, va_y, epochs = tr_x[:128], tr_y[:128], va_x[:64], va_y[:64], 1
        print("[INFO] SMOKE TEST: 128 train, 64 val, 1 epoch (kết quả không có ý nghĩa)")

    # (2) tokenizer riêng của model + DataLoader
    tokenizer = get_tokenizer(model_name)
    train_loader = make_loader(tr_x, tr_y, tokenizer, tcfg["max_length"], tcfg["batch_size"], shuffle=True)
    val_loader = make_loader(va_x, va_y, tokenizer, tcfg["max_length"], tcfg["batch_size"] * 2, shuffle=False)

    # (3) model 2 lớp (0 = hợp lệ, 1 = lừa đảo), optimizer, scheduler
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg["weight_decay"]))
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * tcfg["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    print(f"[INFO] {len(train_loader)} batch/epoch x {epochs} epoch = {total_steps} bước; max_length={tcfg['max_length']}")

    # (4) vòng lặp train: mỗi epoch train -> đánh giá val -> lưu nếu F1 val tốt nhất
    history, best_f1, best_epoch = [], -1.0, 0
    best_dir = output_dir / "best"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        m = evaluate(model, val_loader, device, use_fp16)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}}
        history.append(row)
        print(f"[INFO] epoch {epoch}: train_loss={row['train_loss']:.4f} | val F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} FPR={m['fpr']:.4f} Acc={m['accuracy']:.4f}")
        if m["f1"] > best_f1:
            best_f1, best_epoch = m["f1"], epoch
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"[INFO] Lưu model tốt nhất (epoch {epoch}, val F1={best_f1:.4f}) -> {best_dir}")

    # (5) ghi lịch sử metric
    metrics_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    hist["selected"] = hist["epoch"] == best_epoch
    out_csv = metrics_dir / f"{tag}_val_history.csv"
    hist.to_csv(out_csv, index=False)
    print(f"[INFO] Ghi lịch sử val -> {out_csv} | chọn epoch {best_epoch} (val F1={best_f1:.4f})")
