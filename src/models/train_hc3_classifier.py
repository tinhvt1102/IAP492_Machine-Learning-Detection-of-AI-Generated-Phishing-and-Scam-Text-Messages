"""train_hc3_classifier.py - RQ2: fine-tune roberta-base phân biệt người viết (0) / ChatGPT viết (1) trên HC3.

Chạy: python src/models/train_hc3_classifier.py --config configs/hc3_config.yaml
Dùng lại NGUYÊN vòng lặp của transformer_trainer.py (không sửa file đó): mỗi epoch train -> đánh giá val ->
lưu model + tokenizer có F1 val tốt nhất vào models/hc3_roberta/best/, ghi lịch sử val ra
results/metrics/hc3_val_history.csv. Chỉ đọc train.csv và val.csv, KHÔNG đọc test.csv.
Siêu tham số giữ giống RQ1 (chưa tinh chỉnh), xem mục transformer trong configs/hc3_config.yaml.
"""
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import `src.*` khi chạy từ CLI
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer_trainer import main  # noqa: E402

if __name__ == "__main__":
    main(model_key="hc3_roberta", out_key="hc3_roberta_dir", tag="hc3")
