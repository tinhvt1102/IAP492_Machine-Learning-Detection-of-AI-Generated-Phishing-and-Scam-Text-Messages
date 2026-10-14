"""train_roberta.py - Fine-tune RoBERTa-base phân loại tin lừa đảo (1) / hợp lệ (0).

Chạy: python src/models/train_roberta.py --config configs/model_config.yaml
Phần logic nằm ở transformer_trainer.py (dùng chung với DeBERTa).
"""
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import `src.*` khi chạy từ CLI
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer_trainer import main  # noqa: E402

if __name__ == "__main__":
    main(model_key="roberta", out_key="roberta_dir", tag="roberta")
