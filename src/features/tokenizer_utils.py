"""tokenizer_utils.py - Đọc dữ liệu và biến tin nhắn thành token cho RoBERTa/DeBERTa.

Mỗi model dùng ĐÚNG tokenizer của nó (get_tokenizer(model_name)). Không có script CLI riêng:
file này là module dùng chung cho train_roberta.py, train_deberta.py và evaluate_models.py.
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name):
    """Tải tokenizer đúng với model (RoBERTa: BPE, DeBERTa-v3: SentencePiece)."""
    return AutoTokenizer.from_pretrained(model_name)


def load_split(csv_path, text_column, label_column):
    """Đọc 1 file csv (train/val/test) -> (danh sách text, danh sách nhãn int)."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise KeyError(f"{csv_path}: thiếu cột '{col}'. Cột hiện có: {list(df.columns)}")
    print(f"[INFO] Đọc {len(df)} bản ghi từ {csv_path}")
    return df[text_column].astype(str).tolist(), df[label_column].astype(int).tolist()


class TextDataset(Dataset):
    """Giữ (text, nhãn); việc chuyển thành token làm theo từng batch trong collator."""

    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class Collator:
    """Gom 1 batch: tokenize, đệm (padding) tới tin dài nhất trong batch, cắt ở max_length."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, batch):
        texts = [t for t, _ in batch]
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch], dtype=torch.long)
        return enc


def make_loader(texts, labels, tokenizer, max_length, batch_size, shuffle):
    return DataLoader(TextDataset(texts, labels), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=Collator(tokenizer, max_length))
