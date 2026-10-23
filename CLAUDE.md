# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Binary SMS scam (smishing) detection: `label_binary` 1 = smishing, 0 = ham. English only. Comments, config notes and log messages are written in Vietnamese; match that when editing existing files.

Data: IMC25 (English, smishing only) + Mishra & Soni (ham/smishing). SmishTank was DROPPED; ignore any old numbers based on it. Final dataset: 9,666 messages, balanced 50/50, split 70/15/15 (6,766/1,450/1,450).

README.md, PROJECT_CONTEXT.md and requirements.txt are currently empty. Stub (empty) files still to be written: `src/models/train_hc3_classifier.py`, `src/generate/generate_ai_messages.py`.

## Commands

All scripts run from the project root (paths in configs are relative to it), take `--config`, and must be run in this order. There is no test suite or linter.

```
python src/data/merge_labels.py    --config configs/data_config.yaml   # raw -> data/interim/merged.csv
python src/data/clean_text.py      --config configs/data_config.yaml   # -> data/interim/cleaned_full.csv
python src/data/balance_data.py    --config configs/data_config.yaml   # -> data/interim/cleaned.csv
python src/data/split_data.py      --config configs/data_config.yaml   # -> data/processed/phishing_detection/{train,val,test}.csv
python src/features/tfidf_vectorizer.py --config configs/model_config.yaml   # -> models/baseline/{X,y}_*.npz/npy + vectorizer
python src/models/train_lr.py      --config configs/model_config.yaml
python src/models/train_mnb.py     --config configs/model_config.yaml
python src/evaluate/evaluate_models.py --config configs/model_config.yaml --models lr mnb roberta deberta
```

RoBERTa/DeBERTa were fine-tuned on Kaggle GPU (train_roberta.py, train_deberta.py, shared logic in transformer_trainer.py); checkpoints are in `models/{roberta,deberta}/best/`.

Dependencies (not pinned anywhere): pandas, pyyaml, scikit-learn, scipy, joblib, numpy, torch, transformers, sentencepiece, protobuf.

## Architecture

A linear file-based pipeline; each script has exactly one job and communicates with the next only via files on disk. Keep that separation (e.g. merge does not clean, clean does not split, train scripts never read test).

- **Configs drive everything.** `configs/data_config.yaml` and `configs/model_config.yaml`. Change behavior there rather than in code.
- **Two raw sources** under `data/raw/` (read-only): IMC25 (smishing only; English rows kept; scam_type spam and wrong number dropped) and Mishra & Soni (ham/spam/smishing, inconsistent label casing; spam dropped in cleaning). `balance_data.py` subsamples IMC25 (stratified by scam_type, seed 42) so label 1 = 50%.
- **Cleaning is deliberately light**: fixes mojibake, strips control chars, normalizes URL/email/phone to placeholders, dedups. It does NOT lowercase or strip punctuation; lowercasing happens only inside the TF-IDF vectorizer.
- **Split** is grouped by message template and stratified on `(label_binary, source)`, and asserts no text or group overlaps between train/val/test.
- **Baselines**: TF-IDF is fit on train only. LR and MNB tune one hyperparameter on val by F1 with a tolerance rule: pick the simplest setting within `selection_tolerance` of the best F1.
- **Test set** is read only by `evaluate_models.py`. Never tune models or hyperparameters from test results.

## Data caveat

All ham comes from Mishra & Soni while 89.8% of smishing comes from IMC25, so models may learn source, length or URL/phone artifacts. Report metrics per source when evaluating.

## RQ2 (HC3)

- Independent of RQ1: fine-tune roberta-base on HC3 (human vs ChatGPT), evaluate only on the HC3 test split, never on SMS.
- Do not modify RQ1 files (configs/model_config.yaml, configs/data_config.yaml, src/data/*, train_lr/mnb/roberta/deberta.py, src/evaluate/evaluate_models.py).
- HC3 files use the hc3 prefix: data/processed/hc3, models/hc3_roberta, results/metrics/hc3_*, configs/hc3_config.yaml.
- Never write into data/raw. Do not invent numbers or assumptions; ask when information is missing.

## Repo note

This directory sits inside a git repository rooted at the user's home directory (many unrelated untracked files); stage only paths under `scam-detection-project/`.
