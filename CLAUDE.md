# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Binary SMS scam (smishing) detection: `label_binary` 1 = smishing, 0 = ham. Comments, config notes and log messages are written in Vietnamese; match that when editing existing files. README.md, PROJECT_CONTEXT.md and requirements.txt are currently empty. Stub (empty) files exist for the planned stages: `src/evaluate/evaluate_models.py`, `src/models/train_{roberta,deberta,hc3_classifier}.py`, `src/generate/generate_ai_messages.py`, `src/features/tokenizer_utils.py`.

## Commands

All scripts run from the project root (paths in configs are relative to it), take `--config`, and must be run in this order. There is no test suite or linter.

```
python src/data/merge_labels.py    --config configs/data_config.yaml   # raw -> data/interim/merged.csv
python src/data/clean_text.py      --config configs/data_config.yaml   # -> data/interim/cleaned.csv
python src/data/split_data.py      --config configs/data_config.yaml   # -> data/processed/phishing_detection/{train,val,test}.csv
python src/features/tfidf_vectorizer.py --config configs/model_config.yaml   # -> models/baseline/{X,y}_*.npz/npy + vectorizer
python src/models/train_lr.py      --config configs/model_config.yaml
python src/models/train_mnb.py     --config configs/model_config.yaml
python notebooks/inspect_raw.py    # prints schema/columns of raw datasets (used to fill data_config.yaml)
```

Dependencies (not pinned anywhere): pandas, pyyaml, scikit-learn, scipy, joblib, numpy.

## Architecture

A linear file-based pipeline; each script has exactly one job and communicates with the next only via files on disk. Keep that separation (e.g. merge does not clean, clean does not split, train scripts never read test).

- **Configs drive everything.** `configs/data_config.yaml` (paths, per-dataset column/label mappings, `drop_labels`, split ratios/seed/stratify keys) and `configs/model_config.yaml` (TF-IDF params, LR/MNB grids). Change behavior there rather than in code.
- **Two raw sources** under `data/raw/` (read-only; `merge_labels.py` refuses to write there): SmishTank (smishing only, latin-1, text in `MainText`) and Mishra & Soni (ham/spam/smishing, inconsistent label casing). Spam is mapped to NA then dropped in cleaning because it is ambiguous. Unknown raw labels raise instead of being skipped.
- **Cleaning is deliberately light**: fixes mojibake, strips control chars, dedups (same text with conflicting labels is dropped entirely). It intentionally does NOT lowercase or strip URLs/phone numbers/punctuation, since those are phishing signals; lowercasing happens only inside the TF-IDF vectorizer.
- **Split** is stratified on `(label_binary, source)` and asserts no text overlaps between train/val/test.
- **Baselines**: TF-IDF is fit on train only. LR and MNB tune one hyperparameter on val by F1 using a tolerance rule: pick the simplest setting within `selection_tolerance` of the best F1 (smallest `C` for LR, largest `alpha` for MNB, opposite directions). Class balancing uses `class_weight` for LR and `compute_sample_weight` for MNB. Outputs: `models/baseline/*.joblib` and `results/metrics/*_val_search.csv`. Test-set evaluation is reserved for `evaluate_models.py`.

## Data caveat

Ham examples come only from Mishra & Soni while most smishing comes from SmishTank, so models may learn source artifacts. That is why splits stratify by `source`; report metrics per source when evaluating.

## Repo note

This directory sits inside a git repository rooted at the user's home directory (many unrelated untracked files); stage only paths under `scam-detection-project/`.
