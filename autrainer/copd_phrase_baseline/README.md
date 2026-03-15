# KAP COPD

This repository contains the code, processed tables, saved results, and final paper for my KAP project on computational voice biomarkers for COPD.

The project tests whether Danish speech contains enough acoustic information to distinguish speakers with COPD from controls. It compares phrase-based, passage-vowel, and sustained-vowel pipelines using eGeMAPS acoustic features and classical machine learning.

The paper submitted with this project is included in this repository.

## Included

- speaker labels and metadata
- phrase and passage-vowel segment tables
- phrase, passage-vowel, and sustained-vowel eGeMAPS feature tables
- saved classical ML evaluation outputs used for the paper
- QC tables and compact paper tables
- Autrainer phrase-baseline data, config, and helper scripts
- original pipeline files kept under `legacy/original_pipeline/`
- final paper for the KAP submission

## Repo layout

- `data/metadata/` speaker labels and metadata
- `data/interim/` segment tables and sustained-vowel inventory
- `data/processed/` feature tables
- `results/classical_ml/` saved evaluation runs
- `results/paper_tables/` compact paper tables
- `results/qc/` QC tables
- `autrainer/` phrase baseline
- `src/kap_copd/` shared code
- `scripts/` command-line entry points
- `legacy/original_pipeline/` original working files
- paper file included in the repository root or paper folder, depending on the final submission structure

## Main result files

Saved phrase and passage-vowel evaluation run:

- `results/classical_ml/run_20260118_185627/phrase_summary.csv`
- `results/classical_ml/run_20260118_185627/vowel_summary.csv`
- `results/classical_ml/run_20260118_185627/KAP_COPD_Evaluation_Report.xlsx`

Sustained-vowel evaluation runs:

- `results/classical_ml/sustained_vowels_with_age_gender/summary.csv`
- `results/classical_ml/sustained_vowels_without_age_gender/summary.csv`

Compact paper tables:

- `results/paper_tables/main_results.csv`
- `results/paper_tables/dataset_overview.csv`
- `results/paper_tables/label_overview.csv`
- `results/paper_tables/phrase_age_gender_ablation.csv`
- `results/paper_tables/sustained_vowel_results.csv`
- `results/paper_tables/sustained_vowel_inventory_summary.csv`

## Setup

Editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

You can still use `PYTHONPATH=src` instead of `pip install -e .`, but the editable install is the cleaner option for running the scripts directly.

## Commands

Build QC tables:

```bash
python scripts/build_qc_tables.py
```

Export compact paper tables:

```bash
python scripts/export_paper_tables.py
```

Run the phrase age/gender ablation:

```bash
python scripts/run_age_gender_ablation.py
```

Run classical ML evaluation on a feature table:

```bash
python scripts/run_classical_eval.py \
  --csv data/processed/phrase_features_with_labels.csv \
  --out_dir results/classical_ml/recomputed_phrase
```

Index sustained vowels from the raw speaker-folder layout:

```bash
python scripts/build_sustained_vowel_index.py \
  --audio_root /path/to/raw_audio_root \
  --out_csv data/interim/sustained_vowel_inventory.csv
```

Extract sustained-vowel features and merge labels:

```bash
python scripts/build_sustained_vowel_features.py \
  --audio_root /path/to/raw_audio_root \
  --labels_csv data/metadata/speaker_labels.csv \
  --out_inventory_csv data/interim/sustained_vowel_inventory.csv \
  --out_features_csv data/processed/sustained_vowel_features_eGeMAPS.csv \
  --out_labeled_csv data/processed/sustained_vowel_features_with_labels.csv \
  --out_summary_csv results/qc/sustained_vowel_inventory_summary.csv
```

Run the repo check:

```bash
python scripts/check_repo.py
```

## Notes

- The committed processed tables are the working dataset for this repository.
- Raw audio is not included.
- The sustained-vowel branch is separate from the passage-vowel branch.
- `results/classical_ml/run_20260118_185627/` is a saved evaluation artifact kept for the paper submission.
- The Autrainer preparation script can rebuild the fold directories from raw audio if needed.
- The final paper is included in this repository as part of the KAP submission.