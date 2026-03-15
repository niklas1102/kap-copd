from __future__ import annotations

from pathlib import Path
import pandas as pd


def _maybe_load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build_dataset_summary(repo_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_root = Path(repo_root)
    labels = pd.read_csv(repo_root / 'data/metadata/speaker_labels.csv')
    phrases = pd.read_csv(repo_root / 'data/interim/phrases.csv')
    vowels = pd.read_csv(repo_root / 'data/interim/vowels.csv')
    phrase_feat = pd.read_csv(repo_root / 'data/processed/phrase_features_with_labels.csv')
    vowel_feat = pd.read_csv(repo_root / 'data/processed/vowel_features_with_labels.csv')
    sustained_inventory = _maybe_load(repo_root / 'data/interim/sustained_vowel_inventory.csv')
    sustained_feat = _maybe_load(repo_root / 'data/processed/sustained_vowel_features_with_labels.csv')

    rows = [
        {'table': 'speaker_labels', 'rows': len(labels), 'speakers': labels['file_id'].nunique()},
        {'table': 'phrases', 'rows': len(phrases), 'speakers': phrases['file_id'].nunique()},
        {'table': 'phrase_features_with_labels', 'rows': len(phrase_feat), 'speakers': phrase_feat['file_id'].nunique()},
        {'table': 'vowels', 'rows': len(vowels), 'speakers': vowels['file_id'].nunique()},
        {'table': 'vowel_features_with_labels', 'rows': len(vowel_feat), 'speakers': vowel_feat['file_id'].nunique()},
    ]
    if sustained_inventory is not None:
        rows.append({'table': 'sustained_vowel_inventory', 'rows': len(sustained_inventory), 'speakers': sustained_inventory['file_id'].nunique()})
    if sustained_feat is not None:
        rows.append({'table': 'sustained_vowel_features_with_labels', 'rows': len(sustained_feat), 'speakers': sustained_feat['file_id'].nunique()})
    summary = pd.DataFrame(rows)

    label_summary = pd.DataFrame([
        {'metric': 'n_speakers', 'value': int(labels['file_id'].nunique())},
        {'metric': 'n_control', 'value': int((labels['copd'] == 0).sum())},
        {'metric': 'n_copd', 'value': int((labels['copd'] == 1).sum())},
        {'metric': 'mean_age', 'value': float(labels['age'].mean())},
        {'metric': 'female', 'value': int(labels['gender'].astype(str).str.upper().str.startswith('F').sum())},
        {'metric': 'male', 'value': int(labels['gender'].astype(str).str.upper().str.startswith('M').sum())},
    ])
    return summary, label_summary
