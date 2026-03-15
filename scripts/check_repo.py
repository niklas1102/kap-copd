#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / 'data/metadata/speaker_labels.csv',
        root / 'data/interim/phrases.csv',
        root / 'data/interim/vowels.csv',
        root / 'data/interim/sustained_vowel_inventory.csv',
        root / 'data/processed/phrase_features_with_labels.csv',
        root / 'data/processed/vowel_features_with_labels.csv',
        root / 'data/processed/sustained_vowel_features_with_labels.csv',
        root / 'results/classical_ml/run_20260118_185627/phrase_summary.csv',
        root / 'results/classical_ml/run_20260118_185627/vowel_summary.csv',
        root / 'results/classical_ml/sustained_vowels_with_age_gender/summary.csv',
        root / 'results/classical_ml/sustained_vowels_without_age_gender/summary.csv',
        root / 'autrainer/copd_phrase_baseline/data/COPD_PhraseSegments/shared_audio/segments.csv',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit('missing files:\n' + '\n'.join(missing))

    for fold in range(1, 6):
        default_dir = root / 'autrainer/copd_phrase_baseline/data/COPD_PhraseSegments' / f'fold_{fold:02d}' / 'default'
        if not default_dir.exists():
            raise SystemExit(f'missing Autrainer fold audio directory: {default_dir}')

    labels = pd.read_csv(root / 'data/metadata/speaker_labels.csv', dtype={'file_id': str})
    phrases = pd.read_csv(root / 'data/interim/phrases.csv', dtype={'file_id': str})
    sustained = pd.read_csv(root / 'data/processed/sustained_vowel_features_with_labels.csv', dtype={'file_id': str})
    labels['file_id'] = labels['file_id'].str.zfill(5)
    phrases['file_id'] = phrases['file_id'].str.zfill(5)
    sustained['file_id'] = sustained['file_id'].str.zfill(5)

    if labels['file_id'].nunique() != 78:
        raise SystemExit('expected 78 labeled speakers')

    counts = phrases.groupby('file_id').size()
    if counts.nunique() != 1:
        raise SystemExit('phrase counts per speaker are not uniform')
    if int(counts.iloc[0]) != 20:
        raise SystemExit('expected 20 phrases per speaker')

    if sustained['file_id'].nunique() != 74:
        raise SystemExit('expected 74 labeled speakers in sustained vowels')

    non_features = {'file_id', 'file_name', 'token_index', 'vowel_label', 'duration_sec', 'source_path', 'age', 'gender', 'copd'}
    feature_cols = [c for c in sustained.columns if c not in non_features]
    bad_rows = sustained[feature_cols].isna().all(axis=1)
    if bad_rows.any():
        raise SystemExit('sustained-vowel feature table contains rows with all feature values missing')

    print('repo check passed')


if __name__ == '__main__':
    main()
