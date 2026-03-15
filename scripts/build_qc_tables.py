#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
from kap_copd.qc import build_dataset_summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary, labels = build_dataset_summary(root)
    out_dir = root / 'results/qc'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'dataset_summary.csv', index=False)
    labels.to_csv(out_dir / 'label_summary.csv', index=False)

    inv_path = root / 'data/interim/sustained_vowel_inventory.csv'
    if inv_path.exists():
        inv = pd.read_csv(inv_path, dtype={'file_id': str})
        inv['file_id'] = inv['file_id'].str.zfill(5)
        label_df = pd.read_csv(root / 'data/metadata/speaker_labels.csv', dtype={'file_id': str})
        label_df['file_id'] = label_df['file_id'].str.zfill(5)
        missing = sorted(set(label_df['file_id']) - set(inv['file_id']))
        pd.DataFrame({'file_id': missing}).to_csv(out_dir / 'sustained_vowel_missing_labeled_speakers.csv', index=False)


if __name__ == '__main__':
    main()
