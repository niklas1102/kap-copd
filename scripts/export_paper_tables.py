#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
from kap_copd.qc import build_dataset_summary


def _append_rows(rows: list[dict], dataset_name: str, summary_df: pd.DataFrame, include_age_gender: bool | None = None) -> None:
    for _, row in summary_df.iterrows():
        out = {
            'dataset': dataset_name,
            'model': row['model'],
            'speaker_auc_mean': row['speaker_auc_mean'],
            'speaker_acc_mean': row.get('speaker_acc_mean', float('nan')),
            'seg_auc_mean': row.get('seg_auc_mean', float('nan')),
            'seg_acc_mean': row.get('seg_acc_mean', float('nan')),
        }
        if include_age_gender is not None:
            out['include_age_gender'] = int(include_age_gender)
        rows.append(out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / 'results/classical_ml/run_20260118_185627'
    phrase_summary = pd.read_csv(run_dir / 'phrase_summary.csv')
    vowel_summary = pd.read_csv(run_dir / 'vowel_summary.csv')
    dataset_summary, label_summary = build_dataset_summary(root)

    out_dir = root / 'results/paper_tables'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    _append_rows(rows, 'phrases', phrase_summary)
    _append_rows(rows, 'passage_vowels', vowel_summary)

    sustained_rows = []
    sustained_with = root / 'results/classical_ml/sustained_vowels_with_age_gender/summary.csv'
    sustained_without = root / 'results/classical_ml/sustained_vowels_without_age_gender/summary.csv'
    if sustained_with.exists():
        sustained_with_df = pd.read_csv(sustained_with)
        _append_rows(rows, 'sustained_vowels', sustained_with_df, include_age_gender=True)
        sustained_with_df.insert(0, 'include_age_gender', 1)
        sustained_rows.append(sustained_with_df)
    if sustained_without.exists():
        sustained_without_df = pd.read_csv(sustained_without)
        _append_rows(rows, 'sustained_vowels', sustained_without_df, include_age_gender=False)
        sustained_without_df.insert(0, 'include_age_gender', 0)
        sustained_rows.append(sustained_without_df)

    pd.DataFrame(rows).to_csv(out_dir / 'main_results.csv', index=False)
    dataset_summary.to_csv(out_dir / 'dataset_overview.csv', index=False)
    label_summary.to_csv(out_dir / 'label_overview.csv', index=False)

    if sustained_rows:
        pd.concat(sustained_rows, ignore_index=True).to_csv(out_dir / 'sustained_vowel_results.csv', index=False)

    inv_summary = root / 'results/qc/sustained_vowel_inventory_summary.csv'
    if inv_summary.exists():
        pd.read_csv(inv_summary).to_csv(out_dir / 'sustained_vowel_inventory_summary.csv', index=False)


if __name__ == '__main__':
    main()
