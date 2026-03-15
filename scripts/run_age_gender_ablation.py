
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

from kap_copd.evaluation import evaluate_csv

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'data/processed/phrase_features_with_labels.csv'
OUT_PATH = ROOT / 'results/paper_tables/phrase_age_gender_ablation.csv'


def main() -> None:
    rows = []
    for include_age_gender, name in [(True, 'with_age_gender'), (False, 'acoustic_only')]:
        _, summary = evaluate_csv(CSV_PATH, include_age_gender=include_age_gender, model_names=['RandomForest'])
        record = summary.iloc[0].to_dict()
        record['setting'] = name
        rows.append(record)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

if __name__ == '__main__':
    main()
