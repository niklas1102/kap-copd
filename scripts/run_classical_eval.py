
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kap_copd.evaluation import evaluate_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--without_age_gender', action='store_true')
    ap.add_argument('--models', default='all')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_splits', type=int, default=5)
    args = ap.parse_args()
    model_names = None if args.models == 'all' else [m.strip() for m in args.models.split(',') if m.strip()]
    fold_df, summary_df = evaluate_csv(args.csv, include_age_gender=not args.without_age_gender, seed=args.seed, n_splits=args.n_splits, model_names=model_names)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / 'folds.csv', index=False)
    summary_df.to_csv(out_dir / 'summary.csv', index=False)

if __name__ == '__main__':
    main()
