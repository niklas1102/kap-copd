#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kap_copd.sustained_vowels import build_inventory, extract_features, merge_with_labels, build_inventory_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio_root', required=True)
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--out_inventory_csv', required=True)
    ap.add_argument('--out_features_csv', required=True)
    ap.add_argument('--out_labeled_csv', required=True)
    ap.add_argument('--out_summary_csv', required=False)
    ap.add_argument('--min_dur', type=float, default=0.03)
    args = ap.parse_args()

    inventory_df = build_inventory(Path(args.audio_root))
    features_df = extract_features(inventory_df, min_dur=args.min_dur)
    labeled_df = merge_with_labels(features_df, Path(args.labels_csv))

    out_inventory = Path(args.out_inventory_csv)
    out_features = Path(args.out_features_csv)
    out_labeled = Path(args.out_labeled_csv)
    out_inventory.parent.mkdir(parents=True, exist_ok=True)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_labeled.parent.mkdir(parents=True, exist_ok=True)

    inventory_df.to_csv(out_inventory, index=False)
    features_df.to_csv(out_features, index=False)
    labeled_df.to_csv(out_labeled, index=False)

    if args.out_summary_csv:
        out_summary = Path(args.out_summary_csv)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        build_inventory_summary(inventory_df, args.labels_csv).to_csv(out_summary, index=False)


if __name__ == '__main__':
    main()
