"""Aggregate phrase-level predictions to speaker-level.

This script reads Autrainer predictions for a fold, maps them back to the phrase segments in the test split, averages probabilities per speaker, and reports speaker-level metrics. It accepts a predictions CSV explicitly or searches for one under the supplied results directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


def _find_predictions_file(results_dir: Path) -> Path:
    """Try to find a predictions csv inside an autrainer run folder."""
    # Most common pattern from `autrainer infer`:
    #   <id>_predictions.csv
    # Training may store something like predictions_test.csv.
    candidates = list(results_dir.rglob("*pred*csv"))
    if not candidates:
        raise FileNotFoundError(f"No predictions CSV found under: {results_dir}")
    # pick the most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--predictions_csv", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    fold_dir = root / "data" / "COPD_PhraseSegments" / f"fold_{args.fold:02d}"
    shared_meta = root / "data" / "COPD_PhraseSegments" / "shared_audio" / "segments.csv"

    test_csv = fold_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing fold test split: {test_csv}")
    if not shared_meta.exists():
        raise FileNotFoundError(f"Missing shared metadata: {shared_meta}")

    test = pd.read_csv(test_csv)
    meta = pd.read_csv(shared_meta)

    # which segments are in test set?
    test = test.merge(meta[["path", "speaker_id"]], on="path", how="left")
    if test["speaker_id"].isna().any():
        missing = test[test["speaker_id"].isna()]["path"].head(5).tolist()
        raise RuntimeError(f"Some test paths missing speaker mapping. Example: {missing}")

    results_dir = Path(args.results_dir)

    pred_path = Path(args.predictions_csv) if args.predictions_csv else _find_predictions_file(results_dir)
    preds = pd.read_csv(pred_path)

    # Autrainer inference output usually has an index column that matches file names.
    # We need to map that back to our `path` in test.csv.
    # Common columns:
    # - 'index' or 'path'
    # - 'prediction' (class) and/or 'probability'/'prob_1' etc.
    cols = set(preds.columns)

    # Identify path key
    if "path" in cols:
        key_col = "path"
    elif "index" in cols:
        key_col = "index"
    else:
        raise RuntimeError(f"Don't know how to map predictions to samples. Columns: {list(preds.columns)}")

    # Identify probability for class 1
    prob_col: Optional[str] = None
    for c in ["prob_1", "p1", "probability", "score", "logit_1"]:
        if c in cols:
            prob_col = c
            break

    if prob_col is None:
        # sometimes autrainer writes one column per class like 'probability_0', 'probability_1'
        for c in preds.columns:
            if "1" in c and "prob" in c.lower():
                prob_col = c
                break

    if prob_col is None:
        raise RuntimeError(
            "Couldn't find a class-1 probability column in predictions. "
            f"Columns: {list(preds.columns)}"
        )

    # Normalise the key so it matches our 'path' values
    # If key is just the file name, strip folders.
    preds[key_col] = preds[key_col].astype(str)

    # Most robust: match by suffix (our path is like '<speaker>/phrase_001.wav')
    preds["_path_suffix"] = preds[key_col].str.replace("\\\\", "/").str.split("/").apply(lambda x: "/".join(x[-2:]) if len(x) >= 2 else x[-1])
    test["_path_suffix"] = test["path"].str.replace("\\\\", "/")

    merged = test.merge(preds[["_path_suffix", prob_col]], on="_path_suffix", how="left")
    if merged[prob_col].isna().any():
        missing = merged[merged[prob_col].isna()]["path"].head(10).tolist()
        raise RuntimeError(
            "Some test segments have no prediction. "
            "This usually means the prediction CSV uses a different key. "
            f"Missing examples: {missing}"
        )

    # phrase-level
    y_true_phrase = merged["label"].astype(int).values
    y_prob_phrase = merged[prob_col].astype(float).values
    y_pred_phrase = (y_prob_phrase >= 0.5).astype(int)

    phrase_acc = accuracy_score(y_true_phrase, y_pred_phrase)
    phrase_f1 = f1_score(y_true_phrase, y_pred_phrase)
    phrase_uar = recall_score(y_true_phrase, y_pred_phrase, average="macro")
    try:
        phrase_auc = roc_auc_score(y_true_phrase, y_prob_phrase)
    except Exception:
        phrase_auc = float("nan")

    # speaker-level
    spk = merged.groupby(["speaker_id"], as_index=False).agg(
        y_true=("label", "first"),
        y_prob=(prob_col, "mean"),
    )
    spk["y_pred"] = (spk["y_prob"] >= 0.5).astype(int)

    spk_acc = accuracy_score(spk["y_true"], spk["y_pred"])
    spk_f1 = f1_score(spk["y_true"], spk["y_pred"])
    spk_uar = recall_score(spk["y_true"], spk["y_pred"], average="macro")
    try:
        spk_auc = roc_auc_score(spk["y_true"], spk["y_prob"])
    except Exception:
        spk_auc = float("nan")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spk.to_csv(out_path, index=False)

    print("--- Phrase-level ---")
    print(f"ACC={phrase_acc:.3f}  F1={phrase_f1:.3f}  UAR={phrase_uar:.3f}  AUC={phrase_auc:.3f}")
    print("--- Speaker-level ---")
    print(f"ACC={spk_acc:.3f}  F1={spk_f1:.3f}  UAR={spk_uar:.3f}  AUC={spk_auc:.3f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
