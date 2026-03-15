from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
PHRASE_FEATURES_CSV = BASE_DIR / "phrase_features_with_labels.csv"
OUT_CSV = BASE_DIR / "speaker_features_eGeMAPS.csv"


def main():
    df = pd.read_csv(PHRASE_FEATURES_CSV, dtype={"file_id": str})
    print(f"phrase-level data: {df.shape[0]} rows x {df.shape[1]} cols")

    meta_cols = ["file_id", "phrase_index", "start_time", "end_time", "age", "gender", "copd"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Number of feature: {len(feature_cols)}")

    for c in feature_cols + ["age", "copd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_dict = {c: "mean" for c in feature_cols}
    agg_dict["age"] = "first"
    agg_dict["copd"] = "first"
    agg_dict["gender"] = "first"

    grouped = df.groupby("file_id").agg(agg_dict).reset_index()

    print(f"shape: {grouped.shape[0]} rows x {grouped.shape[1]} cols")
    print(grouped[["file_id", "age", "gender", "copd"]].head())

    # Save
    grouped.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
