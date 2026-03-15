from __future__ import annotations

from pathlib import Path
import contextlib
import re
import wave

import pandas as pd


EXPECTED_LABELS = ('ae', 'u', 'oe', 'o', 'e')


def _duration_seconds(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return frames / float(rate)


def _speaker_id(name: str) -> str:
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return digits.zfill(5) if digits else str(name)


def _token_index_from_name(name: str) -> int | None:
    m = re.match(r'^(\d+)', Path(name).stem)
    return int(m.group(1)) if m else None


def _label_from_name(name: str) -> str:
    stem = Path(name).stem.lower().replace(' ', '').replace('_', '')
    stem = re.sub(r'^[0-9]+', '', stem)
    return stem


def build_inventory(audio_root: str | Path) -> pd.DataFrame:
    audio_root = Path(audio_root)
    rows = []
    for speaker_dir in sorted(p for p in audio_root.iterdir() if p.is_dir()):
        vowel_dir = speaker_dir / 'Vowels'
        if not vowel_dir.exists():
            continue
        for wav_path in sorted(vowel_dir.glob('*.wav')):
            rows.append(
                {
                    'file_id': _speaker_id(speaker_dir.name),
                    'file_name': wav_path.name,
                    'token_index': _token_index_from_name(wav_path.name),
                    'vowel_label': _label_from_name(wav_path.name),
                    'duration_sec': round(_duration_seconds(wav_path), 6),
                    'source_path': str(wav_path),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['file_id', 'file_name', 'token_index', 'vowel_label', 'duration_sec', 'source_path'])
    df = df.sort_values(['file_id', 'token_index', 'file_name']).reset_index(drop=True)
    return df


def extract_features(
    inventory_df: pd.DataFrame,
    min_dur: float = 0.03,
) -> pd.DataFrame:
    import opensmile

    if inventory_df.empty:
        return pd.DataFrame(columns=['file_id', 'file_name', 'token_index', 'vowel_label', 'duration_sec', 'source_path'])

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    rows = []
    for row in inventory_df.itertuples(index=False):
        if float(row.duration_sec) < min_dur:
            continue
        feats = smile.process_file(str(row.source_path))
        feat_row = feats.iloc[0].to_dict()
        feat_row.update(
            {
                'file_id': row.file_id,
                'file_name': row.file_name,
                'token_index': row.token_index,
                'vowel_label': row.vowel_label,
                'duration_sec': row.duration_sec,
                'source_path': row.source_path,
            }
        )
        rows.append(feat_row)

    df = pd.DataFrame(rows)
    meta_cols = ['file_id', 'file_name', 'token_index', 'vowel_label', 'duration_sec', 'source_path']
    if df.empty:
        return pd.DataFrame(columns=meta_cols)
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + feature_cols]


def merge_with_labels(features_df: pd.DataFrame, labels_csv: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv, dtype={'file_id': str})
    labels['file_id'] = labels['file_id'].astype(str).str.zfill(5)
    features_df = features_df.copy()
    features_df['file_id'] = features_df['file_id'].astype(str).str.zfill(5)
    merged = features_df.merge(labels, on='file_id', how='inner')
    return merged


def build_inventory_summary(inventory_df: pd.DataFrame, labels_csv: str | Path | None = None) -> pd.DataFrame:
    rows = []
    if inventory_df.empty:
        return pd.DataFrame(columns=['metric', 'value'])

    rows.append({'metric': 'n_rows', 'value': int(len(inventory_df))})
    rows.append({'metric': 'n_speakers', 'value': int(inventory_df['file_id'].nunique())})
    rows.append({'metric': 'mean_duration_sec', 'value': float(inventory_df['duration_sec'].mean())})
    rows.append({'metric': 'median_duration_sec', 'value': float(inventory_df['duration_sec'].median())})

    for label in EXPECTED_LABELS:
        rows.append(
            {
                'metric': f'n_{label}',
                'value': int((inventory_df['vowel_label'] == label).sum()),
            }
        )

    observed = set(inventory_df['vowel_label'].astype(str))
    other_labels = sorted(observed - set(EXPECTED_LABELS))
    rows.append({'metric': 'other_labels', 'value': ','.join(other_labels)})

    if labels_csv is not None:
        labels = pd.read_csv(labels_csv, dtype={'file_id': str})
        labels['file_id'] = labels['file_id'].astype(str).str.zfill(5)
        inventory_ids = set(inventory_df['file_id'].astype(str).str.zfill(5))
        label_ids = set(labels['file_id'])
        rows.append({'metric': 'labeled_speakers_with_audio', 'value': int(len(label_ids & inventory_ids))})
        rows.append({'metric': 'labeled_speakers_missing_audio', 'value': int(len(label_ids - inventory_ids))})

    return pd.DataFrame(rows)
