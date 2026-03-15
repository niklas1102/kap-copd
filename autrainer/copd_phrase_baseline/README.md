# Autrainer phrase baseline

This folder contains the prepared phrase-segment baseline.

## Included

- `data/COPD_PhraseSegments/` prepared phrase audio and fold CSVs
- `conf/` Autrainer config
- `scripts/prepare_phrase_segment_dataset.py` rebuilds the prepared phrase dataset from raw speaker folders
- `scripts/aggregate_predictions_to_speakers.py` aggregates segment predictions to speaker level

## Raw audio layout expected by the preparation script

```text
<raw_audio_root>/00082/Text/text.wav
<raw_audio_root>/00082/Vowels/1ae.wav
<raw_audio_root>/00082/Cough/cough.wav
```

## Rebuild the prepared phrase dataset

From `autrainer/copd_phrase_baseline/` run:

```bash
python scripts/prepare_phrase_segment_dataset.py \
  --phrases_csv ../../data/interim/phrases.csv \
  --labels_csv ../../data/metadata/speaker_labels.csv \
  --audio_root /path/to/raw_audio_root
```

This writes the shared phrase audio under `data/COPD_PhraseSegments/shared_audio/default/` and creates five fold directories with train/dev/test CSV files. The fold directories are linked to the shared audio to avoid duplicating the waveform files. If a zip extractor drops directory symlinks, rerun the preparation script or copy `shared_audio/default/` into each fold directory as `default/`.

## Train

```bash
autrainer train
```

By default, `conf/config.yaml` points to `fold_01`.
