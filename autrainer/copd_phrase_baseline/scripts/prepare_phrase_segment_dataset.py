from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / 'data' / 'COPD_PhraseSegments'
SHARED_AUDIO = OUT_BASE / 'shared_audio'
SHARED_DEFAULT = SHARED_AUDIO / 'default'


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError("command failed:\n" + " ".join(cmd) + "\n\n" + proc.stderr)


def _ffmpeg_cut(in_wav: Path, out_wav: Path, start_s: float, end_s: float) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if end_s <= start_s:
        return
    _run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{start_s:.6f}', '-to', f'{end_s:.6f}',
        '-i', str(in_wav), '-ar', '16000', '-ac', '1', '-y', str(out_wav),
    ])


def _speaker_audio_path(audio_root: Path, file_id: str) -> Path:
    candidates = [str(file_id), str(file_id).zfill(5)]
    if str(file_id).isdigit():
        candidates.append(str(int(file_id)))
    for key in candidates:
        path = audio_root / key / 'Text' / 'text.wav'
        if path.exists():
            return path
    raise FileNotFoundError(f'missing text audio for {file_id}')


def _assign_folds(labels: pd.DataFrame, seed: int = 42) -> dict[str, int]:
    labels = labels.copy()
    labels['file_id'] = labels['file_id'].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_map: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(skf.split(labels['file_id'], labels['copd']), start=1):
        for idx in test_idx:
            fold_map[labels.iloc[idx]['file_id']] = fold
    return fold_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrases_csv', required=True)
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--audio_root', required=True)
    args = ap.parse_args()

    phrases = pd.read_csv(args.phrases_csv)
    labels = pd.read_csv(args.labels_csv)
    audio_root = Path(args.audio_root)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    SHARED_DEFAULT.mkdir(parents=True, exist_ok=True)

    fold_map = _assign_folds(labels)
    label_map = {str(row.file_id): int(row.copd) for row in labels.itertuples(index=False)}
    rows = []
    for row in phrases.itertuples(index=False):
        file_id = str(row.file_id)
        out_rel = Path(file_id) / f'phrase_{int(row.phrase_index):03d}.wav'
        out_wav = SHARED_DEFAULT / out_rel
        if not out_wav.exists():
            in_wav = _speaker_audio_path(audio_root, file_id)
            _ffmpeg_cut(in_wav, out_wav, float(row.start_time), float(row.end_time))
        if file_id in label_map:
            rows.append({
                'path': out_rel.as_posix(),
                'speaker_id': file_id,
                'label': label_map[file_id],
                'fold': fold_map[file_id],
            })

    meta = pd.DataFrame(rows)
    SHARED_AUDIO.mkdir(parents=True, exist_ok=True)
    meta.to_csv(SHARED_AUDIO / 'segments.csv', index=False)
    pd.DataFrame({'speaker_id': list(fold_map.keys()), 'fold': list(fold_map.values())}).to_csv(
        SHARED_AUDIO / 'speaker_folds.csv', index=False
    )

    for test_fold in range(1, 6):
        dev_fold = (test_fold % 5) + 1
        fold_dir = OUT_BASE / f'fold_{test_fold:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        link_dir = fold_dir / 'default'
        if link_dir.exists() or link_dir.is_symlink():
            if link_dir.is_symlink() or link_dir.is_file():
                link_dir.unlink()
            else:
                shutil.rmtree(link_dir)
        try:
            rel_target = os.path.relpath(SHARED_DEFAULT, start=fold_dir)
            link_dir.symlink_to(rel_target, target_is_directory=True)
        except Exception:
            shutil.copytree(SHARED_DEFAULT, link_dir)
        is_test = meta['fold'] == test_fold
        is_dev = meta['fold'] == dev_fold
        is_train = ~(is_test | is_dev)
        meta.loc[is_train, ['path', 'label']].to_csv(fold_dir / 'train.csv', index=False)
        meta.loc[is_dev, ['path', 'label']].to_csv(fold_dir / 'dev.csv', index=False)
        meta.loc[is_test, ['path', 'label']].to_csv(fold_dir / 'test.csv', index=False)


if __name__ == '__main__':
    main()
