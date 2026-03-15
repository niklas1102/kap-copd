
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class LoadedTable:
    df: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_cols: list[str]


def _norm_gender(value) -> str:
    if pd.isna(value):
        return ''
    s = str(value).strip().lower()
    if s in {'m', 'male', 'man', '1'}:
        return 'M'
    if s in {'f', 'female', 'woman', '0'}:
        return 'F'
    return str(value).strip().upper()[:1]


def load_feature_table(csv_path: str | Path, include_age_gender: bool = True) -> LoadedTable:
    df = pd.read_csv(csv_path)
    required = {'file_id', 'age', 'gender', 'copd'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'missing required columns: {sorted(missing)}')
    df = df.copy()
    df['gender'] = df['gender'].apply(_norm_gender)
    df['gender_bin'] = df['gender'].map({'F': 0, 'M': 1}).fillna(0)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(pd.to_numeric(df['age'], errors='coerce').median())
    df['copd'] = pd.to_numeric(df['copd'], errors='coerce').astype(int)
    non_features = {'file_id', 'age', 'gender', 'gender_bin', 'copd', 'start_time', 'end_time', 'phrase_index', 'vowel_index', 'vowel_label', 'token_index', 'duration_sec', 'file_name', 'source_path'}
    feature_cols = [c for c in df.columns if c not in non_features]
    if feature_cols:
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        medians = df[feature_cols].median(numeric_only=True)
        df[feature_cols] = df[feature_cols].fillna(medians).fillna(0.0)
    pieces = []
    if include_age_gender:
        pieces.append(df[['age', 'gender_bin']].to_numpy(float))
    pieces.append(df[feature_cols].to_numpy(float))
    x = np.hstack(pieces)
    y = df['copd'].to_numpy(int)
    groups = df['file_id'].astype(str).to_numpy()
    return LoadedTable(df=df, x=x, y=y, groups=groups, feature_cols=feature_cols)


def default_models(seed: int = 42) -> dict[str, object]:
    return {
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000)),
        ]),
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, gamma='scale', random_state=seed)),
        ]),
        'RandomForest': RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=seed, n_jobs=4),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=350, learning_rate=0.05, max_depth=3, random_state=seed),
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def evaluate_loaded_table(table: LoadedTable, seed: int = 42, n_splits: int = 5, model_names: Iterable[str] | None = None):
    models = default_models(seed)
    if model_names is not None:
        models = {name: models[name] for name in model_names}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    summary_rows = []
    for model_name, estimator in models.items():
        model_fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(table.x, table.y, table.groups), start=1):
            clf = clone(estimator)
            clf.fit(table.x[train_idx], table.y[train_idx])
            prob = clf.predict_proba(table.x[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            seg_acc = float(accuracy_score(table.y[test_idx], pred))
            seg_auc = _safe_auc(table.y[test_idx], prob)
            fold_df = pd.DataFrame({'file_id': table.groups[test_idx], 'y_true': table.y[test_idx], 'y_prob': prob})
            speaker_df = fold_df.groupby('file_id', as_index=False).agg(y_true=('y_true', 'first'), y_prob=('y_prob', 'mean'))
            speaker_pred = (speaker_df['y_prob'] >= 0.5).astype(int)
            row = {
                'model': model_name,
                'fold': fold,
                'seg_acc': seg_acc,
                'seg_auc': seg_auc,
                'speaker_acc': float(accuracy_score(speaker_df['y_true'], speaker_pred)),
                'speaker_auc': _safe_auc(speaker_df['y_true'].to_numpy(), speaker_df['y_prob'].to_numpy()),
                'n_segments': int(len(test_idx)),
                'n_speakers': int(speaker_df.shape[0]),
            }
            fold_rows.append(row)
            model_fold_rows.append(row)
        df = pd.DataFrame(model_fold_rows)
        summary_rows.append({
            'model': model_name,
            'seg_acc_mean': df['seg_acc'].mean(),
            'seg_acc_std': df['seg_acc'].std(ddof=1),
            'seg_auc_mean': df['seg_auc'].mean(),
            'seg_auc_std': df['seg_auc'].std(ddof=1),
            'speaker_acc_mean': df['speaker_acc'].mean(),
            'speaker_acc_std': df['speaker_acc'].std(ddof=1),
            'speaker_auc_mean': df['speaker_auc'].mean(),
            'speaker_auc_std': df['speaker_auc'].std(ddof=1),
            'n_segments_total': int(df['n_segments'].sum()),
            'n_speakers_avg': float(df['n_speakers'].mean()),
        })
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def evaluate_csv(csv_path: str | Path, include_age_gender: bool = True, seed: int = 42, n_splits: int = 5, model_names: Iterable[str] | None = None):
    return evaluate_loaded_table(load_feature_table(csv_path, include_age_gender=include_age_gender), seed=seed, n_splits=n_splits, model_names=model_names)
