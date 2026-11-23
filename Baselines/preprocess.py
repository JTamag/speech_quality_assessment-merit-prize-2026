#!/usr/bin/env python3
"""Train-only median imputation and standard scaling for the A224 feature CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = [
    "utterance_id",
    "relative_audio_path",
    "official_split",
    "system_id",
    "mos_native",
    "mos_0_5",
    "class_native_1_5",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS]


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fit_preprocessor(train_df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    cols = feature_columns(train_df)
    x = train_df[cols].apply(pd.to_numeric, errors="coerce")
    dropped = [c for c in cols if x[c].isna().all()]
    kept = [c for c in cols if c not in dropped]
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipe.fit(x[kept])
    return pipe, kept


def transform(pipe: Pipeline, df: pd.DataFrame, cols: list[str]):
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return pipe.transform(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="instructor/preprocess")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(data_dir / "train.csv")
    pipe, cols = fit_preprocessor(train)
    joblib.dump({"pipeline": pipe, "feature_columns": cols}, out_dir / "preprocessor.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
    print(f"fitted on train n={len(train)} n_features={len(cols)}")


if __name__ == "__main__":
    main()
