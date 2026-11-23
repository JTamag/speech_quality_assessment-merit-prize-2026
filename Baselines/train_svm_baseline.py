#!/usr/bin/env python3
"""RBF-SVM baseline classifier for 5-class MOS bins. Tune on dev, evaluate test once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC
from scipy.stats import spearmanr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import ID_COLUMNS, fit_preprocessor, load_split, transform


def metrics(y_true, y_pred, mos_true=None) -> dict:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5]).tolist(),
        "report": classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5], digits=3, output_dict=True, zero_division=0),
    }
    if mos_true is not None:
        # Class labels are 1..5; using them as a coarse MOS estimate.
        pred_mos = np.asarray(y_pred, dtype=float)
        err = pred_mos - np.asarray(mos_true, dtype=float)
        out["utt_mae_from_class"] = float(np.mean(np.abs(err)))
        out["utt_rmse_from_class"] = float(np.sqrt(np.mean(err ** 2)))
        srcc = spearmanr(mos_true, pred_mos)
        out["utt_srcc_from_class"] = None if np.isnan(srcc.correlation) else float(srcc.correlation)
    return out


def system_srcc(df: pd.DataFrame, y_pred) -> float | None:
    tmp = df.copy()
    tmp["pred"] = y_pred
    g = tmp.groupby("system_id").agg(true=("mos_native", "mean"), pred=("pred", "mean"))
    if len(g) < 2:
        return None
    val = spearmanr(g["true"], g["pred"]).correlation
    return None if np.isnan(val) else float(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="instructor")
    parser.add_argument("--C-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir / "train.csv")
    dev = load_split(data_dir / "dev.csv")
    test = load_split(data_dir / "test.csv")
    pipe, cols = fit_preprocessor(train)
    x_tr = transform(pipe, train, cols)
    x_dv = transform(pipe, dev, cols)
    x_te = transform(pipe, test, cols)
    y_tr = train["class_native_1_5"].astype(int).to_numpy()
    y_dv = dev["class_native_1_5"].astype(int).to_numpy()
    y_te = test["class_native_1_5"].astype(int).to_numpy()

    trials = []
    best = None
    for C in args.C_grid:
        clf = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced", random_state=13)
        clf.fit(x_tr, y_tr)
        pred_dv = clf.predict(x_dv)
        met = metrics(y_dv, pred_dv, mos_true=dev["mos_native"].to_numpy())
        met["C"] = C
        met["dev_sys_srcc_from_class"] = system_srcc(dev, pred_dv)
        trials.append(met)
        score = (met["quadratic_weighted_kappa"], met["macro_f1"])
        if best is None or score > best["score"]:
            best = {"C": C, "clf": clf, "score": score, "dev": met}

    pred_te = best["clf"].predict(x_te)
    test_met = metrics(y_te, pred_te, mos_true=test["mos_native"].to_numpy())
    test_met["sys_srcc_from_class"] = system_srcc(test, pred_te)
    majority = int(pd.Series(y_tr).mode().iloc[0])
    maj_pred = np.full_like(y_te, majority)
    majority_met = metrics(y_te, maj_pred, mos_true=test["mos_native"].to_numpy())

    joblib.dump(
        {"pipeline": pipe, "feature_columns": cols, "classifier": best["clf"], "C": best["C"]},
        out_dir / "svm_baseline.joblib",
    )
    pred_df = test[["utterance_id", "system_id", "mos_native", "class_native_1_5"]].copy()
    pred_df["class_pred"] = pred_te
    pred_df.to_csv(out_dir / "svm_test_predictions.csv", index=False)
    payload = {
        "selection_criterion": "dev quadratic weighted kappa, then macro-F1",
        "selected_C": best["C"],
        "n_features": len(cols),
        "n_train": int(len(train)),
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "forbidden_input_columns": ID_COLUMNS,
        "dev_trials": [{k: v for k, v in t.items() if k != "report"} for t in trials],
        "dev_selected": {k: v for k, v in best["dev"].items() if k != "report"},
        "test": {k: v for k, v in test_met.items() if k != "report"},
        "test_classification_report": test_met["report"],
        "test_majority_class_baseline": {k: v for k, v in majority_met.items() if k != "report"},
        "note": "Hyperparameters were chosen on dev only. Test was evaluated once.",
    }
    (out_dir / "baseline_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_C": best["C"], "test_qwk": test_met["quadratic_weighted_kappa"], "test_macro_f1": test_met["macro_f1"], "test_acc": test_met["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
