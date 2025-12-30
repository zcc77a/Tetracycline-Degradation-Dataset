#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine-learning benchmark for predicting Gibbs free energy (ΔG).

Data split:
- 70% training
- 15% validation
- 15% test

Cross-validation:
- 5-fold CV performed ONLY on the training set (70%).

Outputs (in --out directory):
- cv_summary.csv
- val_metrics.csv
- test_metrics.csv
- <model>_train.xlsx, <model>_val.xlsx, <model>_test.xlsx
- <model>_feature_importance.csv ONLY for xgboost/gbr/knn

Requirements: scikit-learn >= 1.4, xgboost >= 2.0, pandas >= 2.2
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Regressors
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

CV_FOLDS = 5
PERM_N_REPEATS = 30
IMPORTANT_MODELS = {"xgboost", "gbr", "knn"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _minmax_pipe(model) -> Pipeline:
    """Wrap a model with MinMaxScaler."""
    return Pipeline([("scaler", MinMaxScaler()), ("model", model)])


def get_models() -> Dict[str, Pipeline]:
    """Return dict {name: pipeline} for all algorithms."""
    return {
        "xgboost": _minmax_pipe(
            XGBRegressor(
                booster="gbtree",
                objective="reg:squarederror",
                n_estimators=360,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=3,
                colsample_bytree=1.0,
                subsample=0.8,
                gamma=0,
                reg_alpha=1.0,
                reg_lambda=0.1,
                base_score=0.4,
                random_state=0,
                verbosity=0,
            )
        ),
        "gbr": _minmax_pipe(
            GradientBoostingRegressor(
                n_estimators=400,
                subsample=1.0,
                max_depth=5,
                min_samples_split=3,
                min_samples_leaf=5,
                loss="squared_error",
                learning_rate=0.1,
                max_features="sqrt",
                alpha=0.9,
            )
        ),
        "knn": _minmax_pipe(
            KNeighborsRegressor(
                n_neighbors=2,
                weights="distance",
                algorithm="kd_tree",
                leaf_size=3,
            )
        ),
        "ridge": _minmax_pipe(Ridge(alpha=0.2, tol=1e-2, random_state=0)),
        "lasso": _minmax_pipe(
            Lasso(
                alpha=0.0035,
                max_iter=1000,
                tol=1e-4,
                random_state=0,
                selection="cyclic",
            )
        ),
        "linear": _minmax_pipe(LinearRegression(fit_intercept=True)),
        "dt": _minmax_pipe(
            DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=5,
                max_features=0.8,
                min_samples_leaf=5,
                min_samples_split=3,
                random_state=0,
            )
        ),
        "extratrees": _minmax_pipe(
            ExtraTreesRegressor(
                n_estimators=75,
                max_depth=3,
                criterion="squared_error",
                min_samples_split=3,
                random_state=0,
                n_jobs=-1,
            )
        ),
        "mlp": _minmax_pipe(
            MLPRegressor(
                hidden_layer_sizes=(2,),
                activation="relu",
                solver="adam",
                learning_rate_init=0.05,
                alpha=1e-3,
                batch_size=1,
                early_stopping=True,
                max_iter=200,
                random_state=0,
                verbose=False,
            )
        ),
        "svr": _minmax_pipe(
            SVR(
                C=0.5,
                epsilon=1.0,
                gamma="scale",
                coef0=0.1,
                tol=0.1,
                kernel="rbf",
            )
        ),
    }


def evaluate_cv(models: Dict[str, Pipeline], X_train: Any, y_train: Any) -> pd.DataFrame:
    """Return CV metrics for every model (performed on training set only)."""
    rows: List[dict] = []
    for name, pipe in models.items():
        scores = cross_validate(
            estimator=pipe,
            X=X_train,
            y=y_train,
            scoring=("neg_mean_absolute_error", "neg_mean_squared_error", "r2"),
            cv=CV_FOLDS,      # <-- cv=5 is here
            n_jobs=-1,
            verbose=0,
        )
        mse = -scores["test_neg_mean_squared_error"].mean()
        rows.append(
            {
                "model": name,
                "MAE_CV": -scores["test_neg_mean_absolute_error"].mean(),
                "MSE_CV": mse,
                "RMSE_CV": math.sqrt(mse),
                "R2_CV": scores["test_r2"].mean(),
                "fit_time(s)": scores["fit_time"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("RMSE_CV")


def _save_feature_importance(
    name: str, importances: np.ndarray, feature_names: List[str], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .to_csv(out_dir / f"{name}_feature_importance.csv", index=False)
    )


def _compute_feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # Only k-NN reaches here per IMPORTANT_MODELS (permutation importance)
    p_res = permutation_importance(
        estimator=pipe,
        X=X,
        y=y,
        n_repeats=PERM_N_REPEATS,
        scoring="neg_mean_squared_error",
        random_state=0,
        n_jobs=-1,
    )
    return p_res.importances_mean


def train_predict_and_importance(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 1: fit on train(70%) -> evaluate on val(15%)
    Step 2: refit on train+val(85%) -> evaluate on test(15%)
    """
    val_rows: List[dict] = []
    test_rows: List[dict] = []

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    for name, pipe in models.items():
        print(f"Training {name:>10} … ", end="", flush=True)

        # Fit on TRAIN, evaluate on VAL
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_elapsed = time.perf_counter() - t0

        y_pred_train = pipe.predict(X_train)
        y_pred_val = pipe.predict(X_val)

        pd.DataFrame({"y_true": y_train, "y_pred": y_pred_train}).to_excel(
            out_dir / f"{name}_train.xlsx", index=False
        )
        pd.DataFrame({"y_true": y_val, "y_pred": y_pred_val}).to_excel(
            out_dir / f"{name}_val.xlsx", index=False
        )

        val_mse = mean_squared_error(y_val, y_pred_val)
        val_rows.append(
            {
                "model": name,
                "MAE_val": mean_absolute_error(y_val, y_pred_val),
                "MSE_val": val_mse,
                "RMSE_val": math.sqrt(val_mse),
                "R2_val": r2_score(y_val, y_pred_val),
                "train_time(s)": train_elapsed,
            }
        )

        # Refit on TRAIN+VAL, evaluate on TEST (final)
        t1 = time.perf_counter()
        pipe.fit(X_trainval, y_trainval)
        trainval_elapsed = time.perf_counter() - t1

        y_pred_test = pipe.predict(X_test)
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}).to_excel(
            out_dir / f"{name}_test.xlsx", index=False
        )

        if name in IMPORTANT_MODELS:
            imps = _compute_feature_importance(pipe, X_test, y_test)
            _save_feature_importance(name, imps, feature_names, out_dir)

        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rows.append(
            {
                "model": name,
                "MAE_test": mean_absolute_error(y_test, y_pred_test),
                "MSE_test": test_mse,
                "RMSE_test": math.sqrt(test_mse),
                "R2_test": r2_score(y_test, y_pred_test),
                "train_time_trainval(s)": trainval_elapsed,
            }
        )

        print("done")

    val_df = pd.DataFrame(val_rows).set_index("model").sort_values("RMSE_val")
    test_df = pd.DataFrame(test_rows).set_index("model").sort_values("RMSE_test")
    return val_df, test_df


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------
def run(
    csv_path: Path,
    out_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 0,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-9):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    data = pd.read_csv(csv_path).replace(np.nan, 0)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # Split: first take out TEST (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state
    )

    # Then split TRAIN/VAL from remaining 85%
    # val_frac is 0.15 of total => 0.15/0.85 of trainval
    val_size_of_trainval = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_of_trainval, random_state=random_state
    )

    out_dir.mkdir(exist_ok=True)
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"{CV_FOLDS}-fold cross-validation is performed on TRAIN set only.")

    models = get_models()

    # 5-fold CV on TRAIN only
    cv_df = evaluate_cv(models, X_train, y_train)
    cv_df.to_csv(out_dir / "cv_summary.csv")
    print("CV summary saved to cv_summary.csv")
    print(cv_df.round(4))

    # Train/Val/Test evaluation + importance
    val_df, test_df = train_predict_and_importance(
        models=models,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=X.columns.tolist(),
        out_dir=out_dir,
    )

    val_df.to_csv(out_dir / "val_metrics.csv")
    test_df.to_csv(out_dir / "test_metrics.csv")

    print("Validation metrics saved to val_metrics.csv")
    print(val_df.round(4))
    print("Test metrics saved to test_metrics.csv (feature importances for XGBoost/GBR/kNN only).")
    print(test_df.round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Gibbs free energy with multiple ML models (feature importance for XGB/GBR/kNN)."
    )
    parser.add_argument("--csv", type=Path, default=Path("9533_TC_Gibbs_energy.csv"), help="Input CSV file")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    args = parser.parse_args()

    run(csv_path=args.csv, out_dir=args.out)
