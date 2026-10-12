"""
Rossmann Sales ML Learning Lab - Machine Learning Models & Evaluation
Implements Scikit-Learn based Linear Regression and K-Nearest Neighbors (KNN)
with cross-validation, hyperparameter sweep, residual analysis, and live inference:
1. Linear Regression (Ordinary Least Squares baseline)
2. K-Nearest Neighbors Regressor (Instance-based nonlinear comparison)
3. K-Fold Cross-Validation & Overfitting Diagnostic
4. K-Sweep Bias-Variance Analysis
5. Real-Time Retail Sales Inference Engine
"""

import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from backend.preprocessing import RossmannPreprocessor, prepare_train_test_split

logger = logging.getLogger("rossmann.models")

# In-memory storage for active trained models
_ACTIVE_MODELS: Dict[str, Any] = {}


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute core regression evaluation metrics:
    RMSE (Root Mean Squared Error), MAE (Mean Absolute Error),
    and R² (Coefficient of Determination).
    """
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    # Prevent negative sales predictions in evaluation
    y_pred_clipped = np.maximum(y_pred_arr, 0.0)

    mse = float(mean_squared_error(y_true_arr, y_pred_clipped))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_arr, y_pred_clipped))
    r2 = float(r2_score(y_true_arr, y_pred_clipped))

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mse": round(mse, 2),
    }


def analyze_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze prediction residuals (actual - predicted) for diagnostic plots
    and homoscedasticity checking.
    """
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.maximum(np.array(y_pred, dtype=float), 0.0)
    residuals = y_true_arr - y_pred_arr

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    skewness = float(pd.Series(residuals).skew()) if len(residuals) > 2 else 0.0

    # Create a 10-bin histogram of residuals
    counts, bin_edges = np.histogram(residuals, bins=10)
    hist_data = [
        {
            "bin": f"{round(bin_edges[i], 0):,} to {round(bin_edges[i+1], 0):,}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    return {
        "mean_residual": round(mean_res, 2),
        "std_residual": round(std_res, 2),
        "skewness": round(skewness, 3),
        "histogram": hist_data,
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    features: List[str],
    split_percent: float = 80.0,
    k: int = 9,
    metric: str = "euclidean",
    weighted: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the full end-to-end model training workflow:
    1. Splits dataset into train and test folds.
    2. Fits preprocessor exclusively on train fold (leakage-free).
    3. Trains Linear Regression and K-Nearest Neighbors.
    4. Evaluates both models on train and test sets.
    5. Computes K-fold cross validation.
    6. Extracts coefficients and prediction samples.
    """
    global _ACTIVE_MODELS
    start_time = time.time()
    test_size = 1.0 - (split_percent / 100.0)

    # 1. Leakage-free train/test partition and transformation
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
        df=df,
        features=features,
        target_col="Sales",
        test_size=test_size,
        random_state=random_state,
    )

    feature_names = list(X_train.columns)

    # 2. Fit Linear Regression
    logger.info("Fitting Scikit-Learn Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    train_pred_lin = linear_model.predict(X_train)
    test_pred_lin = linear_model.predict(X_test)

    lin_train_metrics = compute_regression_metrics(y_train, train_pred_lin)
    lin_test_metrics = compute_regression_metrics(y_test, test_pred_lin)
    lin_overfit_gap = round(
        ((lin_test_metrics["rmse"] - lin_train_metrics["rmse"]) / max(1.0, lin_train_metrics["rmse"])) * 100.0, 2
    )
    lin_train_metrics["overfitting_gap"] = lin_overfit_gap
    lin_test_metrics["overfitting_gap"] = lin_overfit_gap

    # 3. Fit K-Nearest Neighbors Regressor
    logger.info(f"Fitting Scikit-Learn KNN Regressor (k={k}, metric={metric}, weighted={weighted})...")
    knn_weights = "distance" if weighted else "uniform"
    knn_model = KNeighborsRegressor(
        n_neighbors=k,
        weights=knn_weights,
        metric=metric,
        n_jobs=-1,
    )
    knn_model.fit(X_train, y_train)

    train_pred_knn = knn_model.predict(X_train)
    test_pred_knn = knn_model.predict(X_test)

    knn_train_metrics = compute_regression_metrics(y_train, train_pred_knn)
    knn_test_metrics = compute_regression_metrics(y_test, test_pred_knn)
    knn_overfit_gap = round(
        ((knn_test_metrics["rmse"] - knn_train_metrics["rmse"]) / max(1.0, knn_train_metrics["rmse"])) * 100.0, 2
    )
    knn_train_metrics["overfitting_gap"] = knn_overfit_gap
    knn_test_metrics["overfitting_gap"] = knn_overfit_gap

    # 4. Extract Feature Importances (Coefficients)
    coefficients: List[Dict[str, Any]] = []
    for col, coef in zip(feature_names, linear_model.coef_):
        coefficients.append({
            "feature": col,
            "coefficient": round(float(coef), 2),
            "abs_importance": round(abs(float(coef)), 2),
        })
    coefficients.sort(key=lambda x: x["abs_importance"], reverse=True)

    # 5. K-Fold Cross Validation on Training Partition
    logger.info(f"Executing {cv_folds}-Fold Cross Validation...")
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results: List[Dict[str, Any]] = []
    
    cv_lin_rmse_list: List[float] = []
    cv_knn_rmse_list: List[float] = []

    fold_idx = 1
    # Sample subset for fast CV if train size is large
    cv_limit = min(15000, len(X_train))
    X_train_cv = X_train.iloc[:cv_limit]
    y_train_cv = y_train.iloc[:cv_limit]

    for train_f_idx, val_f_idx in cv.split(X_train_cv):
        X_tr_f, X_val_f = X_train_cv.iloc[train_f_idx], X_train_cv.iloc[val_f_idx]
        y_tr_f, y_val_f = y_train_cv.iloc[train_f_idx], y_train_cv.iloc[val_f_idx]

        # Fold Linear
        lin_f = LinearRegression().fit(X_tr_f, y_tr_f)
        val_lin_pred = lin_f.predict(X_val_f)
        lin_f_metrics = compute_regression_metrics(y_val_f, val_lin_pred)

        # Fold KNN
        knn_f = KNeighborsRegressor(n_neighbors=k, weights=knn_weights, metric=metric, n_jobs=-1).fit(X_tr_f, y_tr_f)
        val_knn_pred = knn_f.predict(X_val_f)
        knn_f_metrics = compute_regression_metrics(y_val_f, val_knn_pred)

        cv_results.append({
            "fold": fold_idx,
            "linear_rmse": lin_f_metrics["rmse"],
            "knn_rmse": knn_f_metrics["rmse"],
            "linear_mae": lin_f_metrics["mae"],
            "knn_mae": knn_f_metrics["mae"],
            "linear_r2": lin_f_metrics["r2"],
            "knn_r2": knn_f_metrics["r2"],
        })
        cv_lin_rmse_list.append(lin_f_metrics["rmse"])
        cv_knn_rmse_list.append(knn_f_metrics["rmse"])
        fold_idx += 1

    cv_mean = {
        "mean_linear_rmse": round(float(np.mean(cv_lin_rmse_list)), 2),
        "std_linear_rmse": round(float(np.std(cv_lin_rmse_list)), 2),
        "mean_knn_rmse": round(float(np.mean(cv_knn_rmse_list)), 2),
        "std_knn_rmse": round(float(np.std(cv_knn_rmse_list)), 2),
    }

    # 6. Prediction Comparison Samples (First 15 test rows)
    prediction_samples: List[Dict[str, Any]] = []
    sample_indices = list(range(min(20, len(y_test))))
    y_test_list = list(y_test.iloc[sample_indices])
    lin_pred_list = list(test_pred_lin[sample_indices])
    knn_pred_list = list(test_pred_knn[sample_indices])

    for i in range(len(sample_indices)):
        act = float(y_test_list[i])
        l_p = max(0.0, float(lin_pred_list[i]))
        k_p = max(0.0, float(knn_pred_list[i]))
        prediction_samples.append({
            "index": i + 1,
            "actual": round(act, 2),
            "linear_pred": round(l_p, 2),
            "knn_pred": round(k_p, 2),
            "linear_error": round(abs(act - l_p), 2),
            "knn_error": round(abs(act - k_p), 2),
        })

    # 7. Residual Diagnostics
    linear_residuals = analyze_residuals(y_test, test_pred_lin)
    knn_residuals = analyze_residuals(y_test, test_pred_knn)

    elapsed = round(time.time() - start_time, 2)

    # Save active models in memory for live predictions
    _ACTIVE_MODELS = {
        "linear_model": linear_model,
        "knn_model": knn_model,
        "preprocessor": preprocessor,
        "features": features,
        "feature_names": feature_names,
    }

    return {
        "status": "success",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "selected_features": features,
        "linear_metrics": {
            "train": lin_train_metrics,
            "test": lin_test_metrics,
        },
        "knn_metrics": {
            "train": knn_train_metrics,
            "test": knn_test_metrics,
        },
        "coefficients": coefficients,
        "intercept": round(float(linear_model.intercept_), 2),
        "cv_results": cv_results,
        "cv_mean": cv_mean,
        "prediction_samples": prediction_samples,
        "linear_residuals": linear_residuals,
        "knn_residuals": knn_residuals,
        "training_time_seconds": elapsed,
    }


def execute_knn_sweep(
    df: pd.DataFrame,
    features: List[str],
    k_values: Optional[List[int]] = None,
    metric: str = "euclidean",
    weighted: bool = True,
    sample_size: int = 10000,
) -> Dict[str, Any]:
    """
    Evaluates KNN performance across varying values of K to generate
    the bias-variance tradeoff curve and identify the optimal neighbor count.
    """
    if k_values is None:
        k_values = [3, 5, 7, 9, 11, 15, 21]

    # Sample dataset for fast interactive sweep
    sample_df = df.iloc[:sample_size] if len(df) > sample_size else df

    X_train, X_test, y_train, y_test, _ = prepare_train_test_split(
        df=sample_df,
        features=features,
        target_col="Sales",
        test_size=0.20,
    )

    weights_mode = "distance" if weighted else "uniform"
    sweep_results: List[Dict[str, Any]] = []

    best_k = k_values[0]
    best_rmse = float("inf")

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k, weights=weights_mode, metric=metric, n_jobs=-1)
        knn.fit(X_train, y_train)

        tr_pred = knn.predict(X_train)
        te_pred = knn.predict(X_test)

        tr_metrics = compute_regression_metrics(y_train, tr_pred)
        te_metrics = compute_regression_metrics(y_test, te_pred)

        if te_metrics["rmse"] < best_rmse:
            best_rmse = te_metrics["rmse"]
            best_k = k

        sweep_results.append({
            "k": k,
            "train_rmse": tr_metrics["rmse"],
            "test_rmse": te_metrics["rmse"],
            "test_mae": te_metrics["mae"],
            "test_r2": te_metrics["r2"],
        })

    return {
        "sweep_results": sweep_results,
        "best_k": best_k,
        "best_rmse": best_rmse,
    }


def predict_single_store(features_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate live sales predictions from both trained models given arbitrary store features.
    """
    global _ACTIVE_MODELS
    if not _ACTIVE_MODELS or "linear_model" not in _ACTIVE_MODELS:
        raise RuntimeError("No trained model currently available in memory. Please train models first.")

    preprocessor: RossmannPreprocessor = _ACTIVE_MODELS["preprocessor"]
    linear_model: LinearRegression = _ACTIVE_MODELS["linear_model"]
    knn_model: KNeighborsRegressor = _ACTIVE_MODELS["knn_model"]
    required_features: List[str] = _ACTIVE_MODELS["features"]

    # Construct single-row DataFrame
    row_data = {}
    for feat in required_features:
        row_data[feat] = features_input.get(feat, None)

    input_df = pd.DataFrame([row_data])

    # Transform using fitted preprocessor
    transformed_input = preprocessor.transform(input_df)

    # Predict
    linear_pred = max(0.0, float(linear_model.predict(transformed_input)[0]))
    knn_pred = max(0.0, float(knn_model.predict(transformed_input)[0]))

    return {
        "linear_prediction": round(linear_pred, 2),
        "knn_prediction": round(knn_pred, 2),
        "input_received": features_input,
    }
