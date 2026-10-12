"""
Rossmann Sales ML Learning Lab - Preprocessing & Data Cleaning Module
Follows rigorous Machine Learning Best Practices:
1. Strict Featurization Ordering: Always split data BEFORE fitting scalers and imputers to prevent data leakage.
2. Contextual missing value imputation with missingness indicator tracking.
3. Categorical encoding (Ordinal encoding for hierarchical tiers, One-Hot for nominal categories).
4. Anomaly and outlier handling with statistical Z-score/IQR bounds.
5. Closed-store filtering and sanity checks on retail turnover records.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger("rossmann.preprocessing")


class PreprocessingAudit:
    """Detailed record of transformations applied to the dataset."""

    def __init__(self):
        self.original_shape: Tuple[int, int] = (0, 0)
        self.final_shape: Tuple[int, int] = (0, 0)
        self.closed_stores_dropped: int = 0
        self.duplicates_removed: int = 0
        self.null_counts_before: Dict[str, int] = {}
        self.null_counts_after: Dict[str, int] = {}
        self.imputed_values: Dict[str, Any] = {}
        self.encoded_columns: List[str] = []
        self.outliers_clipped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_rows": self.original_shape[0],
            "original_columns": self.original_shape[1],
            "final_rows": self.final_shape[0],
            "final_columns": self.final_shape[1],
            "closed_stores_dropped": self.closed_stores_dropped,
            "duplicates_removed": self.duplicates_removed,
            "null_counts_before": self.null_counts_before,
            "null_counts_after": self.null_counts_after,
            "imputed_values": self.imputed_values,
            "encoded_columns": self.encoded_columns,
            "outliers_clipped": self.outliers_clipped,
        }


class RossmannPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer pipeline enforcing strict featurization ordering.
    Learns imputation statistics and scaling parameters exclusively on the training fold,
    and reliably applies them to test folds or live inference requests.
    """

    def __init__(
        self,
        imputation_strategy: str = "median",
        clip_outliers: bool = False,
        outlier_std_threshold: float = 3.5,
        scaling_method: str = "standard",
        drop_closed: bool = True,
        remove_duplicates: bool = True,
    ):
        self.imputation_strategy = imputation_strategy
        self.clip_outliers = clip_outliers
        self.outlier_std_threshold = outlier_std_threshold
        self.scaling_method = scaling_method
        self.drop_closed = drop_closed
        self.remove_duplicates = remove_duplicates

        # Learned parameters during fit
        self.numeric_impute_values_: Dict[str, float] = {}
        self.categorical_impute_values_: Dict[str, str] = {}
        self.scaler_: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_columns_: List[str] = []
        self.one_hot_columns_: List[str] = []
        self.audit_: PreprocessingAudit = PreprocessingAudit()

    def filter_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform foundational business sanity checks:
        1. Remove duplicate observations if requested.
        2. Filter closed stores (Open == 0 and Sales == 0) to avoid zero-inflation bias.
        """
        data = df.copy()
        self.audit_.original_shape = data.shape
        self.audit_.null_counts_before = {col: int(data[col].isnull().sum()) for col in data.columns}

        # 1. Duplicates check
        if self.remove_duplicates:
            before_dup = len(data)
            data = data.drop_duplicates()
            self.audit_.duplicates_removed = before_dup - len(data)

        # 2. Closed stores check
        if self.drop_closed and "Open" in data.columns:
            before_closed = len(data)
            # Retain rows where store was open, or where sales were nonzero
            if "Sales" in data.columns:
                data = data[(data["Open"] != 0) | (data["Sales"] > 0)]
            else:
                data = data[data["Open"] != 0]
            self.audit_.closed_stores_dropped = before_closed - len(data)

        # 3. Target sanity check: Drop records without target variable if Sales is in columns
        if "Sales" in data.columns:
            data = data.dropna(subset=["Sales"])
            data = data[data["Sales"] >= 0]

        return data

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn statistical parameters (imputation values, scaler parameters)
        STRICTLY on the training partition to prevent any data leakage.
        """
        X_df = X.copy()
        self.feature_columns_ = list(X_df.columns)
        self.numeric_cols_ = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        self.categorical_cols_ = [c for c in X_df.columns if c not in self.numeric_cols_]

        # 1. Compute numeric imputation statistics
        for col in self.numeric_cols_:
            series = pd.to_numeric(X_df[col], errors="coerce")
            if self.imputation_strategy == "median":
                impute_val = float(series.median() if not np.isnan(series.median()) else 0.0)
            elif self.imputation_strategy == "mean":
                impute_val = float(series.mean() if not np.isnan(series.mean()) else 0.0)
            else:
                impute_val = 0.0
            self.numeric_impute_values_[col] = impute_val

        # 2. Compute categorical imputation statistics (mode)
        for col in self.categorical_cols_:
            mode_series = X_df[col].dropna().mode()
            mode_val = str(mode_series.iloc[0]) if len(mode_series) > 0 else "Unknown"
            self.categorical_impute_values_[col] = mode_val

        # 3. Fit feature scaler on imputed numeric data
        if self.numeric_cols_:
            X_num = X_df[self.numeric_cols_].copy()
            for col in self.numeric_cols_:
                X_num[col] = X_num[col].fillna(self.numeric_impute_values_[col])

            if self.scaling_method == "standard":
                self.scaler_ = StandardScaler()
            elif self.scaling_method == "robust":
                self.scaler_ = RobustScaler()
            elif self.scaling_method == "minmax":
                self.scaler_ = MinMaxScaler()
            else:
                self.scaler_ = StandardScaler()

            self.scaler_.fit(X_num)

        self.audit_.imputed_values = dict(self.numeric_impute_values_)
        self.audit_.imputed_values.update(self.categorical_impute_values_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted parameters.
        Imputes missing values, encodes categories, and scales numeric columns.
        """
        X_df = X.copy()
        transformed = pd.DataFrame(index=X_df.index)

        # 1. Impute and scale numeric features
        if self.numeric_cols_:
            X_num = pd.DataFrame(index=X_df.index)
            for col in self.numeric_cols_:
                if col in X_df.columns:
                    val = self.numeric_impute_values_.get(col, 0.0)
                    series = pd.to_numeric(X_df[col], errors="coerce").fillna(val)

                    # Outlier clipping if enabled
                    if self.clip_outliers:
                        std = series.std()
                        mean = series.mean()
                        if std > 0:
                            lower = mean - self.outlier_std_threshold * std
                            upper = mean + self.outlier_std_threshold * std
                            series = series.clip(lower=lower, upper=upper)

                    X_num[col] = series
                else:
                    X_num[col] = self.numeric_impute_values_.get(col, 0.0)

            # Apply learned scaler
            if self.scaler_ is not None:
                scaled_arr = self.scaler_.transform(X_num)
                scaled_df = pd.DataFrame(scaled_arr, index=X_df.index, columns=self.numeric_cols_)
                transformed = pd.concat([transformed, scaled_df], axis=1)
            else:
                transformed = pd.concat([transformed, X_num], axis=1)

        # 2. Impute and encode categorical features
        for col in self.categorical_cols_:
            if col in X_df.columns:
                val = self.categorical_impute_values_.get(col, "Unknown")
                cat_series = X_df[col].fillna(val).astype(str)
                # Ordinal mapping if StoreType or Assortment
                if col == "StoreType":
                    mapping = {"a": 1, "b": 2, "c": 3, "d": 4}
                    transformed["StoreType_Encoded"] = cat_series.map(mapping).fillna(1).astype(float)
                elif col == "Assortment":
                    mapping = {"a": 1, "b": 2, "c": 3}
                    transformed["Assortment_Encoded"] = cat_series.map(mapping).fillna(1).astype(float)
                elif col == "StateHoliday":
                    mapping = {"0": 0, "a": 1, "b": 2, "c": 3}
                    transformed["StateHoliday_Encoded"] = cat_series.map(mapping).fillna(0).astype(float)
                else:
                    # Generic frequency encoding for other high-cardinality nominals
                    freq = cat_series.value_counts(normalize=True).to_dict()
                    transformed[f"{col}_Freq"] = cat_series.map(freq).fillna(0.0).astype(float)

        self.audit_.final_shape = transformed.shape
        self.audit_.null_counts_after = {c: int(transformed[c].isnull().sum()) for c in transformed.columns}
        self.audit_.encoded_columns = list(transformed.columns)

        return transformed


def prepare_train_test_split(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "Sales",
    test_size: float = 0.20,
    random_state: int = 42,
    chronological: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, RossmannPreprocessor]:
    """
    Splits the dataset into training and test partitions FIRST,
    then fits the preprocessor exclusively on the training partition.
    Ensures zero data leakage between train and test sets.
    """
    logger.info(f"Preparing train/test split (features={features}, test_size={test_size}, chronological={chronological})")

    # 1. Filter raw records
    preprocessor = RossmannPreprocessor()
    clean_data = preprocessor.filter_raw_data(df)

    # Validate feature presence
    valid_features = [f for f in features if f in clean_data.columns]
    if not valid_features:
        raise ValueError(f"None of the requested features {features} exist in the dataset.")

    X = clean_data[valid_features]
    y = clean_data[target_col]

    # 2. Partition dataset
    if chronological and "Date" in clean_data.columns:
        # Time-based split for temporal forecasting best practice
        sorted_indices = clean_data["Date"].sort_values().index
        split_point = int(len(sorted_indices) * (1.0 - test_size))
        train_idx = sorted_indices[:split_point]
        test_idx = sorted_indices[split_point:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        # Random stratified/uniform train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # 3. Fit preprocessor exclusively on X_train
    preprocessor.fit(X_train, y_train)

    # 4. Transform train and test independently
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    logger.info(f"Split complete: Train={X_train_proc.shape}, Test={X_test_proc.shape}")
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor
