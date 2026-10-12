"""
Rossmann Sales ML Learning Lab - Pydantic Request & Response Schemas
Provides strict validation, data contracts, and type safety for API requests.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    version: str
    dataset_dir: str
    datasets_ready: bool

class FileMetadata(BaseModel):
    name: str
    exists: bool
    size_bytes: int
    size_mb: float
    modified_time: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None

class DatasetStatusResponse(BaseModel):
    dataset_dir: str
    directory_exists: bool
    files: Dict[str, FileMetadata]
    all_required_present: bool

class TablePreviewResponse(BaseModel):
    table_name: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    sample: List[Dict[str, Any]]

class FeatureItem(BaseModel):
    name: str
    category: str  # "sales", "store", "weather", "trend", "temporal"
    dtype: str
    description: str
    is_numeric: bool
    missing_count: int
    missing_percent: float

class DatasetFeaturesResponse(BaseModel):
    total_features: int
    features: List[FeatureItem]
    default_selected: List[str]

class MergePipelineRequest(BaseModel):
    sample_size: int = Field(default=25000, ge=1000, le=250000)
    include_store: bool = True
    include_weather: bool = True
    include_trends: bool = True
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    random_seed: Optional[int] = 42

class MergedDatasetResponse(BaseModel):
    status: str
    total_loaded_rows: int
    total_columns: int
    columns: List[str]
    sample_preview: List[Dict[str, Any]]
    summary_stats: Dict[str, Dict[str, float]]
    missing_summary: Dict[str, int]
    tables_joined: List[str]

class PreprocessOptionsRequest(BaseModel):
    remove_duplicates: bool = True
    drop_closed: bool = True
    imputation_strategy: str = Field(default="median", pattern="^(mean|median|zero)$")
    clip_outliers: bool = False
    outlier_std_threshold: float = 3.0
    encode_categoricals: bool = True

class PreprocessAuditResponse(BaseModel):
    original_rows: int
    processed_rows: int
    rows_dropped: int
    dropped_closed_count: int
    duplicate_rows_count: int
    imputed_columns: Dict[str, int]
    encoded_features: List[str]

class ModelTrainingRequest(BaseModel):
    features: List[str] = Field(min_length=1)
    split_percent: float = Field(default=80.0, ge=50.0, le=95.0)
    k: int = Field(default=9, ge=1, le=100)
    metric: str = Field(default="euclidean", pattern="^(euclidean|manhattan|minkowski)$")
    weighted_knn: bool = True
    cv_folds: int = Field(default=5, ge=2, le=10)
    sample_size: Optional[int] = Field(default=25000, ge=1000, le=250000)
    seed: Optional[int] = 42

class CVFoldResult(BaseModel):
    fold: int
    linear_rmse: float
    knn_rmse: float
    linear_mae: float
    knn_mae: float
    linear_r2: float
    knn_r2: float

class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    r2: float
    overfitting_gap: float

class FeatureWeight(BaseModel):
    feature: str
    coefficient: float
    abs_importance: float

class PredictionSample(BaseModel):
    index: int
    actual: float
    linear_pred: float
    knn_pred: float
    linear_error: float
    knn_error: float

class ModelTrainingResponse(BaseModel):
    status: str
    train_size: int
    test_size: int
    selected_features: List[str]
    linear_metrics: Dict[str, ModelMetrics]  # "train", "test"
    knn_metrics: Dict[str, ModelMetrics]     # "train", "test"
    coefficients: List[FeatureWeight]
    intercept: float
    cv_results: List[CVFoldResult]
    cv_mean: Dict[str, float]
    prediction_samples: List[PredictionSample]
    training_time_seconds: float

class KNNSweepRequest(BaseModel):
    features: List[str] = Field(min_length=1)
    k_values: List[int] = Field(default=[3, 5, 7, 9, 11, 15, 21])
    metric: str = "euclidean"
    weighted: bool = True
    sample_size: Optional[int] = 10000

class KNNSweepPoint(BaseModel):
    k: int
    train_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float

class KNNSweepResponse(BaseModel):
    sweep_results: List[KNNSweepPoint]
    best_k: int
    best_rmse: float

class SinglePredictionRequest(BaseModel):
    features: Dict[str, Union[float, int, str]]

class SinglePredictionResponse(BaseModel):
    linear_prediction: float
    knn_prediction: float
    input_received: Dict[str, Any]
