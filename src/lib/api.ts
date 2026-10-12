/**
 * Rossmann Sales ML Lab - Frontend API Client
 * Interfaces with the FastAPI backend running on port 8000 via Vite proxy.
 */

export type BackendStatus = {
  status: string;
  version: string;
  dataset_dir: string;
  datasets_ready: boolean;
};

export type DatasetFileStatus = {
  name: string;
  exists: boolean;
  size_bytes: number;
  size_mb: number;
  modified_time?: string;
};

export type DatasetStatusResponse = {
  dataset_dir: string;
  directory_exists: boolean;
  files: Record<string, DatasetFileStatus>;
  all_required_present: boolean;
};

export type FeatureItem = {
  name: string;
  category: "sales" | "store" | "weather" | "trend" | "temporal" | "geographic" | "other";
  dtype: string;
  description: string;
  is_numeric: boolean;
  missing_count: number;
  missing_percent: number;
};

export type MergedDatasetResponse = {
  status: string;
  total_loaded_rows: number;
  total_columns: number;
  columns: string[];
  sample_preview: Record<string, unknown>[];
  summary_stats: Record<string, { mean: number; median: number; std: number; min: number; max: number }>;
  missing_summary: Record<string, number>;
  tables_joined: string[];
};

export type TrainResponse = {
  status: string;
  train_size: number;
  test_size: number;
  selected_features: string[];
  linear_metrics: {
    train: { rmse: number; mae: number; r2: number; overfitting_gap: number };
    test: { rmse: number; mae: number; r2: number; overfitting_gap: number };
  };
  knn_metrics: {
    train: { rmse: number; mae: number; r2: number; overfitting_gap: number };
    test: { rmse: number; mae: number; r2: number; overfitting_gap: number };
  };
  coefficients: { feature: string; coefficient: number; abs_importance: number }[];
  intercept: number;
  cv_results: {
    fold: number;
    linear_rmse: number;
    knn_rmse: number;
    linear_mae: number;
    knn_mae: number;
    linear_r2: number;
    knn_r2: number;
  }[];
  cv_mean: {
    mean_linear_rmse: number;
    std_linear_rmse: number;
    mean_knn_rmse: number;
    std_knn_rmse: number;
  };
  prediction_samples: {
    index: number;
    actual: number;
    linear_pred: number;
    knn_pred: number;
    linear_error: number;
    knn_error: number;
  }[];
  linear_residuals?: {
    mean_residual: number;
    std_residual: number;
    skewness: number;
    histogram: { bin: string; count: number }[];
  };
  knn_residuals?: {
    mean_residual: number;
    std_residual: number;
    skewness: number;
    histogram: { bin: string; count: number }[];
  };
  training_time_seconds: number;
};

export type KNNSweepResponse = {
  sweep_results: {
    k: number;
    train_rmse: number;
    test_rmse: number;
    test_mae: number;
    test_r2: number;
  }[];
  best_k: number;
  best_rmse: number;
};

export type SinglePredictionResponse = {
  linear_prediction: number;
  knn_prediction: number;
  input_received: Record<string, unknown>;
};

const API_BASE = "/api";

export async function fetchHealth(): Promise<BackendStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetStatus(): Promise<DatasetStatusResponse> {
  const res = await fetch(`${API_BASE}/dataset/status`);
  if (!res.ok) throw new Error(`Dataset status failed: ${res.statusText}`);
  return res.json();
}

export async function loadMergedDataset(sampleSize = 25000): Promise<MergedDatasetResponse> {
  const res = await fetch(`${API_BASE}/dataset/load-merged`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sample_size: sampleSize,
      include_store: true,
      include_weather: true,
      include_trends: true,
    }),
  });
  if (!res.ok) throw new Error(`Failed to load merged dataset: ${res.statusText}`);
  return res.json();
}

export async function fetchFeaturesCatalog(): Promise<{ total_features: number; features: FeatureItem[]; default_selected: string[] }> {
  const res = await fetch(`${API_BASE}/dataset/features`);
  if (!res.ok) throw new Error(`Failed to fetch features: ${res.statusText}`);
  return res.json();
}

export async function fetchCorrelations(target = "Sales"): Promise<{ target: string; correlations: { feature: string; correlation: number; abs_correlation: number }[] }> {
  const res = await fetch(`${API_BASE}/dataset/correlations?target=${encodeURIComponent(target)}`);
  if (!res.ok) throw new Error(`Failed to fetch correlations: ${res.statusText}`);
  return res.json();
}

export async function trainModelsAPI(params: {
  features: string[];
  splitPercent: number;
  k: number;
  metric: string;
  weighted: boolean;
  sampleSize?: number;
}): Promise<TrainResponse> {
  const res = await fetch(`${API_BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features: params.features,
      split_percent: params.splitPercent,
      k: params.k,
      metric: params.metric,
      weighted_knn: params.weighted,
      sample_size: params.sampleSize ?? 25000,
      cv_folds: 5,
    }),
  });
  if (!res.ok) throw new Error(`Training failed: ${res.statusText}`);
  return res.json();
}

export async function runKNNSweepAPI(features: string[], sampleSize = 10000): Promise<KNNSweepResponse> {
  const res = await fetch(`${API_BASE}/train/knn-sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features,
      k_values: [3, 5, 7, 9, 11, 15, 21],
      metric: "euclidean",
      weighted: true,
      sample_size: sampleSize,
    }),
  });
  if (!res.ok) throw new Error(`KNN sweep failed: ${res.statusText}`);
  return res.json();
}

export async function predictLiveAPI(features: Record<string, unknown>): Promise<SinglePredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return res.json();
}
