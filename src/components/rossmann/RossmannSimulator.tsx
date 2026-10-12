import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import Papa from "papaparse";
import {
  Activity, ArrowDownToLine, BarChart3, Check, ChevronLeft, ChevronRight,
  Database, FileUp, Gauge, Moon, Play, RotateCcw, Sun, WandSparkles,
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, ComposedChart, Legend, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  correlation, countMissing, generateSyntheticData, getMetrics, normalizeUploadedRow,
  numericColumns, predictOne, preprocessData, selectableFeatures, trainAndEvaluate,
  type ModelOptions, type PredictionResult, type PreprocessOptions, type RossmannRow,
} from "@/lib/rossmann";
import {
  fetchHealth, fetchDatasetStatus, loadMergedDataset, fetchFeaturesCatalog,
  trainModelsAPI, runKNNSweepAPI, predictLiveAPI,
  type DatasetStatusResponse, type FeatureItem,
  type TrainResponse, type KNNSweepResponse,
} from "@/lib/api";

const stages = [
  {
    label: "Data Story & Ingestion",
    short: "1. Data",
    icon: Database,
    story: "Relational Integration: Connecting Sales transactions with Store metadata, German Federal States, Daily Weather, and Google Trends.",
  },
  {
    label: "Exploratory Insights",
    short: "2. Explore",
    icon: BarChart3,
    story: "Retail Drivers: Investigating how customer traffic volume, promotional campaigns, weather severity, and weekday timing impact daily revenue.",
  },
  {
    label: "Data Preparation & Audit",
    short: "3. Prepare",
    icon: Activity,
    story: "Leakage-Free Cleaning: Filtering closed store zeros, imputing competitor distances, and inspecting distribution shifts.",
  },
  {
    label: "Feature Engineering",
    short: "4. Features",
    icon: Gauge,
    story: "Signal Selection: Choosing features across store characteristics, weather swings, calendar seasonality, and regional search interest.",
  },
  {
    label: "Model Training",
    short: "5. Training",
    icon: WandSparkles,
    story: "Supervised Learning: Fitting Scikit-Learn Ordinary Least Squares Linear Regression alongside instance-based K-Nearest Neighbors (KNN).",
  },
  {
    label: "Scientific Evaluation",
    short: "6. Evaluate",
    icon: Check,
    story: "Generalization Assessment: Comparing held-out RMSE, MAE, R², 5-fold cross-validation variance, and residual distributions.",
  },
  {
    label: "Store Manager Simulator",
    short: "7. Predict",
    icon: Play,
    story: "Live Scenario Playground: Simulating store turnover under varied traffic, promotion, and seasonal conditions for inventory planning.",
  },
];

const featureLabels: Record<string, string> = {
  Customers: "Customers", Promo: "Promotion", DayOfWeek: "Day of week",
  SchoolHoliday: "School holiday", Store: "Store number", Open: "Open status",
  StoreType: "Store model (a/b/c/d)", Assortment: "Assortment level",
  CompetitionDistance: "Competitor distance (m)", MonthsSinceCompetition: "Months with competitor",
  Promo2Active: "Promo2 coupon active", Mean_TemperatureC: "Mean temperature (°C)",
  Precipitationmm: "Precipitation (mm)", Mean_Humidity: "Mean humidity (%)",
  Max_Wind_SpeedKm_h: "Wind speed (km/h)", trend: "Google Search index",
  Month: "Month of year", WeekOfYear: "Week of year", IsWeekend: "Weekend flag",
};

const money = (value: number) => `${Math.round(value).toLocaleString("en-US")} kr`;
const numberFormat = (value: number) => Math.round(value).toLocaleString("en-US");
const tooltipStyle = {
  contentStyle: {
    border: "1px solid var(--border)",
    borderRadius: 6,
    background: "var(--popover)",
    color: "var(--popover-foreground)",
  },
};

type ChartDatum = Record<string, number | string>;

function Panel({
  title,
  subtitle,
  children,
  className = "",
  badge,
  accent,
}: {
  title: string;
  subtitle?: string | undefined;
  children: ReactNode;
  className?: string;
  badge?: ReactNode | undefined;
  accent?: "primary" | "success" | "warning" | "none";
}) {
  const accentColor =
    accent === "success" ? "from-chart-2/60 to-chart-2/10" :
    accent === "warning" ? "from-chart-3/60 to-chart-3/10" :
    accent === "none" ? "" :
    "from-primary/60 to-primary/10";
  return (
    <section className={`group min-w-0 rounded-xl border border-border bg-card shadow-sm transition-shadow hover:shadow-md ${className}`}>
      {accent !== "none" && (
        <div className={`h-0.5 w-full rounded-t-xl bg-gradient-to-r ${accentColor}`} />
      )}
      <div className="p-4 sm:p-5">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-foreground">{title}</h3>
              {badge}
            </div>
            {subtitle && <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>}
          </div>
        </div>
        {children}
      </div>
    </section>
  );
}

function Metric({
  label, value, detail, color = "primary",
}: {
  label: string; value: string; detail: string;
  color?: "primary" | "success" | "warning" | "neutral";
}) {
  const borderColor =
    color === "success" ? "border-chart-2" :
    color === "warning" ? "border-chart-3" :
    color === "neutral" ? "border-muted-foreground/40" :
    "border-primary";
  const bgColor =
    color === "success" ? "bg-chart-2/5" :
    color === "warning" ? "bg-chart-3/5" :
    color === "neutral" ? "" :
    "bg-primary/5";
  const textColor =
    color === "success" ? "text-chart-2" :
    color === "warning" ? "text-chart-3" :
    color === "neutral" ? "text-muted-foreground" :
    "text-primary";
  return (
    <div className={`rounded-lg border-l-2 ${borderColor} ${bgColor} py-2 pl-3 pr-2`}>
      <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
      <p className={`mt-1 text-xl font-semibold tabular-nums ${textColor}`}>{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex min-h-56 flex-col items-center justify-center rounded-xl border border-dashed border-border bg-gradient-to-br from-muted/40 to-muted/10 px-5 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted/60 ring-4 ring-border/40">
        <Database className="h-5 w-5 text-muted-foreground" />
      </div>
      <p className="text-sm font-semibold text-foreground">{title}</p>
      <div className="mt-2 max-w-sm text-xs leading-5 text-muted-foreground">{children}</div>
    </div>
  );
}


export default function RossmannSimulator() {
  const [stage, setStage] = useState(0);
  const [rows, setRows] = useState<RossmannRow[]>([]);
  const [processedRows, setProcessedRows] = useState<RossmannRow[]>([]);
  const [dark, setDark] = useState(false);
  const [stores, setStores] = useState(50);
  const [days, setDays] = useState(365);
  const [noise, setNoise] = useState(18);
  const [missingPercent, setMissingPercent] = useState(2);
  const [outlierPercent, setOutlierPercent] = useState(1);
  const [preprocess, setPreprocess] = useState<PreprocessOptions>({
    removeDuplicates: true,
    dropClosed: true,
    imputation: "median",
    clipOutliers: false,
  });
  const [features, setFeatures] = useState<string[]>([
    "Customers", "Promo", "DayOfWeek", "SchoolHoliday", "CompetitionDistance", "Mean_TemperatureC"
  ]);
  const [splitPercent, setSplitPercent] = useState(80);
  const [k, setK] = useState(9);
  const [metric, setMetric] = useState<ModelOptions["metric"]>("euclidean");
  const [weighted, setWeighted] = useState(true);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [training, setTraining] = useState(false);
  const [fullRun, setFullRun] = useState(false);
  const [progress, setProgress] = useState(0);
  const [sourceName, setSourceName] = useState("Synthetic dataset");
  const [promoInput, setPromoInput] = useState(true);
  const [customersInput, setCustomersInput] = useState(550);
  const [dayInput, setDayInput] = useState(4);
  const [schoolInput, setSchoolInput] = useState(false);
  const [compDistanceInput, setCompDistanceInput] = useState(1200);
  const [tempInput, setTempInput] = useState(16);
  const fileInput = useRef<HTMLInputElement>(null);

  // Backend Integration States
  const [backendReady, setBackendReady] = useState(false);
  const [datasetFiles, setDatasetFiles] = useState<DatasetStatusResponse | null>(null);
  const [backendTrainResult, setBackendTrainResult] = useState<TrainResponse | null>(null);
  const [knnSweep, setKnnSweep] = useState<KNNSweepResponse | null>(null);
  const [backendFeatures, setBackendFeatures] = useState<FeatureItem[]>([]);
  const [loadingMerged, setLoadingMerged] = useState(false);
  const [sampleSizeSetting, setSampleSizeSetting] = useState(25000);
  const [livePredictionData, setLivePredictionData] = useState<{ linear: number; knn: number } | null>(null);

  // Poll backend health and dataset presence on initial load
  useEffect(() => {
    async function checkBackend() {
      try {
        const health = await fetchHealth();
        setBackendReady(health.status === "healthy");
        const statusData = await fetchDatasetStatus();
        setDatasetFiles(statusData);
        const featData = await fetchFeaturesCatalog();
        setBackendFeatures(featData.features);
      } catch (err) {
        console.warn("Backend not active or offline:", err);
        setBackendReady(false);
      }
    }
    checkBackend();
  }, []);

  // Sync dark class to <html> so Radix portals (selects, dialogs) get dark styles
  useEffect(() => {
    const root = document.documentElement;
    if (dark) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    return () => root.classList.remove("dark");
  }, [dark]);

  const prepResult = useMemo(() => preprocessData(rows, preprocess), [rows, preprocess]);
  const effectiveRows = processedRows.length ? processedRows : prepResult.rows;


  const analysis = useMemo(() => {
    if (!effectiveRows.length) return null;
    const rowsWithSales = effectiveRows.filter((row) => typeof row.Sales === "number");
    const grouped = (key: "year" | "weekday" | "quarter" | "promo") => {
      const buckets = new Map<string, { sales: number; customers: number; count: number }>();
      for (const row of rowsWithSales) {
        const date = new Date(`${row.Date}T00:00:00Z`);
        const label = key === "year"
          ? String(Number.isNaN(date.getTime()) ? "Unknown" : date.getUTCFullYear())
          : key === "weekday"
            ? `${row.DayOfWeek ?? 0}`
            : key === "quarter"
              ? `Q${Math.floor(date.getUTCMonth() / 3) + 1}`
              : row.Promo ? "Promo" : "No promo";
        const bucket = buckets.get(label) ?? { sales: 0, customers: 0, count: 0 };
        bucket.sales += row.Sales ?? 0;
        bucket.customers += row.Customers ?? 0;
        bucket.count += 1;
        buckets.set(label, bucket);
      }
      return Array.from(buckets, ([label, values]) => ({
        label,
        sales: values.sales / values.count,
        customers: values.customers / values.count,
        rows: values.count,
      }));
    };
    const yearly = grouped("year");
    const weekdays = grouped("weekday").sort((a, b) => Number(a.label) - Number(b.label));
    const quarters = grouped("quarter");
    const promo = grouped("promo");
    const scatter = rowsWithSales
      .filter((row) => typeof row.Customers === "number")
      .slice(0, 1400)
      .map((row) => ({
        customers: row.Customers ?? 0,
        sales: row.Sales ?? 0,
        year: row.Date.slice(0, 4),
      }));
    const salesValues = rowsWithSales.map((row) => row.Sales ?? 0);
    const customerValues = rowsWithSales.map((row) => row.Customers ?? 0);
    const stats = (values: number[]) => {
      const mean = values.reduce((sum, value) => sum + value, 0) / (values.length || 1);
      return {
        mean,
        sd: Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length || 1)),
      };
    };
    const histogram = (values: number[], count = 12) => {
      if (!values.length) return [];
      const max = Math.max(...values);
      const min = Math.min(...values);
      const range = max - min || 1;
      const bins = Array.from({ length: count }, (_, index) => ({
        label: numberFormat(min + (range * index) / count),
        count: 0,
      }));
      for (const value of values) {
        const bin = Math.min(count - 1, Math.floor(((value - min) / range) * count));
        bins[bin]!.count += 1;
      }
      return bins;
    };
    const promoSales = promo.find((item) => item.label === "Promo")?.sales ?? 0;
    const noPromoSales = promo.find((item) => item.label === "No promo")?.sales ?? 0;
    const promoCustomers = promo.find((item) => item.label === "Promo")?.customers ?? 0;
    const noPromoCustomers = promo.find((item) => item.label === "No promo")?.customers ?? 0;
    const uplift = noPromoSales ? (promoSales / noPromoSales - 1) * 100 : 0;
    const correlations = selectableFeatures.map((feature) => ({
      feature,
      value: correlation(
        effectiveRows.map((row) => Number(row[feature]) || 0),
        effectiveRows.map((row) => Number(row.Sales) || 0)
      ),
    })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const heatmap = numericColumns.map((x) => ({
      feature: x,
      values: numericColumns.map((y) => ({
        feature: y,
        value: correlation(
          effectiveRows.map((row) => Number(row[x]) || 0),
          effectiveRows.map((row) => Number(row[y]) || 0)
        ),
      })),
    }));
    return {
      yearly, weekdays, quarters, promo, scatter, correlations, heatmap,
      salesStats: stats(salesValues), customerStats: stats(customerValues),
      salesHist: histogram(salesValues), customerHist: histogram(customerValues),
      promoSales, noPromoSales, promoCustomers, noPromoCustomers, uplift,
    };
  }, [effectiveRows]);

  const metrics = useMemo(() => {
    if (backendTrainResult) {
      return {
        linearTrain: {
          rmse: backendTrainResult.linear_metrics.train.rmse,
          mape: backendTrainResult.linear_metrics.train.mae,
          r2: backendTrainResult.linear_metrics.train.r2,
        },
        linearTest: {
          rmse: backendTrainResult.linear_metrics.test.rmse,
          mape: backendTrainResult.linear_metrics.test.mae,
          r2: backendTrainResult.linear_metrics.test.r2,
        },
        knnTrain: {
          rmse: backendTrainResult.knn_metrics.train.rmse,
          mape: backendTrainResult.knn_metrics.train.mae,
          r2: backendTrainResult.knn_metrics.train.r2,
        },
        knnTest: {
          rmse: backendTrainResult.knn_metrics.test.rmse,
          mape: backendTrainResult.knn_metrics.test.mae,
          r2: backendTrainResult.knn_metrics.test.r2,
        },
      };
    }
    if (!result) return null;
    return {
      linearTrain: getMetrics(result.trainActual, result.trainLinear),
      linearTest: getMetrics(result.actual, result.linear),
      knnTrain: getMetrics(result.trainActual, result.trainKnn),
      knnTest: getMetrics(result.actual, result.knn),
    };
  }, [backendTrainResult, result]);

  // Load Real Multi-Dataset from FastAPI Backend
  async function handleLoadRealDatasets() {
    setLoadingMerged(true);
    try {
      const mergedRes = await loadMergedDataset(sampleSizeSetting);
      const normalizedRows = (mergedRes.sample_preview as Record<string, unknown>[]).map(normalizeUploadedRow);
      setRows(normalizedRows);
      setProcessedRows([]);
      setResult(null);
      setBackendTrainResult(null);
      setSourceName(`Merged Rossmann Data · ${numberFormat(mergedRes.total_loaded_rows)} rows across 5 tables`);
      setStage(0);
    } catch (err) {
      console.error("Failed to load merged dataset from backend:", err);
    } finally {
      setLoadingMerged(false);
    }
  }

  function createRows() {
    const generated = generateSyntheticData({ stores, days, noise, missingPercent, outlierPercent });
    setRows(generated);
    setProcessedRows([]);
    setResult(null);
    setBackendTrainResult(null);
    setSourceName("Synthetic dataset");
    setStage(0);
  }

  function uploadCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    Papa.parse<Record<string, unknown>>(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: ({ data }) => {
        const parsed = data.slice(0, 20_000).map(normalizeUploadedRow);
        setRows(parsed);
        setProcessedRows([]);
        setResult(null);
        setBackendTrainResult(null);
        setSourceName(`${file.name} · ${numberFormat(parsed.length)} rows`);
        setStage(0);
      },
    });
    event.target.value = "";
  }

  // Model Training Execution (FastAPI Backend or Client Fallback)
  async function runTraining() {
    setTraining(true);
    try {
      if (backendReady) {
        const res = await trainModelsAPI({
          features,
          splitPercent,
          k,
          metric,
          weighted,
          sampleSize: sampleSizeSetting,
        });
        setBackendTrainResult(res);
        // Also run KNN sweep for chart
        try {
          const sweep = await runKNNSweepAPI(features, 10000);
          setKnnSweep(sweep);
        } catch (e) {
          console.warn("KNN sweep warning:", e);
        }
      } else {
        const nextResult = trainAndEvaluate(effectiveRows, {
          features: features as (keyof RossmannRow)[],
          splitPercent,
          k,
          metric,
          weighted,
        });
        setResult(nextResult);
      }
    } catch (error) {
      console.error("Training error:", error);
    } finally {
      setTraining(false);
    }
  }

  // Live Scenario Prediction
  useEffect(() => {
    async function updateLivePrediction() {
      if (backendReady && backendTrainResult) {
        try {
          const pred = await predictLiveAPI({
            Customers: customersInput,
            DayOfWeek: dayInput,
            Promo: promoInput ? 1 : 0,
            SchoolHoliday: schoolInput ? 1 : 0,
            CompetitionDistance: compDistanceInput,
            Mean_TemperatureC: tempInput,
          });
          setLivePredictionData({
            linear: pred.linear_prediction,
            knn: pred.knn_prediction,
          });
        } catch (e) {
          console.warn("Live prediction API failed, fallback to client:", e);
        }
      }
    }
    const timer = setTimeout(updateLivePrediction, 250);
    return () => clearTimeout(timer);
  }, [backendReady, backendTrainResult, customersInput, dayInput, promoInput, schoolInput, compDistanceInput, tempInput]);

  async function runFullSimulation() {
    setFullRun(true);
    setProgress(15);
    if (backendReady) {
      await handleLoadRealDatasets();
      setProgress(45);
      setStage(1);
      await new Promise((r) => setTimeout(r, 180));
      setProgress(65);
      setStage(3);
      await new Promise((r) => setTimeout(r, 180));
      setProgress(80);
      setStage(4);
      await runTraining();
      setProgress(100);
      setStage(5);
    } else {
      const generated = rows.length ? rows : generateSyntheticData({ stores, days, noise, missingPercent, outlierPercent });
      if (!rows.length) {
        setRows(generated);
        setSourceName("Synthetic dataset");
      }
      setProgress(45);
      await new Promise((r) => setTimeout(r, 150));
      const prepared = preprocessData(generated, preprocess).rows;
      setProcessedRows(prepared);
      setProgress(75);
      await new Promise((r) => setTimeout(r, 150));
      const nextResult = trainAndEvaluate(prepared, {
        features: features as (keyof RossmannRow)[],
        splitPercent,
        k,
        metric,
        weighted,
      });
      setResult(nextResult);
      setProgress(100);
      setStage(5);
    }
    setFullRun(false);
  }

  function updatePreprocess(key: keyof PreprocessOptions, value: boolean | "mean" | "median") {
    const next = { ...preprocess, [key]: value } as PreprocessOptions;
    setPreprocess(next);
    setProcessedRows(preprocessData(rows, next).rows);
    setResult(null);
    setBackendTrainResult(null);
  }

  function downloadResults() {
    if (!metrics) return;
    const lines = [
      ["Model", "Train RMSE", "Test RMSE", "Test R2"],
      ["Linear Regression", metrics.linearTrain.rmse, metrics.linearTest.rmse, metrics.linearTest.r2],
      ["KNN Regression", metrics.knnTrain.rmse, metrics.knnTest.rmse, metrics.knnTest.r2],
    ];
    const csv = lines.map((line) => line.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "rossmann-model-results.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function toggleFeature(feature: string, checked: boolean | "indeterminate") {
    setFeatures((current) =>
      checked === true ? [...current, feature] : current.filter((item) => item !== feature)
    );
    setResult(null);
    setBackendTrainResult(null);
  }

  function renderStage() {
    if (!rows.length && stage > 0) {
      return (
        <EmptyState title="Start with a Rossmann dataset">
          Load the merged multi-table pipeline from Google Drive or generate synthetic records to start this workflow.
        </EmptyState>
      );
    }

    switch (stage) {
      case 0:
        return (
          <>
            <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
              <Panel
                title="1. Multi-Dataset Pipeline & Ingestion"
                subtitle="Join train, store metadata, federal states, weather, and Google Trends on common keys."
                accent="primary"
                badge={
                  backendReady ? (
                    <Badge variant="outline" className="border-chart-2/40 bg-chart-2/10 text-chart-2">
                      FastAPI + Drive Active
                    </Badge>
                  ) : (
                    <Badge variant="secondary">Offline / Local</Badge>
                  )
                }
              >
                <div className="mb-5 space-y-3 rounded-xl border border-border/80 bg-muted/30 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-foreground">Relational Sources Detected:</span>
                    <span className="rounded-full bg-background px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground ring-1 ring-border">
                      {datasetFiles
                        ? `${Object.values(datasetFiles.files).filter((f) => f.exists).length}/${Object.keys(datasetFiles.files).length} tables verified`
                        : "Multi-table schema"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2.5 text-xs sm:grid-cols-3">
                    {[
                      { name: "train.csv", meta: "36.3 MB · Sales & Promos", tone: "text-primary", dot: "bg-primary" },
                      { name: "store.csv", meta: "45 KB · 1,115 Stores", tone: "text-chart-2", dot: "bg-chart-2" },
                      { name: "weather.csv", meta: "1.45 MB · Daily Weather", tone: "text-chart-3", dot: "bg-chart-3" },
                      { name: "googletrend.csv", meta: "86 KB · Search Volume", tone: "text-foreground", dot: "bg-chart-4" },
                      { name: "store_states.csv", meta: "16 Federal States", tone: "text-muted-foreground", dot: "bg-muted-foreground" },
                      { name: "state_names.csv", meta: "Full State Names", tone: "text-muted-foreground", dot: "bg-muted-foreground" },
                    ].map((src) => {
                      const liveFile = datasetFiles?.files[src.name];
                      return (
                        <div key={src.name} className="group relative overflow-hidden rounded-lg border border-border bg-card p-2.5 transition-all hover:border-primary/40 hover:shadow-xs">
                          <div className="flex items-center justify-between gap-1">
                            <p className={`truncate font-semibold ${src.tone}`}>{src.name}</p>
                            <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${liveFile ? (liveFile.exists ? "bg-chart-2" : "bg-destructive") : src.dot}`} />
                          </div>
                          <p className="mt-0.5 text-[11px] text-muted-foreground">
                            {liveFile?.size_mb ? `${liveFile.size_mb.toFixed(2)} MB · Ready` : src.meta}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <div className="mb-2 flex items-center justify-between text-xs">
                      <label className="font-medium text-foreground">Dataset Sample Size</label>
                      <span className="tabular-nums font-semibold text-primary">
                        {numberFormat(sampleSizeSetting)} rows
                      </span>
                    </div>
                    <Slider
                      value={[sampleSizeSetting]}
                      min={5000}
                      max={100000}
                      step={5000}
                      onValueChange={(v) => setSampleSizeSetting(v[0] ?? 25000)}
                    />
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      Sub-second interactive execution in Python memory.
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2 pt-2">
                    <Button
                      disabled={loadingMerged}
                      onClick={handleLoadRealDatasets}
                      className="bg-primary text-primary-foreground hover:bg-primary/90"
                    >
                      <Database className="h-4 w-4" />
                      {loadingMerged ? "Joining Tables in Backend…" : "Load Merged Pipeline (Google Drive)"}
                    </Button>
                    <Button variant="outline" onClick={createRows}>
                      <WandSparkles className="h-4 w-4" />
                      Synthetic Generator
                    </Button>
                    <Button variant="ghost" onClick={() => fileInput.current?.click()}>
                      <FileUp className="h-4 w-4" />
                      Upload CSV
                    </Button>
                    <input ref={fileInput} className="hidden" type="file" accept=".csv,text/csv" onChange={uploadCsv} />
                  </div>
                </div>
              </Panel>

              <Panel title="The Retail Story & Motivation" subtitle="Store Turnover Forecasting in Fast-Moving Retail" accent="success">
                <div className="rounded-lg border-l-3 border-primary bg-primary/5 p-3.5 text-xs leading-6 text-foreground">
                  <p>
                    Rossmann operates over 3,000 drug stores across Europe. Predicting daily sales accurately ensures
                    optimized staff scheduling, prevents stockouts during promotions, and guides competitive inventory
                    allocations.
                  </p>
                  <p className="mt-2 text-muted-foreground">
                    By linking store formats, competitor proximity, and meteorological variables, our models learn
                    multivariate patterns rather than simple historical averages.
                  </p>
                </div>
                <div className="mt-5 grid grid-cols-3 gap-3">
                  <Metric label="Data source" value={rows.length ? "Ready" : "—"} detail={sourceName.split("·")[0] ?? ""} color="primary" />
                  <Metric label="Loaded rows" value={rows.length ? numberFormat(rows.length) : "0"} detail="Store × date" color="success" />
                  <Metric label="ML Models" value="02" detail="Linear · KNN" color="warning" />
                </div>
              </Panel>
            </div>

            {rows.length > 0 && (
              <div className="mt-6 space-y-5">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Metric label="Total Rows" value={numberFormat(rows.length)} detail="Ready in memory" color="primary" />
                  <Metric label="Missing values" value={numberFormat(countMissing(rows))} detail="Handled in Stage 3" color="warning" />
                  <Metric label="Open days" value={`${Math.round((rows.filter((r) => r.Open === 1).length / rows.length) * 100)}%`} detail="Active trading" color="success" />
                  <Metric label="Promotional days" value={`${Math.round((rows.filter((r) => r.Promo === 1).length / rows.length) * 100)}%`} detail="With coupon lift" color="neutral" />
                </div>
                <Panel title="Data preview" subtitle={`Showing first ${Math.min(rows.length, 12)} rows · ${sourceName}`}>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[820px] text-left text-xs">
                      <thead>
                        <tr className="border-b border-border text-muted-foreground">
                          {Object.keys(rows[0] ?? {}).slice(0, 10).map((column) => (
                            <th key={column} className="px-3 py-2.5 font-medium">{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.slice(0, 8).map((row, index) => (
                          <tr key={`${row.Store}-${row.Date}-${index}`} className="border-b border-border/60 transition-colors hover:bg-muted/30 last:border-0">
                            {Object.values(row).slice(0, 10).map((value, columnIndex) => (
                              <td key={`${columnIndex}`} className="px-3 py-2 tabular-nums">
                                {value === null ? (
                                  <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] font-medium text-destructive">null</span>
                                ) : typeof value === "number" ? (
                                  numberFormat(value)
                                ) : (
                                  String(value)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              </div>
            )}
          </>
        );

      case 1:
        return analysis ? (
          <div className="space-y-5">
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <Metric label="Sales mean" value={money(analysis.salesStats.mean)} detail={`SD ${money(analysis.salesStats.sd)}`} color="primary" />
              <Metric label="Customer mean" value={numberFormat(analysis.customerStats.mean)} detail={`SD ${numberFormat(analysis.customerStats.sd)}`} color="neutral" />
              <Metric label="Promo sales lift" value={`+${analysis.uplift.toFixed(1)}%`} detail="Turnover boost from promos" color="success" />
              <Metric label="Analyzed rows" value={numberFormat(effectiveRows.length)} detail="Active in memory" color="warning" />
            </div>
            <div className="grid gap-5 xl:grid-cols-2">
              <ChartPanel title="Average sales by year" subtitle="Year-over-year trading performance" data={analysis.yearly} dataKey="sales" xKey="label" color="var(--chart-1)" formatter={money} />
              <ChartPanel title="Sales by day of week" subtitle="Mean daily sales (1=Mon..7=Sun)" data={analysis.weekdays} dataKey="sales" xKey="label" color="var(--chart-3)" formatter={money} />
              <Panel title="Sales vs customers" subtitle="Each point represents a store-day observation.">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 8, right: 10, bottom: 8, left: 6 }}>
                      <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                      <XAxis dataKey="customers" type="number" name="Customers" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
                      <YAxis dataKey="sales" type="number" name="Sales" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
                      <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: "3 3" }} />
                      <Scatter name="Store days" data={analysis.scatter} fill="var(--chart-2)" fillOpacity={0.54} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
              <ChartPanel title="Promo comparison" subtitle="Sales uplift under coupon promotions" data={analysis.promo} dataKey="sales" secondaryKey="customers" xKey="label" color="var(--chart-1)" secondaryColor="var(--chart-3)" formatter={money} />
              <HistogramPanel title="Sales distribution" bins={analysis.salesHist} mean={analysis.salesStats.mean} sd={analysis.salesStats.sd} color="var(--chart-1)" />
              <HistogramPanel title="Customer distribution" bins={analysis.customerHist} mean={analysis.customerStats.mean} sd={analysis.customerStats.sd} color="var(--chart-2)" />
            </div>
          </div>
        ) : null;

      case 2:
        return (
          <div className="space-y-5">
            <Panel title="Data Preparation & Hygiene" subtitle="Strict ML best practice: fit scaling and imputation parameters on train fold only." accent="primary">
              <div className="divide-y divide-border">
                <ToggleRow label="Remove duplicate records" description="Eliminate identical transaction logs." checked={preprocess.removeDuplicates} onChange={(checked) => updatePreprocess("removeDuplicates", checked)} />
                <ToggleRow label="Drop closed days" description="Filter non-trading days (Open == 0 and Sales == 0) to prevent zero inflation." checked={preprocess.dropClosed} onChange={(checked) => updatePreprocess("dropClosed", checked)} />
                <ToggleRow label="Clip outliers" description="Clamp extreme sales/customer spikes using standard deviation bounds." checked={preprocess.clipOutliers} onChange={(checked) => updatePreprocess("clipOutliers", checked)} />
              </div>
              <div className="mt-5 grid items-center gap-3 border-t border-border pt-4 sm:grid-cols-[1fr_200px]">
                <div>
                  <p className="text-sm font-medium">Missing-value imputation strategy</p>
                  <p className="mt-1 text-xs text-muted-foreground">Contextual replacement for competitor distances and missing flags.</p>
                </div>
                <Select value={preprocess.imputation} onValueChange={(value: string) => updatePreprocess("imputation", value as "mean" | "median")}>
                  <SelectTrigger aria-label="Imputation method"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mean">Column mean</SelectItem>
                    <SelectItem value="median">Column median</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </Panel>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <Metric label="Rows before" value={numberFormat(prepResult.summary.beforeRows)} detail="Loaded from source" color="neutral" />
              <Metric label="Rows after" value={numberFormat(prepResult.summary.afterRows)} detail="Ready for modeling" color="primary" />
              <Metric label="Missing values" value={`${numberFormat(prepResult.summary.beforeMissing)} → 0`} detail="After contextual imputation" color="success" />
              <Metric label="Duplicates removed" value={numberFormat(prepResult.summary.duplicatesRemoved)} detail="Identical entries dropped" color="warning" />
            </div>
            <Panel title="Prepared records" subtitle="Transformed dataset preview ready for feature selection.">
              <PreviewTable rows={prepResult.rows} />
            </Panel>
          </div>
        );

      case 3: {
        const featureCategoryMap: Record<string, { label: string; tone: string }> = {
          Customers: { label: "Traffic", tone: "bg-primary/10 text-primary" },
          Promo: { label: "Promo", tone: "bg-chart-2/15 text-chart-2" },
          Promo2Active: { label: "Promo", tone: "bg-chart-2/15 text-chart-2" },
          DayOfWeek: { label: "Calendar", tone: "bg-chart-4/15 text-chart-4" },
          SchoolHoliday: { label: "Calendar", tone: "bg-chart-4/15 text-chart-4" },
          Month: { label: "Calendar", tone: "bg-chart-4/15 text-chart-4" },
          WeekOfYear: { label: "Calendar", tone: "bg-chart-4/15 text-chart-4" },
          IsWeekend: { label: "Calendar", tone: "bg-chart-4/15 text-chart-4" },
          StoreType: { label: "Store", tone: "bg-chart-3/15 text-chart-3" },
          Assortment: { label: "Store", tone: "bg-chart-3/15 text-chart-3" },
          CompetitionDistance: { label: "Rival", tone: "bg-chart-3/15 text-chart-3" },
          MonthsSinceCompetition: { label: "Rival", tone: "bg-chart-3/15 text-chart-3" },
          Mean_TemperatureC: { label: "Weather", tone: "bg-chart-1/15 text-chart-1" },
          Precipitationmm: { label: "Weather", tone: "bg-chart-1/15 text-chart-1" },
          Mean_Humidity: { label: "Weather", tone: "bg-chart-1/15 text-chart-1" },
          Max_Wind_SpeedKm_h: { label: "Weather", tone: "bg-chart-1/15 text-chart-1" },
          trend: { label: "Search", tone: "bg-chart-5/15 text-chart-5" },
        };
        const allFeatureKeys = [
          "Customers", "Promo", "DayOfWeek", "SchoolHoliday",
          "StoreType", "Assortment", "CompetitionDistance",
          "MonthsSinceCompetition", "Promo2Active",
          "Mean_TemperatureC", "Precipitationmm", "Mean_Humidity",
          "Max_Wind_SpeedKm_h", "trend", "Month", "WeekOfYear", "IsWeekend",
        ];

        return (
          <div className="space-y-5">
            {analysis && (
              <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
                <Panel title="Correlation with Sales" subtitle="Pearson linear correlation coefficient across numeric features." accent="primary">
                  <div className="space-y-3">
                    {analysis.correlations.map(({ feature, value }) => (
                      <div key={feature} className="grid grid-cols-[140px_1fr_60px] items-center gap-3 text-xs">
                        <span className="truncate font-medium text-foreground">{featureLabels[feature] ?? feature}</span>
                        <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${value >= 0 ? "bg-gradient-to-r from-chart-1 to-chart-2" : "bg-gradient-to-r from-chart-5 to-chart-3"}`}
                            style={{ width: `${Math.max(4, Math.abs(value) * 100)}%` }}
                          />
                        </div>
                        <span className={`text-right font-semibold tabular-nums ${value >= 0 ? "text-chart-1" : "text-chart-3"}`}>
                          {value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </Panel>
                <Panel title="Correlation Matrix Heatmap" subtitle="Pairwise collinearity across key features." accent="warning">
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[420px] text-center text-[10px]">
                      <thead>
                        <tr>
                          <th className="p-1.5" />
                          {numericColumns.map((col) => (
                            <th key={col} className="p-1.5 font-medium text-muted-foreground">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {analysis.heatmap.map((row) => (
                          <tr key={row.feature}>
                            <th className="max-w-24 truncate p-1.5 text-left font-medium text-muted-foreground">{row.feature}</th>
                            {row.values.map(({ feature, value }) => (
                              <td key={feature} className="p-1">
                                <div
                                  className="rounded-md py-2 font-semibold tabular-nums transition-transform hover:scale-105"
                                  style={{
                                    backgroundColor: `color-mix(in oklab, ${value >= 0 ? "var(--chart-1)" : "var(--chart-3)"} ${Math.max(10, Math.abs(value) * 75)}%, var(--muted))`,
                                    color: "var(--foreground)",
                                  }}
                                >
                                  {value.toFixed(2)}
                                </div>
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              </div>
            )}

            <Panel
              title="Select Features from Merged Family"
              subtitle="Choose signals across Store, Weather, Search Trends, and Calendar components."
              accent="success"
              badge={
                <div className="flex flex-wrap gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setFeatures(["Customers", "Promo", "DayOfWeek", "SchoolHoliday", "CompetitionDistance", "Mean_TemperatureC"]);
                      setResult(null);
                      setBackendTrainResult(null);
                    }}
                  >
                    Core 6 Signals
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setFeatures(allFeatureKeys);
                      setResult(null);
                      setBackendTrainResult(null);
                    }}
                  >
                    Select All ({allFeatureKeys.length})
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setFeatures(["Customers", "Promo"]);
                      setResult(null);
                      setBackendTrainResult(null);
                    }}
                  >
                    Minimal (2)
                  </Button>
                </div>
              }
            >
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {allFeatureKeys.map((feat) => {
                  const isSelected = features.includes(feat);
                  const cat = featureCategoryMap[feat] ?? { label: "Feature", tone: "bg-muted text-muted-foreground" };
                  const catalogMeta = backendFeatures.find((f) => f.name === feat);
                  return (
                    <label
                      key={feat}
                      className={`group flex cursor-pointer items-center gap-3 rounded-xl border px-3.5 py-3 text-sm transition-all duration-150 ${
                        isSelected
                          ? "border-primary/50 bg-primary/5 shadow-2xs"
                          : "border-border bg-card hover:border-border/80 hover:bg-muted/40"
                      }`}
                    >
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={(checked) => toggleFeature(feat, checked)}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate text-xs font-semibold text-foreground">{featureLabels[feat] ?? feat}</p>
                          <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${cat.tone}`}>
                            {catalogMeta?.category ?? cat.label}
                          </span>
                        </div>
                        <div className="mt-0.5 flex items-center justify-between text-[10px] text-muted-foreground">
                          <span className="font-mono">{feat}</span>
                          {catalogMeta && (
                            <span>{catalogMeta.dtype} · {catalogMeta.missing_percent}% null</span>
                          )}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
                <span>
                  <strong className="text-foreground">{features.length}</strong> of {allFeatureKeys.length} features selected for model fitting
                </span>
                <Badge variant="outline">{backendReady ? "Scikit-Learn StandardScaler" : "Browser Z-score Normalizer"}</Badge>
              </div>
            </Panel>
          </div>
        );
      }

      case 4: {
        const activeCoefficients = backendTrainResult?.coefficients ?? (
          result
            ? features.map((feat, idx) => {
                const coeff = result.coefficients[idx] ?? 0;
                return {
                  feature: feat,
                  coefficient: coeff,
                  abs_importance: Math.abs(coeff),
                };
              })
            : []
        );

        return (
          <div className="space-y-5">
            <Panel title="Model Training Configuration" subtitle="Fit Linear Regression baseline and K-Nearest Neighbors in Scikit-Learn." accent="primary">
              <div className="grid gap-6 xl:grid-cols-2">
                <div>
                  <SliderField label="Training split" value={splitPercent} unit="%" min={60} max={90} onChange={(v) => { setSplitPercent(v); setResult(null); setBackendTrainResult(null); }} />
                  <p className="mt-2 text-xs text-muted-foreground">{splitPercent}% train partition · {100 - splitPercent}% held-out test partition</p>
                </div>
                <div>
                  <SliderField label="K nearest neighbors" value={k} unit="" min={1} max={50} onChange={(v) => { setK(v); setResult(null); setBackendTrainResult(null); }} />
                  <p className="mt-2 text-xs text-muted-foreground">{weighted ? "Inverse distance weighted" : "Uniform neighbor averaging"}</p>
                </div>
                <div>
                  <p className="mb-2 text-xs font-medium text-muted-foreground">Distance metric</p>
                  <Select value={metric} onValueChange={(value: string) => { setMetric(value as ModelOptions["metric"]); setResult(null); setBackendTrainResult(null); }}>
                    <SelectTrigger aria-label="Distance metric"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="euclidean">Euclidean (L2 norm)</SelectItem>
                      <SelectItem value="manhattan">Manhattan (L1 norm)</SelectItem>
                      <SelectItem value="minkowski">Minkowski (p = 3)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between rounded-lg border border-border px-3.5 py-2.5">
                  <div>
                    <p className="text-sm font-medium">Distance weighting</p>
                    <p className="text-xs text-muted-foreground">Closer neighbors exert higher influence.</p>
                  </div>
                  <Switch checked={weighted} onCheckedChange={(checked) => { setWeighted(checked); setResult(null); setBackendTrainResult(null); }} aria-label="Distance weighting" />
                </div>
              </div>
              <div className="mt-6 flex flex-wrap items-center gap-3">
                <Button
                  disabled={training || features.length === 0}
                  onClick={runTraining}
                  className="bg-gradient-to-r from-primary to-primary/85 text-primary-foreground shadow-xs hover:from-primary/90 hover:to-primary/75"
                >
                  <WandSparkles className="h-4 w-4" />
                  {training ? "Fitting models…" : "Train Linear Regression vs KNN"}
                </Button>
                <span className="text-xs text-muted-foreground">
                  {backendReady ? "Powered by Scikit-Learn & NumPy" : "Client-side calculation"} · {features.length} features selected
                </span>
              </div>
            </Panel>

            {(backendTrainResult || result) && (
              <div className="space-y-5">
                <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
                  <Panel
                    title="OLS Regression Equation"
                    subtitle="Coefficients: marginal turnover per standardized feature unit"
                    accent="primary"
                  >
                    <div className="break-words rounded-lg bg-muted/40 p-3.5 font-mono text-xs leading-7 ring-1 ring-border/60 sm:text-sm">
                      Sales = {money(backendTrainResult?.intercept ?? result?.intercept ?? 0)}
                      {activeCoefficients.map((c) => (
                        <span key={c.feature} className={c.coefficient >= 0 ? "text-chart-1" : "text-chart-5"}>
                          {" "}{c.coefficient >= 0 ? "+" : "−"} {money(Math.abs(c.coefficient))} × z({featureLabels[c.feature] ?? c.feature})
                        </span>
                      ))}
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {activeCoefficients.map((c) => (
                        <span
                          key={c.feature}
                          className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${c.coefficient >= 0 ? "border-chart-1/30 bg-chart-1/10 text-chart-1" : "border-chart-5/30 bg-chart-5/10 text-chart-5"}`}
                        >
                          {featureLabels[c.feature] ?? c.feature}
                          <span className="font-bold">{c.coefficient >= 0 ? `+${c.coefficient.toFixed(1)}` : c.coefficient.toFixed(1)}</span>
                        </span>
                      ))}
                    </div>
                  </Panel>

                  {activeCoefficients.length > 0 && (
                    <FeatureImportanceChart coefficients={activeCoefficients} />
                  )}
                </div>

                <div className="grid gap-5 xl:grid-cols-2">
                  <PredictionScatter
                    title="Linear Regression · Held-Out Test Data"
                    actual={backendTrainResult ? backendTrainResult.prediction_samples.map((s) => s.actual) : result?.actual ?? []}
                    predicted={backendTrainResult ? backendTrainResult.prediction_samples.map((s) => s.linear_pred) : result?.linear ?? []}
                    color="var(--chart-1)"
                  />
                  <PredictionScatter
                    title="KNN Regression · Held-Out Test Data"
                    actual={backendTrainResult ? backendTrainResult.prediction_samples.map((s) => s.actual) : result?.actual ?? []}
                    predicted={backendTrainResult ? backendTrainResult.prediction_samples.map((s) => s.knn_pred) : result?.knn ?? []}
                    color="var(--chart-3)"
                  />
                </div>
              </div>
            )}
          </div>
        );
      }

      case 5: {
        if (!metrics) {
          return (
            <EmptyState title="Models Awaiting Training">
              Execute model fitting in the previous step to evaluate RMSE, cross-validation, and bias-variance tradeoff.
            </EmptyState>
          );
        }
        const knnWins = metrics.knnTest.rmse < metrics.linearTest.rmse;
        const bestRmse = Math.min(metrics.linearTest.rmse, metrics.knnTest.rmse);
        const bestR2 = Math.max(metrics.linearTest.r2, metrics.knnTest.r2);
        const rmseDelta = Math.abs(metrics.linearTest.rmse - metrics.knnTest.rmse);
        const sweepData = knnSweep?.sweep_results ?? (result?.sweep ? result.sweep.map((s) => ({ k: s.k, test_rmse: s.rmse })) : null);
        const cvRows = backendTrainResult?.cv_results ?? (result?.cv ? result.cv.map((c) => ({ fold: c.fold, linear_rmse: c.linearRmse, knn_rmse: c.knnRmse })) : null);

        return (
          <div className="space-y-5">
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <Metric
                label="Best Model (Held-Out)"
                value={knnWins ? `KNN (k=${k})` : "Linear OLS"}
                detail={knnWins ? "Local neighborhood fit wins" : "Parametric hyperplane wins"}
                color="success"
              />
              <Metric
                label="Best Test RMSE"
                value={money(bestRmse)}
                detail="Out-of-sample prediction error"
                color="primary"
              />
              <Metric
                label="Peak Variance Explained"
                value={`${(bestR2 * 100).toFixed(1)}%`}
                detail={`R² = ${bestR2.toFixed(3)} on test split`}
                color="warning"
              />
              <Metric
                label="Architecture Delta"
                value={money(rmseDelta)}
                detail="RMSE spread between models"
                color="neutral"
              />
            </div>

            <Panel
              title="Model Performance Evaluation"
              subtitle="Held-out validation comparison across root mean squared error and variance explained (R²)."
              accent="success"
              badge={
                <Button variant="outline" size="sm" onClick={downloadResults}>
                  <ArrowDownToLine className="h-3.5 w-3.5" />
                  Export CSV
                </Button>
              }
            >
              <div className="overflow-x-auto">
                <table className="w-full min-w-[650px] text-left text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      {["Model Architecture", "Train RMSE", "Test RMSE", "Test MAE / MAPE", "Test R²", "Generalization Status"].map((l) => (
                        <th key={l} className="px-3 py-3 font-medium">{l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className={`border-b border-border transition-colors ${!knnWins ? "bg-chart-2/5" : ""}`}>
                      <td className="px-3 py-3.5 font-semibold text-foreground">
                        <div className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full bg-chart-1" />
                          Linear Regression (OLS)
                          {!knnWins && (
                            <span className="rounded-full bg-chart-2/15 px-2 py-0.5 text-[10px] font-bold text-chart-2">★ Winner</span>
                          )}
                        </div>
                      </td>
                      <td className="px-3 py-3.5 tabular-nums">{money(metrics.linearTrain.rmse)}</td>
                      <td className="px-3 py-3.5 font-semibold tabular-nums text-primary">{money(metrics.linearTest.rmse)}</td>
                      <td className="px-3 py-3.5 tabular-nums">{money(metrics.linearTest.mape)}</td>
                      <td className="px-3 py-3.5">
                        <div className="flex items-center gap-2">
                          <span className="w-10 font-semibold tabular-nums">{metrics.linearTest.r2.toFixed(3)}</span>
                          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                            <div className="h-full rounded-full bg-chart-1" style={{ width: `${Math.max(0, Math.min(100, metrics.linearTest.r2 * 100))}%` }} />
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-3.5"><Badge variant="outline">Stable Baseline</Badge></td>
                    </tr>
                    <tr className={`transition-colors ${knnWins ? "bg-chart-2/5" : ""}`}>
                      <td className="px-3 py-3.5 font-semibold text-foreground">
                        <div className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full bg-chart-3" />
                          K-Nearest Neighbors (KNN, k={k})
                          {knnWins && (
                            <span className="rounded-full bg-chart-2/15 px-2 py-0.5 text-[10px] font-bold text-chart-2">★ Winner</span>
                          )}
                        </div>
                      </td>
                      <td className="px-3 py-3.5 tabular-nums">{money(metrics.knnTrain.rmse)}</td>
                      <td className="px-3 py-3.5 font-semibold tabular-nums text-chart-3">{money(metrics.knnTest.rmse)}</td>
                      <td className="px-3 py-3.5 tabular-nums">{money(metrics.knnTest.mape)}</td>
                      <td className="px-3 py-3.5">
                        <div className="flex items-center gap-2">
                          <span className="w-10 font-semibold tabular-nums">{metrics.knnTest.r2.toFixed(3)}</span>
                          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                            <div className="h-full rounded-full bg-chart-3" style={{ width: `${Math.max(0, Math.min(100, metrics.knnTest.r2 * 100))}%` }} />
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-3.5">
                        <Badge variant={knnWins ? "default" : "secondary"}>
                          {knnWins ? "Nonlinear Fit Win" : "Local Neighbor Noise"}
                        </Badge>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Panel>

            <div className="grid gap-5 xl:grid-cols-2">
              <Panel title="Train vs Test RMSE Comparison" subtitle="Lower error = better generalization. Train < Test gap indicates overfitting." accent="warning">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { model: "Linear", train: metrics.linearTrain.rmse, test: metrics.linearTest.rmse },
                        { model: "KNN", train: metrics.knnTrain.rmse, test: metrics.knnTest.rmse },
                      ]}
                      barCategoryGap="30%"
                    >
                      <defs>
                        <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--chart-2)" stopOpacity={0.9} />
                          <stop offset="100%" stopColor="var(--chart-2)" stopOpacity={0.5} />
                        </linearGradient>
                        <linearGradient id="testGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.9} />
                          <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.5} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                      <XAxis dataKey="model" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                      <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} tickFormatter={(val) => numberFormat(Number(val))} />
                      <Tooltip {...tooltipStyle} formatter={(val) => money(Number(val))} />
                      <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                      <Bar dataKey="train" name="Train RMSE" fill="url(#trainGrad)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="test" name="Test RMSE" fill="url(#testGrad)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              {sweepData && (
                <ChartPanel
                  title="KNN Bias-Variance Curve (K Sweep)"
                  subtitle="Optimal K minimizes test RMSE while preventing overfitting."
                  data={sweepData}
                  dataKey="test_rmse"
                  xKey="k"
                  color="var(--chart-3)"
                  formatter={money}
                />
              )}
            </div>

            {cvRows && (
              <Panel title="5-Fold Cross-Validation Summary" subtitle="Evaluated across non-overlapping partitions to test variance stability." accent="primary">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[420px] text-left text-xs">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground">
                        <th className="px-3 py-2.5 font-medium">Fold</th>
                        <th className="px-3 py-2.5 font-medium">Linear RMSE</th>
                        <th className="px-3 py-2.5 font-medium">KNN RMSE</th>
                        <th className="px-3 py-2.5 font-medium">Best Performer</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cvRows.map((f) => (
                        <tr key={f.fold} className="border-b border-border/70 transition-colors hover:bg-muted/30">
                          <td className="px-3 py-2.5 font-semibold">Fold {f.fold}</td>
                          <td className="px-3 py-2.5 tabular-nums">{money(f.linear_rmse)}</td>
                          <td className="px-3 py-2.5 tabular-nums">{money(f.knn_rmse)}</td>
                          <td className="px-3 py-2.5">
                            <Badge variant={f.linear_rmse <= f.knn_rmse ? "outline" : "secondary"}>
                              {f.linear_rmse <= f.knn_rmse ? "Linear" : "KNN"}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            )}

            <div className="flex flex-wrap gap-2">
              <Button onClick={downloadResults}><ArrowDownToLine className="h-4 w-4" />Download Evaluation CSV</Button>
              <Button variant="outline" onClick={() => setStage(4)}><RotateCcw className="h-4 w-4" />Adjust Parameters & Retrain</Button>
            </div>
          </div>
        );
      }

      case 6: {
        // Compute offline client-side prediction when backend unavailable
        const offlinePred = (!backendReady && result && effectiveRows.length > 0)
          ? (() => {
              try {
                const row: RossmannRow = {
                  Store: 1, Date: new Date().toISOString().slice(0, 10),
                  Sales: null, Open: 1, StateHoliday: "0",
                  Customers: customersInput, DayOfWeek: dayInput,
                  Promo: promoInput ? 1 : 0, SchoolHoliday: schoolInput ? 1 : 0,
                };
                return predictOne(row, features as (keyof RossmannRow)[], result, effectiveRows, metric, k, weighted);
              } catch { return null; }
            })()
          : null;
        const linearPred = livePredictionData?.linear ?? offlinePred?.linear ?? 0;
        const knnPred = livePredictionData?.knn ?? offlinePred?.knn ?? 0;
        const hasPrediction = linearPred > 0 || knnPred > 0;
        const advisoryAmount = Math.max(linearPred, knnPred);
        const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
        const scenarioPresets = [
          { label: "🔥 Promo Monday Rush", customers: 920, day: 1, promo: true, school: false, comp: 650, temp: 19 },
          { label: "☀️ Typical Midweek", customers: 560, day: 3, promo: false, school: false, comp: 1400, temp: 16 },
          { label: "🎄 Holiday Family Peak", customers: 1080, day: 5, promo: true, school: true, comp: 2200, temp: 8 },
          { label: "❄️ Cold Quiet Tuesday", customers: 390, day: 2, promo: false, school: false, comp: 450, temp: 1 },
        ];

        return (
          <div className="space-y-5">
            <div className="grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
              {/* Left: Scenario controls */}
              <Panel
                title="Retail Scenario Parameters"
                subtitle="Adjust conditions or pick a preset to instantly forecast store turnover"
                accent="primary"
              >
                <div className="space-y-4">
                  {/* Scenario quick-presets */}
                  <div>
                    <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                      Quick Scenario Presets
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {scenarioPresets.map((preset) => (
                        <button
                          key={preset.label}
                          type="button"
                          onClick={() => {
                            setCustomersInput(preset.customers);
                            setDayInput(preset.day);
                            setPromoInput(preset.promo);
                            setSchoolInput(preset.school);
                            setCompDistanceInput(preset.comp);
                            setTempInput(preset.temp);
                          }}
                          className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-left text-xs font-medium text-foreground transition-all hover:border-primary/40 hover:bg-primary/5"
                        >
                          {preset.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-muted-foreground">Customer Foot-Traffic</span>
                        <span className="text-xs font-bold tabular-nums text-primary">{numberFormat(customersInput)}</span>
                      </div>
                      <Input
                        type="number"
                        min={0}
                        value={customersInput}
                        onChange={(e) => setCustomersInput(Number(e.target.value))}
                      />
                      <Slider
                        value={[Math.min(1500, Math.max(100, customersInput))]}
                        min={100}
                        max={1500}
                        step={10}
                        onValueChange={(v) => setCustomersInput(v[0] ?? 550)}
                      />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-muted-foreground">Competitor Distance (m)</span>
                        <span className="text-xs font-bold tabular-nums text-foreground">{numberFormat(compDistanceInput)}m</span>
                      </div>
                      <Input
                        type="number"
                        min={0}
                        value={compDistanceInput}
                        onChange={(e) => setCompDistanceInput(Number(e.target.value))}
                      />
                      <Slider
                        value={[Math.min(10000, Math.max(50, compDistanceInput))]}
                        min={50}
                        max={10000}
                        step={50}
                        onValueChange={(v) => setCompDistanceInput(v[0] ?? 1200)}
                      />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-muted-foreground">Regional Temperature (°C)</span>
                        <span className="text-xs font-bold tabular-nums text-foreground">{tempInput}°C</span>
                      </div>
                      <Input
                        type="number"
                        value={tempInput}
                        onChange={(e) => setTempInput(Number(e.target.value))}
                      />
                      <Slider
                        value={[Math.min(38, Math.max(-10, tempInput))]}
                        min={-10}
                        max={38}
                        step={1}
                        onValueChange={(v) => setTempInput(v[0] ?? 16)}
                      />
                    </div>
                    <label className="block space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Day of Week</span>
                      <Select value={String(dayInput)} onValueChange={(v) => setDayInput(Number(v))}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {days.map((d, i) => (
                            <SelectItem key={d} value={String(i + 1)}>{d}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </label>
                  </div>
                  <div className="divide-y divide-border rounded-lg border border-border px-3">
                    <ToggleRow label="Promotional Campaign Active" description="Coupon flyer boosts foot-traffic ~30%" checked={promoInput} onChange={setPromoInput} />
                    <ToggleRow label="School Holiday Period" description="State public school vacation period" checked={schoolInput} onChange={setSchoolInput} />
                  </div>
                  {/* Active scenario badges */}
                  <div className="flex flex-wrap gap-1.5">
                    <span className="rounded-full bg-primary/10 px-2.5 py-1 text-[11px] font-medium text-primary">{numberFormat(customersInput)} customers</span>
                    <span className="rounded-full bg-muted px-2.5 py-1 text-[11px] font-medium text-muted-foreground">{days[dayInput - 1]}</span>
                    <span className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${promoInput ? "bg-chart-2/15 text-chart-2" : "bg-muted text-muted-foreground"}`}>{promoInput ? "🏷 Promo on" : "No promo"}</span>
                    <span className="rounded-full bg-muted px-2.5 py-1 text-[11px] font-medium text-muted-foreground">{compDistanceInput}m rival</span>
                    <span className="rounded-full bg-muted px-2.5 py-1 text-[11px] font-medium text-muted-foreground">{tempInput}°C</span>
                  </div>
                </div>
              </Panel>

              {/* Right: Prediction output */}
              <div className="space-y-5">
                <Panel title="Predicted Daily Turnover" subtitle="Real-time model estimates for the configured store scenario" accent="primary">
                  {hasPrediction ? (
                    <PredictionComparison linear={linearPred} knn={knnPred} k={k} />
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-primary/10 ring-4 ring-primary/5">
                        <WandSparkles className="h-5 w-5 text-primary" />
                      </div>
                      <p className="text-sm font-semibold text-foreground">Train models to unlock live forecasting</p>
                      <p className="mt-1 max-w-xs text-xs text-muted-foreground">
                        Fit Linear Regression and KNN on your active dataset to simulate real-time store turnover.
                      </p>
                      <Button size="sm" className="mt-4" disabled={training} onClick={runTraining}>
                        <WandSparkles className="h-3.5 w-3.5" />
                        {training ? "Training Models…" : "Fit Models Now"}
                      </Button>
                    </div>
                  )}
                </Panel>

                {hasPrediction && (
                  <Panel title="Inventory Advisory" subtitle="Operational recommendation based on model consensus" accent="success">
                    <div className="space-y-3">
                      <div className="flex items-baseline justify-between">
                        <span className="text-xs text-muted-foreground">Recommended stock budget</span>
                        <span className="text-2xl font-bold tabular-nums text-chart-2">{money(advisoryAmount)}</span>
                      </div>
                      <div className="rounded-lg bg-chart-2/8 p-3 text-xs leading-5 text-foreground">
                        Plan staffing and merchandise for approximately{" "}
                        <strong>{money(advisoryAmount)}</strong> in daily sales.
                        {promoInput && " Active promotion — schedule 2 extra staff for peak morning hours."}
                        {schoolInput && " School holiday — expect +12% juvenile product demand."}
                        {customersInput > 700 && " High foot-traffic day — ensure full shelf restocking overnight."}
                      </div>
                    </div>
                  </Panel>
                )}
              </div>
            </div>
          </div>
        );
      }


      default:
        return null;
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="relative overflow-hidden border-b border-border bg-card">
        {/* Animated gradient accent bar */}
        <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-primary/0 via-primary to-chart-2/60" />
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="relative grid h-10 w-10 shrink-0 place-items-center overflow-hidden rounded-lg bg-gradient-to-br from-primary to-primary/70 text-primary-foreground shadow-sm">
              <Activity className="h-5 w-5" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-bold tracking-tight text-foreground">Rossmann Sales ML Lab</p>
              <p className="text-[11px] text-muted-foreground">Linear Regression vs KNN · 5-table relational pipeline</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold transition-colors ${backendReady ? "border-chart-2/40 bg-chart-2/10 text-chart-2" : "border-border bg-muted/50 text-muted-foreground"}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${backendReady ? "animate-pulse bg-chart-2" : "bg-muted-foreground"}`} />
              {backendReady ? "FastAPI Active" : "Client Mode"}
            </div>
            <Button
              aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
              size="icon"
              variant="ghost"
              className="rounded-lg"
              onClick={() => setDark((v) => !v)}
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </header>

      {/* Sticky stage navigation */}
      <div className="sticky top-0 z-20 border-b border-border bg-background/95 backdrop-blur-sm">
        <div className="mx-auto max-w-[1440px] px-4 sm:px-8">
          <nav aria-label="Storyflow pipeline" className="flex overflow-x-auto scrollbar-none">
            {stages.map(({ label, short, icon: Icon }, index) => (
              <button
                key={label}
                aria-current={stage === index ? "step" : undefined}
                onClick={() => setStage(index)}
                className={`group relative flex min-w-[88px] flex-1 flex-col items-center gap-1 px-2 py-3 text-center transition-all sm:min-w-0 sm:px-3 ${
                  stage === index
                    ? "text-primary"
                    : index < stage
                      ? "text-chart-2 hover:text-chart-2/80"
                      : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {/* Active indicator bar */}
                <span
                  className={`absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full transition-all duration-300 ${
                    stage === index ? "bg-primary" : index < stage ? "bg-chart-2/50" : "bg-transparent"
                  }`}
                />
                <span
                  className={`grid h-7 w-7 shrink-0 place-items-center rounded-lg text-[10px] font-semibold transition-all duration-200 ${
                    stage === index
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : index < stage
                        ? "bg-chart-2/15 text-chart-2"
                        : "bg-muted text-muted-foreground group-hover:bg-muted/80"
                  }`}
                >
                  {index < stage ? <Check className="h-3.5 w-3.5" /> : <Icon className="h-3.5 w-3.5" />}
                </span>
                <span className="hidden max-w-[80px] truncate text-[11px] font-medium md:block">{label}</span>
                <span className="text-[10px] font-medium md:hidden">{short}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Stage hero banner */}
      <div className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-chart-2/5" />
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-4 py-5 sm:px-8">
          <div className="flex items-start gap-4">
            <div className="mt-0.5 hidden rounded-lg bg-primary/10 p-2.5 sm:block">
              {(() => { const Icon = stages[stage]?.icon ?? Activity; return <Icon className="h-5 w-5 text-primary" />; })()}
            </div>
            <div>
              <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-primary">
                Step {stage + 1} / {stages.length}
              </p>
              <h1 className="mt-0.5 text-lg font-bold sm:text-xl">{stages[stage]?.label}</h1>
              <p className="mt-1 max-w-2xl text-xs leading-relaxed text-muted-foreground">{stages[stage]?.story}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              className="gap-2 bg-gradient-to-r from-primary to-primary/80 text-primary-foreground shadow-sm hover:from-primary/90 hover:to-primary/70"
              disabled={fullRun}
              onClick={runFullSimulation}
            >
              <Play className="h-4 w-4" />
              {fullRun ? "Simulating…" : "Run Full Pipeline"}
            </Button>
            {rows.length > 0 && (
              <Button variant="outline" size="sm" onClick={createRows}>
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            )}
          </div>
          {fullRun && (
            <div className="w-full space-y-1.5">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Executing ingestion → features → ML fitting</span>
                <span className="font-semibold text-primary">{progress}%</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-chart-2 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>


      <div className="mx-auto max-w-[1440px] px-4 py-6 sm:px-8 sm:py-8">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/70 bg-card/60 px-4 py-2.5">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className={`h-2 w-2 rounded-full ${rows.length ? "bg-chart-2" : "bg-muted-foreground"}`} />
            <span className="font-semibold text-foreground">
              {rows.length ? `${numberFormat(rows.length)} active observations` : "Waiting for dataset"}
            </span>
            {processedRows.length > 0 && (
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                {numberFormat(processedRows.length)} prepared
              </span>
            )}
            <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
              {features.length} features active
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={stage === 0}
              onClick={() => setStage((v) => Math.max(0, v - 1))}
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <Button
              size="sm"
              disabled={stage === stages.length - 1}
              onClick={() => setStage((v) => Math.min(stages.length - 1, v + 1))}
            >
              Next Step
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {renderStage()}

        <footer className="mt-12 flex flex-wrap items-center justify-between gap-4 border-t border-border py-6 text-xs text-muted-foreground">
          <div className="flex items-center gap-2.5">
            <div className="grid h-6 w-6 place-items-center rounded-md bg-primary/15 text-primary">
              <Activity className="h-3.5 w-3.5" />
            </div>
            <span className="font-semibold text-foreground">Rossmann Sales ML Learning Lab</span>
            <span className="hidden sm:inline">· Interactive Retail Forecasting Workbench</span>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="rounded-full border border-border bg-card px-2.5 py-0.5">OLS Linear Regression</span>
            <span className="rounded-full border border-border bg-card px-2.5 py-0.5">K-Nearest Neighbors (k={k})</span>
            <span className="rounded-full border border-border bg-card px-2.5 py-0.5">5-Table Relational Schema</span>
          </div>
        </footer>
      </div>
    </main>
  );
}

function SliderField({
  label,
  value,
  unit,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs">
        <label className="font-medium text-foreground">{label}</label>
        <span className="tabular-nums text-muted-foreground">{value}{unit}</span>
      </div>
      <Slider
        aria-label={label}
        value={[value]}
        min={min}
        max={max}
        step={1}
        onValueChange={(val) => onChange(val[0] ?? min)}
      />
    </div>
  );
}

function ChartPanel({
  title,
  subtitle,
  data,
  dataKey,
  xKey,
  color,
  formatter,
  secondaryKey,
  secondaryColor,
}: {
  title: string;
  subtitle?: string;
  data: ChartDatum[];
  dataKey: string;
  xKey: string;
  color: string;
  formatter: (value: number) => string;
  secondaryKey?: string;
  secondaryColor?: string;
}) {
  const gradId = `grad-${dataKey}`;
  return (
    <Panel title={title} subtitle={subtitle}>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {secondaryKey ? (
            <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 5, left: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey={xKey} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
              <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} tickFormatter={(val) => numberFormat(Number(val))} />
              <Tooltip {...tooltipStyle} formatter={(val, name) => [formatter(Number(val)), name]} />
              <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
              <Bar dataKey={dataKey} name="Avg sales" fill={color} radius={[3, 3, 0, 0]} />
              <Bar dataKey={secondaryKey} name="Avg customers" fill={secondaryColor ?? "var(--chart-3)"} radius={[3, 3, 0, 0]} />
            </ComposedChart>
          ) : (
            <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 5, left: 0 }}>
              <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey={xKey} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
              <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} tickFormatter={(val) => numberFormat(Number(val))} />
              <Tooltip {...tooltipStyle} formatter={(val, name) => [formatter(Number(val)), name]} />
              <Area
                type="monotone"
                dataKey={dataKey}
                name={dataKey === "sales" ? "Avg sales" : "Test RMSE"}
                stroke={color}
                strokeWidth={2}
                fill={`url(#${gradId})`}
                dot={{ fill: color, r: 3, strokeWidth: 0 }}
                activeDot={{ r: 5, strokeWidth: 0 }}
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

function HistogramPanel({
  title,
  bins,
  mean,
  sd,
  color,
}: {
  title: string;
  bins: { label: string; count: number }[];
  mean: number;
  sd: number;
  color: string;
}) {
  return (
    <Panel title={title} subtitle={`Mean ${numberFormat(mean)} · standard deviation ${numberFormat(sd)}`}>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis dataKey="label" tick={{ fill: "var(--muted-foreground)", fontSize: 9 }} />
            <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
            <Tooltip {...tooltipStyle} />
            <Bar dataKey="count" name="Records" fill={color} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <div>
        <p className="text-sm font-medium text-foreground">{label}</p>
        {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
      </div>
      <Switch checked={checked} onCheckedChange={onChange} aria-label={label} />
    </div>
  );
}

function PreviewTable({ rows }: { rows: RossmannRow[] }) {
  if (!rows.length) return <p className="py-8 text-center text-sm text-muted-foreground">No rows available.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[700px] text-left text-xs">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            {Object.keys(rows[0] ?? {}).slice(0, 10).map((column) => (
              <th key={column} className="px-3 py-2.5 font-medium">{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 10).map((row, index) => (
            <tr key={`${row.Store}-${row.Date}-${index}`} className="border-b border-border/60 last:border-0">
              {Object.values(row).slice(0, 10).map((value, columnIndex) => (
                <td key={columnIndex} className="px-3 py-2 tabular-nums">
                  {value === null ? "—" : typeof value === "number" ? numberFormat(value) : String(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PredictionScatter({
  title,
  actual,
  predicted,
  color,
}: {
  title: string;
  actual: number[];
  predicted: number[];
  color: string;
}) {
  const data = actual.map((value, index) => ({ actual: value, predicted: predicted[index] ?? 0 }));
  const max = Math.max(...actual, ...predicted, 1);
  return (
    <Panel title={title} subtitle="Predicted against observed sales · dashed diagonal is perfect fit.">
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis type="number" dataKey="actual" name="Actual" domain={[0, max]} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
            <YAxis type="number" dataKey="predicted" name="Predicted" domain={[0, max]} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
            <Tooltip {...tooltipStyle} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
            <Scatter data={data.slice(0, 800)} fill={color} fillOpacity={0.55} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

/** Horizontal bar chart of linear regression coefficients. */
function FeatureImportanceChart({
  coefficients,
}: {
  coefficients: { feature: string; coefficient: number; abs_importance: number }[];
}) {
  const sorted = [...coefficients]
    .sort((a, b) => b.abs_importance - a.abs_importance)
    .slice(0, 10);
  const maxImp = Math.max(...sorted.map((c) => c.abs_importance), 0.001);
  return (
    <Panel title="Feature Importance" subtitle="Standardized linear regression coefficients — magnitude = predictive weight" accent="primary">
      <div className="space-y-2.5">
        {sorted.map((c) => {
          const pct = Math.round((c.abs_importance / maxImp) * 100);
          const isPos = c.coefficient >= 0;
          return (
            <div key={c.feature} className="grid grid-cols-[160px_1fr_56px] items-center gap-3">
              <span className="truncate text-xs font-medium text-foreground" title={featureLabels[c.feature] ?? c.feature}>
                {featureLabels[c.feature] ?? c.feature}
              </span>
              <div className="relative h-5 overflow-hidden rounded bg-muted">
                <div
                  className={`h-full rounded transition-all duration-700 ease-out ${isPos ? "bg-gradient-to-r from-chart-1/70 to-chart-1" : "bg-gradient-to-r from-chart-5/70 to-chart-5"}`}
                  style={{ width: `${pct}%` }}
                />
                <span className="absolute inset-0 flex items-center px-2 text-[10px] font-semibold text-foreground/80">
                  {isPos ? "+" : ""}
                  {c.coefficient.toFixed(1)}
                </span>
              </div>
              <span className={`text-right text-xs font-bold tabular-nums ${isPos ? "text-chart-1" : "text-chart-5"}`}>
                {pct}%
              </span>
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-[10px] text-muted-foreground">
        Blue = positive correlation with sales · Red = inverse correlation
      </p>
    </Panel>
  );
}

/** Visual gradient bar comparison of Linear vs KNN prediction outputs. */
function PredictionComparison({
  linear,
  knn,
  k,
}: {
  linear: number;
  knn: number;
  k: number;
}) {
  const maxVal = Math.max(linear, knn, 1);
  const deviation = Math.abs(((linear - knn) / maxVal) * 100);
  const agreement = 100 - deviation;
  const winner = linear < knn ? "KNN" : "Linear";

  return (
    <div className="space-y-5">
      {/* Linear prediction bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-chart-1" />
            <span className="font-semibold text-foreground">Linear Regression (OLS)</span>
          </div>
          <span className="tabular-nums font-bold text-chart-1">{money(linear)}</span>
        </div>
        <div className="h-4 overflow-hidden rounded-full bg-muted/60">
          <div
            className="h-full rounded-full bg-gradient-to-r from-chart-1/60 via-chart-1 to-chart-1/80 transition-all duration-700 ease-out"
            style={{ width: `${(linear / maxVal) * 100}%` }}
          />
        </div>
      </div>

      {/* KNN prediction bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-chart-3" />
            <span className="font-semibold text-foreground">KNN Regression (k={k})</span>
          </div>
          <span className="tabular-nums font-bold text-chart-3">{money(knn)}</span>
        </div>
        <div className="h-4 overflow-hidden rounded-full bg-muted/60">
          <div
            className="h-full rounded-full bg-gradient-to-r from-chart-3/60 via-chart-3 to-chart-3/80 transition-all duration-700 ease-out"
            style={{ width: `${(knn / maxVal) * 100}%` }}
          />
        </div>
      </div>

      {/* Agreement indicator */}
      <div className="rounded-xl border border-border bg-muted/30 p-3">
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Model agreement</span>
          <span className={`font-bold tabular-nums ${agreement > 90 ? "text-chart-2" : agreement > 75 ? "text-chart-3" : "text-destructive"}`}>
            {agreement.toFixed(1)}%
          </span>
        </div>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-all duration-500 ${agreement > 90 ? "bg-chart-2" : agreement > 75 ? "bg-chart-3" : "bg-destructive"}`}
            style={{ width: `${agreement}%` }}
          />
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground">
          {deviation.toFixed(1)}% deviation · {winner} predicts {winner === "KNN" ? "higher" : "lower"} turnover
        </p>
      </div>
    </div>
  );
}