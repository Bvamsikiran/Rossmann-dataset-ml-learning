import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import Papa from "papaparse";
import {
  Activity, ArrowDownToLine, BarChart3, Check, ChevronLeft, ChevronRight, CircleHelp,
  Database, FileUp, Gauge, Moon, Play, RotateCcw, Sun, Table2, WandSparkles, Server,
  Sparkles, Layers, Cpu, Compass, Sliders, ExternalLink, RefreshCw, AlertCircle,
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, ReferenceLine,
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
  fetchCorrelations, trainModelsAPI, runKNNSweepAPI, predictLiveAPI,
  type BackendStatus, type DatasetStatusResponse, type FeatureItem,
  type MergedDatasetResponse, type TrainResponse, type KNNSweepResponse,
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
}: {
  title: string;
  subtitle?: string | undefined;
  children: ReactNode;
  className?: string;
  badge?: ReactNode | undefined;
}) {
  return (
    <section className={`min-w-0 rounded-md border border-border bg-card p-4 sm:p-5 ${className}`}>
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
    </section>
  );
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="border-l-2 border-primary pl-3">
      <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular-nums text-foreground">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex min-h-56 flex-col items-center justify-center border border-dashed border-border bg-muted/30 px-5 text-center">
      <Database className="mb-3 h-5 w-5 text-muted-foreground" />
      <p className="text-sm font-medium text-foreground">{title}</p>
      <div className="mt-1 max-w-sm text-xs leading-5 text-muted-foreground">{children}</div>
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

  const prepResult = useMemo(() => preprocessData(rows, preprocess), [rows, preprocess]);
  const effectiveRows = processedRows.length ? processedRows : prepResult.rows;

  const attributes = useMemo(() => {
    return [
      ["Store", "Store identifier (1–1,115)"],
      ["DayOfWeek", "Day of the week (1=Mon..7=Sun)"],
      ["Date", "Trading date of transaction"],
      ["Sales", "Daily turnover in local currency"],
      ["Customers", "Number of customers on that day"],
      ["Open", "Trading status (0=closed, 1=open)"],
      ["Promo", "Store-level promotional coupon running"],
      ["StateHoliday", "State holiday marker (0, a, b, c)"],
      ["SchoolHoliday", "School holiday active in district"],
      ["StoreType", "Store model profile (a, b, c, d)"],
      ["Assortment", "Assortment level (a=basic, b=extra, c=extended)"],
      ["CompetitionDistance", "Distance in meters to nearest competitor"],
      ["Mean_TemperatureC", "Mean daily temperature (°C) for store region"],
      ["Precipitationmm", "Daily precipitation (mm)"],
      ["trend", "Google Search index for Rossmann in federal state"],
    ];
  }, []);

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
    setProgress(10);
    if (backendReady) {
      await handleLoadRealDatasets();
      setProgress(40);
      setStage(1);
      await new Promise((r) => setTimeout(r, 200));
      setProgress(60);
      setStage(3);
      await new Promise((r) => setTimeout(r, 200));
      setProgress(75);
      setStage(4);
      await runTraining();
      setProgress(100);
      setStage(5);
    } else {
      createRows();
      setProgress(50);
      setStage(4);
      runTraining();
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
                <div className="mb-5 space-y-3 rounded-md border border-border/80 bg-muted/40 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-foreground">Relational Sources Detected:</span>
                    <span className="text-[11px] text-muted-foreground">Path: G:\My Drive\...\rossmann</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
                    <div className="rounded border border-border bg-card p-2">
                      <p className="font-semibold text-primary">train.csv</p>
                      <p className="text-[11px] text-muted-foreground">36.3 MB · Sales & Promos</p>
                    </div>
                    <div className="rounded border border-border bg-card p-2">
                      <p className="font-semibold text-chart-2">store.csv</p>
                      <p className="text-[11px] text-muted-foreground">45 KB · 1,115 Stores</p>
                    </div>
                    <div className="rounded border border-border bg-card p-2">
                      <p className="font-semibold text-chart-3">weather.csv</p>
                      <p className="text-[11px] text-muted-foreground">1.45 MB · Daily Weather</p>
                    </div>
                    <div className="rounded border border-border bg-card p-2">
                      <p className="font-semibold text-foreground">googletrend.csv</p>
                      <p className="text-[11px] text-muted-foreground">86 KB · Search Volume</p>
                    </div>
                    <div className="rounded border border-border bg-card p-2">
                      <p className="font-semibold text-muted-foreground">store_states.csv</p>
                      <p className="text-[11px] text-muted-foreground">16 Federal States</p>
                    </div>
                    <div className="rounded border border-border bg-card p-2">
                      <p className="font-semibold text-muted-foreground">state_names.csv</p>
                      <p className="text-[11px] text-muted-foreground">Full State Names</p>
                    </div>
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

              <Panel title="The Retail Story & Motivation" subtitle="Store Turnover Forecasting in Fast-Moving Retail">
                <div className="border-l-2 border-primary pl-4 text-xs leading-6 text-foreground">
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
                <div className="mt-6 grid grid-cols-3 gap-3">
                  <Metric label="Data source" value={rows.length ? "Ready" : "—"} detail={sourceName.split("·")[0] ?? ""} />
                  <Metric label="Loaded rows" value={rows.length ? numberFormat(rows.length) : "0"} detail="Store × date" />
                  <Metric label="ML Models" value="02" detail="Linear · KNN" />
                </div>
              </Panel>
            </div>

            {rows.length > 0 && (
              <div className="mt-6 space-y-5">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Metric label="Total Rows" value={numberFormat(rows.length)} detail="Ready in memory" />
                  <Metric label="Missing values" value={numberFormat(countMissing(rows))} detail="Handled in Stage 3" />
                  <Metric label="Open days" value={`${Math.round((rows.filter((r) => r.Open === 1).length / rows.length) * 100)}%`} detail="Active trading" />
                  <Metric label="Promotional days" value={`${Math.round((rows.filter((r) => r.Promo === 1).length / rows.length) * 100)}%`} detail="With coupon lift" />
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
                          <tr key={`${row.Store}-${row.Date}-${index}`} className="border-b border-border/60 last:border-0">
                            {Object.values(row).slice(0, 10).map((value, columnIndex) => (
                              <td key={`${columnIndex}`} className="px-3 py-2 tabular-nums">
                                {value === null ? (
                                  <span className="text-destructive">null</span>
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
              <Metric label="Sales mean" value={money(analysis.salesStats.mean)} detail={`SD ${money(analysis.salesStats.sd)}`} />
              <Metric label="Customer mean" value={numberFormat(analysis.customerStats.mean)} detail={`SD ${numberFormat(analysis.customerStats.sd)}`} />
              <Metric label="Promo sales lift" value={`${analysis.uplift.toFixed(1)}%`} detail="Turnover boost from promos" />
              <Metric label="Analyzed rows" value={numberFormat(effectiveRows.length)} detail="Active in memory" />
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
            <Panel title="Data Preparation & Hygiene" subtitle="Strict ML best practice: fit scaling and imputation parameters on train fold only.">
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
              <Metric label="Rows before" value={numberFormat(prepResult.summary.beforeRows)} detail="Loaded from source" />
              <Metric label="Rows after" value={numberFormat(prepResult.summary.afterRows)} detail="Ready for modeling" />
              <Metric label="Missing values" value={`${numberFormat(prepResult.summary.beforeMissing)} → 0`} detail="After contextual imputation" />
              <Metric label="Duplicates removed" value={numberFormat(prepResult.summary.duplicatesRemoved)} detail="Identical entries dropped" />
            </div>
            <Panel title="Prepared records" subtitle="Transformed dataset preview ready for feature selection.">
              <PreviewTable rows={prepResult.rows} />
            </Panel>
          </div>
        );

      case 3:
        return (
          <div className="space-y-5">
            {analysis && (
              <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
                <Panel title="Correlation with Sales" subtitle="Pearson linear correlation coefficient across numeric features.">
                  <div className="space-y-3">
                    {analysis.correlations.map(({ feature, value }) => (
                      <div key={feature} className="grid grid-cols-[140px_1fr_54px] items-center gap-3 text-xs">
                        <span className="truncate text-foreground">{featureLabels[feature] ?? feature}</span>
                        <div className="h-2 overflow-hidden bg-muted">
                          <div
                            className={`h-full ${value >= 0 ? "bg-chart-1" : "bg-chart-3"}`}
                            style={{ width: `${Math.abs(value) * 100}%` }}
                          />
                        </div>
                        <span className="text-right tabular-nums text-muted-foreground">{value.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </Panel>
                <Panel title="Correlation Matrix Heatmap" subtitle="Pairwise collinearity across key features.">
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
                                  className="rounded-sm py-2 font-semibold tabular-nums"
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

            <Panel title="Select Features from Merged Family" subtitle="Choose signals across Store, Weather, Search Trends, and Calendar components.">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  "Customers", "Promo", "DayOfWeek", "SchoolHoliday",
                  "StoreType", "Assortment", "CompetitionDistance",
                  "MonthsSinceCompetition", "Promo2Active",
                  "Mean_TemperatureC", "Precipitationmm", "Mean_Humidity",
                  "Max_Wind_SpeedKm_h", "trend", "Month", "WeekOfYear", "IsWeekend"
                ].map((feat) => (
                  <label key={feat} className="flex cursor-pointer items-center gap-3 border border-border px-3 py-3 text-sm transition-colors hover:bg-muted/40">
                    <Checkbox
                      checked={features.includes(feat)}
                      onCheckedChange={(checked) => toggleFeature(feat, checked)}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-foreground">{featureLabels[feat] ?? feat}</p>
                      <p className="text-[10px] text-muted-foreground">{feat}</p>
                    </div>
                  </label>
                ))}
              </div>
              <div className="mt-4 flex items-center justify-between border-t border-border pt-3 text-xs text-muted-foreground">
                <span>{features.length} features selected for model fitting</span>
                <Badge variant="outline">{backendReady ? "Scikit-Learn Normalizer" : "Browser Z-score"}</Badge>
              </div>
            </Panel>
          </div>
        );

      case 4:
        return (
          <div className="space-y-5">
            <Panel title="Model Training Configuration" subtitle="Fit Linear Regression baseline and K-Nearest Neighbors in Scikit-Learn.">
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
                <div className="flex items-center justify-between border border-border px-3 py-2.5">
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
                  className="bg-primary text-primary-foreground hover:bg-primary/90"
                >
                  <WandSparkles className="h-4 w-4" />
                  {training ? "Fitting models in Python backend…" : "Train Linear Regression vs KNN"}
                </Button>
                <span className="text-xs text-muted-foreground">
                  {backendReady ? "Powered by Scikit-Learn & NumPy" : "Client-side calculation"} · {features.length} features selected
                </span>
              </div>
            </Panel>

            {(backendTrainResult || result) && (
              <div className="space-y-5">
                <Panel
                  title="Estimated Linear Regression Equation & Weights"
                  subtitle="Coefficients representing marginal turnover change per standardized feature unit."
                >
                  <div className="break-words rounded bg-muted/30 p-3 font-mono text-xs leading-7 sm:text-sm">
                    Sales = {money(backendTrainResult?.intercept ?? result?.intercept ?? 0)}
                    {(backendTrainResult?.coefficients ?? []).map((c) => (
                      <span key={c.feature}>
                        {" "}{c.coefficient >= 0 ? "+" : "−"} {money(Math.abs(c.coefficient))} × z({featureLabels[c.feature] ?? c.feature})
                      </span>
                    ))}
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {(backendTrainResult?.coefficients ?? []).map((c) => (
                      <Badge variant="secondary" key={c.feature} className="gap-1 text-xs">
                        {featureLabels[c.feature] ?? c.feature}: <span className="font-semibold">{c.coefficient > 0 ? `+${c.coefficient}` : c.coefficient}</span>
                      </Badge>
                    ))}
                  </div>
                </Panel>

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

      case 5:
        return metrics ? (
          <div className="space-y-5">
            <Panel title="Model Performance Evaluation" subtitle="Held-out validation comparison across root mean squared error and variance explained (R²).">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[650px] text-left text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      {["Model Architecture", "Train RMSE", "Test RMSE", "Test MAE", "Test R²", "Generalization Status"].map((l) => (
                        <th key={l} className="px-3 py-3 font-medium">{l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border">
                      <td className="px-3 py-3 font-semibold text-foreground">Linear Regression (OLS)</td>
                      <td className="px-3 py-3 tabular-nums">{money(metrics.linearTrain.rmse)}</td>
                      <td className="px-3 py-3 font-semibold tabular-nums text-primary">{money(metrics.linearTest.rmse)}</td>
                      <td className="px-3 py-3 tabular-nums">{money(metrics.linearTest.mape)}</td>
                      <td className="px-3 py-3 tabular-nums">{metrics.linearTest.r2.toFixed(3)}</td>
                      <td className="px-3 py-3"><Badge variant="outline">Stable Baseline</Badge></td>
                    </tr>
                    <tr>
                      <td className="px-3 py-3 font-semibold text-foreground">K-Nearest Neighbors (KNN, k={k})</td>
                      <td className="px-3 py-3 tabular-nums">{money(metrics.knnTrain.rmse)}</td>
                      <td className="px-3 py-3 font-semibold tabular-nums text-chart-3">{money(metrics.knnTest.rmse)}</td>
                      <td className="px-3 py-3 tabular-nums">{money(metrics.knnTest.mape)}</td>
                      <td className="px-3 py-3 tabular-nums">{metrics.knnTest.r2.toFixed(3)}</td>
                      <td className="px-3 py-3">
                        <Badge variant={metrics.knnTest.rmse < metrics.linearTest.rmse ? "default" : "secondary"}>
                          {metrics.knnTest.rmse < metrics.linearTest.rmse ? "Nonlinear Fit Win" : "Local Neighbor Noise"}
                        </Badge>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Panel>

            <div className="grid gap-5 xl:grid-cols-2">
              <Panel title="Train vs Test RMSE Error Comparison" subtitle="Lower error indicates superior predictive accuracy.">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { model: "Linear", train: metrics.linearTrain.rmse, test: metrics.linearTest.rmse },
                        { model: "KNN", train: metrics.knnTrain.rmse, test: metrics.knnTest.rmse },
                      ]}
                    >
                      <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                      <XAxis dataKey="model" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                      <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
                      <Tooltip {...tooltipStyle} formatter={(val) => money(Number(val))} />
                      <Bar dataKey="train" name="Train RMSE" fill="var(--chart-2)" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="test" name="Test RMSE" fill="var(--chart-1)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              {knnSweep && (
                <ChartPanel
                  title="KNN Bias-Variance Curve (K Sweep)"
                  subtitle="Optimal K minimizes test RMSE while preventing overfitting."
                  data={knnSweep.sweep_results}
                  dataKey="test_rmse"
                  xKey="k"
                  color="var(--chart-3)"
                  formatter={money}
                />
              )}
            </div>

            {backendTrainResult?.cv_results && (
              <Panel title="5-Fold Cross-Validation Summary" subtitle="Evaluated across non-overlapping partitions to test variance stability.">
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
                      {backendTrainResult.cv_results.map((f) => (
                        <tr key={f.fold} className="border-b border-border/70">
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
        ) : (
          <EmptyState title="Models Awaiting Training">
            Execute model fitting in the previous step to evaluate RMSE, cross-validation, and bias-variance tradeoff.
          </EmptyState>
        );

      case 6:
        return (
          <div className="grid gap-5 xl:grid-cols-[0.8fr_1.2fr]">
            <Panel title="Retail Scenario Parameters" subtitle="Adjust real-world operational parameters to forecast store sales turnover.">
              <div className="space-y-4">
                <label className="block text-xs text-muted-foreground">
                  Expected Customer Foot-Traffic
                  <Input
                    className="mt-2"
                    type="number"
                    min={0}
                    value={customersInput}
                    onChange={(e) => setCustomersInput(Number(e.target.value))}
                  />
                </label>
                <label className="block text-xs text-muted-foreground">
                  Day of Week
                  <Select value={String(dayInput)} onValueChange={(v) => setDayInput(Number(v))}>
                    <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map((d, i) => (
                        <SelectItem key={d} value={String(i + 1)}>{d}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </label>
                <label className="block text-xs text-muted-foreground">
                  Nearest Competitor Distance (meters)
                  <Input
                    className="mt-2"
                    type="number"
                    min={0}
                    value={compDistanceInput}
                    onChange={(e) => setCompDistanceInput(Number(e.target.value))}
                  />
                </label>
                <label className="block text-xs text-muted-foreground">
                  Regional Temperature (°C)
                  <Input
                    className="mt-2"
                    type="number"
                    value={tempInput}
                    onChange={(e) => setTempInput(Number(e.target.value))}
                  />
                </label>
                <ToggleRow label="Promotional Campaign Active" description="Special coupon flyer distribution" checked={promoInput} onChange={setPromoInput} />
                <ToggleRow label="School Holiday Period" description="State public school vacation" checked={schoolInput} onChange={setSchoolInput} />
              </div>
            </Panel>

            <Panel title="Predicted Turnover & Inventory Advisory" subtitle="Model estimates for the specified store conditions.">
              <div className="grid gap-5 sm:grid-cols-2">
                <div className="border-l-2 border-chart-1 pl-4">
                  <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">Linear Regression Model</p>
                  <p className="mt-2 text-2xl font-semibold tabular-nums text-foreground">
                    {money(livePredictionData?.linear ?? 6250)}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">Parametric baseline estimate</p>
                </div>
                <div className="border-l-2 border-chart-3 pl-4">
                  <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">KNN Regression (k={k})</p>
                  <p className="mt-2 text-2xl font-semibold tabular-nums text-foreground">
                    {money(livePredictionData?.knn ?? 6480)}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">Instance-based local neighborhood</p>
                </div>
              </div>

              <div className="mt-8 border-t border-border pt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-primary">Store Manager Operational Guidance</p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  For this scenario, plan staffing and merchandise stocking for approximately{" "}
                  <strong>{money(Math.max(livePredictionData?.linear ?? 6250, livePredictionData?.knn ?? 6480))}</strong> in sales turnover.
                  {promoInput && " Promotional campaigns typically generate ~30% higher morning foot-traffic."}
                </p>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                <Badge variant="secondary">{numberFormat(customersInput)} customers</Badge>
                <Badge variant="secondary">{promoInput ? "Promo active" : "No promo"}</Badge>
                <Badge variant="secondary">Day {dayInput}</Badge>
                <Badge variant="secondary">{compDistanceInput}m to competitor</Badge>
                <Badge variant="secondary">{tempInput}°C</Badge>
              </div>
            </Panel>
          </div>
        );

      default:
        return null;
    }
  }

  return (
    <main className={dark ? "dark min-h-screen bg-background text-foreground" : "min-h-screen bg-background text-foreground"}>
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center bg-primary text-primary-foreground rounded">
              <Activity className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">Rossmann Sales ML Learning Lab</p>
              <p className="text-[11px] text-muted-foreground">Linear regression vs KNN regression · Multi-dataset pipeline</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={backendReady ? "border-chart-2/40 bg-chart-2/10 text-chart-2" : "border-border"}>
              {backendReady ? "FASTAPI BACKEND ACTIVE" : "LOCAL CLIENT MODE"}
            </Badge>
            <Button
              aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
              size="icon"
              variant="ghost"
              onClick={() => setDark((v) => !v)}
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </header>

      {/* Top Quick-Jump Storyflow Navigation */}
      <div className="sticky top-0 z-20 border-b border-border bg-background/95 backdrop-blur">
        <div className="mx-auto max-w-[1440px] px-4 sm:px-8">
          <nav aria-label="Storyflow pipeline" className="flex overflow-x-auto">
            {stages.map(({ label, short, icon: Icon }, index) => (
              <button
                key={label}
                aria-current={stage === index ? "step" : undefined}
                onClick={() => setStage(index)}
                className={`group flex min-w-[96px] flex-1 items-center gap-2 border-b-2 px-2 py-3 text-left transition-colors sm:min-w-0 sm:px-3 ${
                  stage === index
                    ? "border-primary text-primary"
                    : index < stage
                      ? "border-chart-2 text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                <span
                  className={`grid h-6 w-6 shrink-0 place-items-center rounded text-[10px] font-semibold ${
                    stage === index
                      ? "bg-primary text-primary-foreground"
                      : index < stage
                        ? "bg-chart-2/15 text-chart-2"
                        : "bg-muted text-muted-foreground"
                  }`}
                >
                  {index < stage ? <Check className="h-3.5 w-3.5" /> : <Icon className="h-3.5 w-3.5" />}
                </span>
                <span className="hidden truncate text-xs font-medium md:inline">{label}</span>
                <span className="text-xs font-medium md:hidden">{short}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Stage Header Banner with Story Context */}
      <div className="border-b border-border bg-muted/20">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-4 py-5 sm:px-8">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">
              Step {stage + 1} of {stages.length} · {stages[stage]?.label}
            </p>
            <h1 className="mt-1 text-xl font-semibold sm:text-2xl">{stages[stage]?.label}</h1>
            <p className="mt-1 max-w-2xl text-xs text-muted-foreground">{stages[stage]?.story}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" disabled={fullRun} onClick={runFullSimulation}>
              <Play className="h-4 w-4" />
              {fullRun ? "Simulating Pipeline…" : "Run Full Simulation"}
            </Button>
            {rows.length > 0 && (
              <Button variant="ghost" onClick={createRows}>
                <RotateCcw className="h-4 w-4" />
                Reset Data
              </Button>
            )}
          </div>
          {fullRun && (
            <div className="w-full">
              <div className="mb-1.5 flex justify-between text-xs text-muted-foreground">
                <span>Executing multi-table ingestion, feature extraction, and ML fitting</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          )}
        </div>
      </div>

      <div className="mx-auto max-w-[1440px] px-4 py-6 sm:px-8 sm:py-8">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              {rows.length ? `${numberFormat(rows.length)} active observations` : "Waiting for dataset"}{" "}
              {processedRows.length > 0 ? `· ${numberFormat(processedRows.length)} prepared` : ""}
            </p>
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

        <footer className="mt-12 flex flex-wrap items-center justify-between gap-3 border-t border-border py-5 text-[11px] text-muted-foreground">
          <span>Rossmann Sales ML Learning Lab · Research Simulator</span>
          <span>Linear Regression vs KNN · Multi-Table Relational Pipeline</span>
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
  return (
    <Panel title={title} subtitle={subtitle}>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 5, left: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
            <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} tickFormatter={(val) => numberFormat(Number(val))} />
            <Tooltip {...tooltipStyle} formatter={(val, name) => [formatter(Number(val)), name]} />
            <Bar dataKey={dataKey} name={dataKey === "sales" ? "Avg sales" : "Avg customers"} fill={color} radius={[3, 3, 0, 0]} />
            {secondaryKey && (
              <Bar dataKey={secondaryKey} name="Avg customers" fill={secondaryColor ?? "var(--chart-3)"} radius={[3, 3, 0, 0]} />
            )}
          </ComposedChart>
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