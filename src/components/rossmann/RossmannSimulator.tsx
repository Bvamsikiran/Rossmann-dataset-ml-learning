import { useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import Papa from "papaparse";
import {
  Activity, ArrowDownToLine, BarChart3, Check, ChevronLeft, ChevronRight, CircleHelp,
  Database, FileUp, Gauge, Moon, Play, RotateCcw, Sun, Table2, WandSparkles,
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

const stages = [
  { label: "Data collection", short: "Data", icon: Database },
  { label: "Exploratory analysis", short: "Explore", icon: BarChart3 },
  { label: "Preprocessing", short: "Prepare", icon: Activity },
  { label: "Feature selection", short: "Features", icon: Gauge },
  { label: "Model training", short: "Training", icon: WandSparkles },
  { label: "Evaluation", short: "Evaluate", icon: Check },
  { label: "Live prediction", short: "Predict", icon: Play },
];
const featureLabels: Record<string, string> = {
  Customers: "Customers", Promo: "Promotion", DayOfWeek: "Day of week",
  SchoolHoliday: "School holiday", Store: "Store number", Open: "Open status",
};
const money = (value: number) => `${Math.round(value).toLocaleString("en-US")} kr`;
const numberFormat = (value: number) => Math.round(value).toLocaleString("en-US");
const tooltipStyle = { contentStyle: { border: "1px solid var(--border)", borderRadius: 6, background: "var(--popover)", color: "var(--popover-foreground)" } };

type ChartDatum = Record<string, number | string>;

function Panel({ title, subtitle, children, className = "" }: { title: string; subtitle?: string; children: ReactNode; className?: string }) {
  return <section className={`min-w-0 rounded-md border border-border bg-card p-4 sm:p-5 ${className}`}>
    <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
      <div><h3 className="text-sm font-semibold text-foreground">{title}</h3>{subtitle && <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>}</div>
    </div>{children}
  </section>;
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="border-l-2 border-primary pl-3"><p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">{label}</p><p className="mt-1 text-xl font-semibold tabular-nums text-foreground">{value}</p><p className="mt-1 text-xs text-muted-foreground">{detail}</p></div>;
}

function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return <div className="flex min-h-56 flex-col items-center justify-center border border-dashed border-border bg-muted/30 px-5 text-center"><Database className="mb-3 h-5 w-5 text-muted-foreground" /><p className="text-sm font-medium text-foreground">{title}</p><div className="mt-1 max-w-sm text-xs leading-5 text-muted-foreground">{children}</div></div>;
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
  const [preprocess, setPreprocess] = useState<PreprocessOptions>({ removeDuplicates: true, dropClosed: true, imputation: "median", clipOutliers: false });
  const [features, setFeatures] = useState<(keyof RossmannRow)[]>(["Customers", "Promo", "DayOfWeek", "SchoolHoliday"]);
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
  const fileInput = useRef<HTMLInputElement>(null);

  const prepResult = useMemo(() => preprocessData(rows, preprocess), [rows, preprocess]);
  const effectiveRows = processedRows.length ? processedRows : prepResult.rows;
  const attributes = useMemo(() => {
    const descriptors = [
      ["Store", "Store identifier"], ["DayOfWeek", "Day of the week (1–7)"], ["Date", "Trading date"],
      ["Sales", "Daily turnover in local currency"], ["Customers", "Number of customers"], ["Open", "Trading status (0/1)"],
      ["Promo", "Promotion active (0/1)"], ["StateHoliday", "State holiday marker"], ["SchoolHoliday", "School holiday (0/1)"],
    ];
    return descriptors;
  }, []);

  const analysis = useMemo(() => {
    if (!effectiveRows.length) return null;
    const rowsWithSales = effectiveRows.filter((row) => typeof row.Sales === "number");
    const grouped = (key: "year" | "weekday" | "quarter" | "promo") => {
      const buckets = new Map<string, { sales: number; customers: number; count: number }>();
      for (const row of rowsWithSales) {
        const date = new Date(`${row.Date}T00:00:00Z`);
        const label = key === "year" ? String(Number.isNaN(date.getTime()) ? "Unknown" : date.getUTCFullYear())
          : key === "weekday" ? `${row.DayOfWeek ?? 0}` : key === "quarter" ? `Q${Math.floor(date.getUTCMonth() / 3) + 1}` : row.Promo ? "Promo" : "No promo";
        const bucket = buckets.get(label) ?? { sales: 0, customers: 0, count: 0 };
        bucket.sales += row.Sales ?? 0; bucket.customers += row.Customers ?? 0; bucket.count += 1; buckets.set(label, bucket);
      }
      return Array.from(buckets, ([label, values]) => ({ label, sales: values.sales / values.count, customers: values.customers / values.count, rows: values.count }));
    };
    const yearly = grouped("year"); const weekdays = grouped("weekday").sort((a, b) => Number(a.label) - Number(b.label));
    const quarters = grouped("quarter"); const promo = grouped("promo");
    const scatter = rowsWithSales.filter((row) => typeof row.Customers === "number").slice(0, 1400).map((row) => ({ customers: row.Customers ?? 0, sales: row.Sales ?? 0, year: row.Date.slice(0, 4) }));
    const salesValues = rowsWithSales.map((row) => row.Sales ?? 0); const customerValues = rowsWithSales.map((row) => row.Customers ?? 0);
    const stats = (values: number[]) => {
      const mean = values.reduce((sum, value) => sum + value, 0) / (values.length || 1);
      return { mean, sd: Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length || 1)) };
    };
    const histogram = (values: number[], count = 12) => {
      if (!values.length) return [];
      const max = Math.max(...values); const min = Math.min(...values); const range = max - min || 1;
      const bins = Array.from({ length: count }, (_, index) => ({ label: numberFormat(min + range * index / count), count: 0 }));
      for (const value of values) { const bin = Math.min(count - 1, Math.floor((value - min) / range * count)); bins[bin]!.count += 1; }
      return bins;
    };
    const promoSales = promo.find((item) => item.label === "Promo")?.sales ?? 0;
    const noPromoSales = promo.find((item) => item.label === "No promo")?.sales ?? 0;
    const promoCustomers = promo.find((item) => item.label === "Promo")?.customers ?? 0;
    const noPromoCustomers = promo.find((item) => item.label === "No promo")?.customers ?? 0;
    const uplift = noPromoSales ? (promoSales / noPromoSales - 1) * 100 : 0;
    const correlations = selectableFeatures.map((feature) => ({
      feature, value: correlation(effectiveRows.map((row) => row[feature] ?? 0), effectiveRows.map((row) => row.Sales ?? 0)),
    })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const heatmap = numericColumns.map((x) => ({ feature: x, values: numericColumns.map((y) => ({ feature: y, value: correlation(effectiveRows.map((row) => row[x] ?? 0), effectiveRows.map((row) => row[y] ?? 0)) })) }));
    return {
      yearly, weekdays, quarters, promo, scatter, correlations, heatmap,
      salesStats: stats(salesValues), customerStats: stats(customerValues),
      salesHist: histogram(salesValues), customerHist: histogram(customerValues),
      promoSales, noPromoSales, promoCustomers, noPromoCustomers, uplift,
    };
  }, [effectiveRows]);

  const evaluateRows = processedRows.length ? processedRows : prepResult.rows;
  const trainingRows = effectiveRows.length ? effectiveRows : evaluateRows;
  const metrics = useMemo(() => {
    if (!result) return null;
    return {
      linearTrain: getMetrics(result.trainActual, result.trainLinear),
      linearTest: getMetrics(result.actual, result.linear),
      knnTrain: getMetrics(result.trainActual, result.trainKnn),
      knnTest: getMetrics(result.actual, result.knn),
    };
  }, [result]);

  function createRows() {
    const generated = generateSyntheticData({ stores, days, noise, missingPercent, outlierPercent });
    setRows(generated); setProcessedRows([]); setResult(null); setSourceName("Synthetic dataset");
    setStage(0);
  }

  function uploadCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    Papa.parse<Record<string, unknown>>(file, {
      header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: ({ data }) => {
        const parsed = data.slice(0, 20_000).map(normalizeUploadedRow);
        setRows(parsed); setProcessedRows([]); setResult(null); setSourceName(`${file.name} · ${numberFormat(parsed.length)} rows`); setStage(0);
      },
    });
    event.target.value = "";
  }

  function runTraining(data = effectiveRows) {
    setTraining(true);
    try {
      const nextResult = trainAndEvaluate(data, { features, splitPercent, k, metric, weighted });
      setResult(nextResult);
      return nextResult;
    } catch (error) {
      console.error(error);
      return null;
    } finally {
      setTraining(false);
    }
  }

  async function runFullSimulation() {
    setFullRun(true); setProgress(8);
    const generated = generateSyntheticData({ stores, days, noise, missingPercent, outlierPercent });
    setRows(generated); setProcessedRows([]); setResult(null); setSourceName("Synthetic dataset");
    await new Promise((resolve) => window.setTimeout(resolve, 140)); setProgress(22); setStage(0);
    const transformed = preprocessData(generated, preprocess).rows;
    setProcessedRows(transformed); setStage(1);
    await new Promise((resolve) => window.setTimeout(resolve, 180)); setProgress(43); setStage(2);
    await new Promise((resolve) => window.setTimeout(resolve, 140)); setProgress(58); setStage(3);
    setStage(4); setProgress(68);
    try {
      const next = trainAndEvaluate(transformed, { features, splitPercent, k, metric, weighted });
      setResult(next); setProgress(94);
    } catch (error) { console.error(error); }
    await new Promise((resolve) => window.setTimeout(resolve, 180)); setProgress(100); setStage(5); setFullRun(false);
  }

  function updatePreprocess(key: keyof PreprocessOptions, value: boolean | "mean" | "median") {
    const next = { ...preprocess, [key]: value } as PreprocessOptions;
    setPreprocess(next);
    setProcessedRows(preprocessData(rows, next).rows);
    setResult(null);
  }

  function downloadResults() {
    if (!metrics || !result) return;
    const lines = [
      ["Model", "Train RMSE", "Test RMSE", "Train MAPE", "Test MAPE", "Test R2"],
      ["Linear Regression", metrics.linearTrain.rmse, metrics.linearTest.rmse, metrics.linearTrain.mape, metrics.linearTest.mape, metrics.linearTest.r2],
      ["KNN Regression", metrics.knnTrain.rmse, metrics.knnTest.rmse, metrics.knnTrain.mape, metrics.knnTest.mape, metrics.knnTest.r2],
    ];
    const csv = lines.map((line) => line.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "rossmann-model-results.csv"; link.click(); URL.revokeObjectURL(link.href);
  }

  function toggleFeature(feature: keyof RossmannRow, checked: boolean | "indeterminate") {
    setFeatures((current) => checked === true ? [...current, feature] : current.filter((item) => item !== feature));
    setResult(null);
  }

  function renderStage() {
    if (!rows.length && stage > 0) return <EmptyState title="Start with a Rossmann dataset">Generate synthetic records or upload a Rossmann train.csv file to begin this workflow.</EmptyState>;
    switch (stage) {
      case 0: return <>
        <div className="grid gap-6 xl:grid-cols-[0.72fr_1.28fr]">
          <Panel title="Create a working dataset" subtitle="Choose the sample scale and data-quality conditions.">
            <div className="grid grid-cols-2 gap-4">
              <label className="space-y-2 text-xs text-muted-foreground">Stores <Input aria-label="Number of stores" type="number" min={1} max={200} value={stores} onChange={(event) => setStores(Number(event.target.value))} /></label>
              <label className="space-y-2 text-xs text-muted-foreground">Days per store <Input aria-label="Days per store" type="number" min={7} max={730} value={days} onChange={(event) => setDays(Number(event.target.value))} /></label>
            </div>
            <div className="mt-5 space-y-5">
              <SliderField label="Noise level" value={noise} unit="%" min={0} max={60} onChange={setNoise} />
              <SliderField label="Missing values" value={missingPercent} unit="%" min={0} max={20} onChange={setMissingPercent} />
              <SliderField label="Outliers" value={outlierPercent} unit="%" min={0} max={15} onChange={setOutlierPercent} />
            </div>
            <div className="mt-6 flex flex-wrap gap-2">
              <Button onClick={createRows}><WandSparkles />Generate Synthetic Rossmann Data</Button>
              <Button variant="outline" onClick={() => fileInput.current?.click()}><FileUp />Upload train.csv</Button>
              <input ref={fileInput} className="hidden" type="file" accept=".csv,text/csv" onChange={uploadCsv} />
            </div>
            <p className="mt-3 text-xs text-muted-foreground">CSV uploads are sampled at up to 20,000 records for smooth analysis.</p>
          </Panel>
          <Panel title="About the research" subtitle="Sales Prediction using Linear and KNN Regression">
            <div className="border-l-2 border-accent pl-4"><p className="text-sm leading-6 text-foreground">Compare an interpretable linear baseline with a distance-based K-nearest neighbors model on Rossmann-style store sales. Follow the same data-to-evaluation path and assess results with RMSE, MAPE and R².</p></div>
            <div className="mt-7 grid grid-cols-3 gap-4">
              <Metric label="Data source" value={rows.length ? "Ready" : "—"} detail={sourceName} />
              <Metric label="Observations" value={rows.length ? numberFormat(rows.length) : "0"} detail="Store × trading day" />
              <Metric label="Models" value="02" detail="Linear · KNN" />
            </div>
            <div className="mt-7 flex items-center gap-2 border-t border-border pt-4 text-xs text-muted-foreground"><CircleHelp className="h-4 w-4 shrink-0" />Sales rises with customer traffic and promotion activity; closed days remain at zero sales.</div>
          </Panel>
        </div>
        {rows.length > 0 && <div className="mt-6 space-y-5">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4"><Metric label="Rows" value={numberFormat(rows.length)} detail="Across all stores" /><Metric label="Missing values" value={numberFormat(countMissing(rows))} detail="Introduced in sample" /><Metric label="Open days" value={`${Math.round(rows.filter((row) => row.Open === 1).length / rows.length * 100)}%`} detail="Of observations" /><Metric label="Promotional days" value={`${Math.round(rows.filter((row) => row.Promo === 1).length / rows.length * 100)}%`} detail="Of observations" /></div>
          <Panel title="Data preview" subtitle={`Showing the first ${Math.min(rows.length, 20)} records · ${sourceName}`}>
            <div className="overflow-x-auto"><table className="w-full min-w-[820px] text-left text-xs"><thead><tr className="border-b border-border text-muted-foreground">{Object.keys(rows[0] ?? {}).map((column) => <th key={column} className="px-3 py-2.5 font-medium">{column}</th>)}</tr></thead><tbody>{rows.slice(0, 20).map((row, index) => <tr key={`${row.Store}-${row.Date}-${index}`} className="border-b border-border/60 last:border-0">{Object.values(row).map((value, columnIndex) => <td key={`${columnIndex}`} className="px-3 py-2 tabular-nums">{value === null ? <span className="text-destructive">missing</span> : typeof value === "number" ? numberFormat(value) : value}</td>)}</tr>)}</tbody></table></div>
          </Panel>
          <Panel title="Column guide" subtitle="Fields available in the Rossmann store-sales frame."><div className="grid gap-x-8 sm:grid-cols-2 lg:grid-cols-3">{attributes.map(([name, description]) => <div key={name} className="flex justify-between gap-3 border-b border-border/60 py-2.5 text-xs"><span className="font-medium text-foreground">{name}</span><span className="text-right text-muted-foreground">{description}</span></div>)}</div></Panel>
        </div>}
      </>;
      case 1: return analysis ? <div className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><Metric label="Sales mean" value={money(analysis.salesStats.mean)} detail={`SD ${money(analysis.salesStats.sd)}`} /><Metric label="Customer mean" value={numberFormat(analysis.customerStats.mean)} detail={`SD ${numberFormat(analysis.customerStats.sd)}`} /><Metric label="Promo sales lift" value={`${analysis.uplift.toFixed(1)}%`} detail="Average vs non-promo" /><Metric label="Analyzed rows" value={numberFormat(effectiveRows.length)} detail="After active filters" /></div>
        <div className="grid gap-5 xl:grid-cols-2">
          <ChartPanel title="Average sales by year" subtitle="Year-over-year daily average" data={analysis.yearly} dataKey="sales" xKey="label" color="var(--chart-1)" formatter={money} />
          <ChartPanel title="Sales by day of week" subtitle="Mean daily sales · day 1 is Monday" data={analysis.weekdays} dataKey="sales" xKey="label" color="var(--chart-3)" formatter={money} />
          <Panel title="Sales vs customers" subtitle="Each mark is a store-day observation."><div className="h-64"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 8, right: 10, bottom: 8, left: 6 }}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis dataKey="customers" type="number" name="Customers" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><YAxis dataKey="sales" type="number" name="Sales" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><Tooltip {...tooltipStyle} cursor={{ strokeDasharray: "3 3" }} /><Scatter name="Store days" data={analysis.scatter} fill="var(--chart-2)" fillOpacity={0.54} /></ScatterChart></ResponsiveContainer></div></Panel>
          <ChartPanel title="Promo comparison" subtitle="Average sales · promo days lift traffic and turnover" data={analysis.promo} dataKey="sales" secondaryKey="customers" xKey="label" color="var(--chart-1)" secondaryColor="var(--chart-3)" formatter={money} />
          <ChartPanel title="Average customers by quarter" data={analysis.quarters} dataKey="customers" xKey="label" color="var(--chart-2)" formatter={numberFormat} />
          <HistogramPanel title="Sales distribution" bins={analysis.salesHist} mean={analysis.salesStats.mean} sd={analysis.salesStats.sd} color="var(--chart-1)" />
          <HistogramPanel title="Customer distribution" bins={analysis.customerHist} mean={analysis.customerStats.mean} sd={analysis.customerStats.sd} color="var(--chart-2)" />
        </div>
        <Panel title="What the data says"><ul className="grid gap-3 text-sm text-foreground sm:grid-cols-3"><li className="border-l-2 border-primary pl-3">Promo days average <strong>{analysis.uplift.toFixed(1)}%</strong> higher sales than non-promo days.</li><li className="border-l-2 border-accent pl-3">Promotional periods average <strong>{numberFormat(analysis.promoCustomers)}</strong> customers versus <strong>{numberFormat(analysis.noPromoCustomers)}</strong> without promotions.</li><li className="border-l-2 border-chart-2 pl-3">Customer counts and sales are shown together for <strong>{numberFormat(analysis.scatter.length)}</strong> sampled store-days.</li></ul></Panel>
      </div> : null;
      case 2: return <div className="space-y-5">
        <Panel title="Data preparation" subtitle="Each selected operation is applied to the records used downstream.">
          <div className="divide-y divide-border">
            <ToggleRow label="Remove duplicate records" description="Keep only the first identical record." checked={preprocess.removeDuplicates} onChange={(checked) => updatePreprocess("removeDuplicates", checked)} />
            <ToggleRow label="Drop closed days" description="Exclude records where Open equals zero." checked={preprocess.dropClosed} onChange={(checked) => updatePreprocess("dropClosed", checked)} />
            <ToggleRow label="Clip sales and customer outliers" description="Clamp values to the 1.5 × IQR range." checked={preprocess.clipOutliers} onChange={(checked) => updatePreprocess("clipOutliers", checked)} />
          </div>
          <div className="mt-5 grid items-center gap-3 border-t border-border pt-4 sm:grid-cols-[1fr_200px]"><div><p className="text-sm font-medium">Missing-value imputation</p><p className="mt-1 text-xs text-muted-foreground">Fill empty numeric cells from the selected column statistic.</p></div><Select value={preprocess.imputation} onValueChange={(value: "mean" | "median") => updatePreprocess("imputation", value)}><SelectTrigger aria-label="Imputation method"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="mean">Column mean</SelectItem><SelectItem value="median">Column median</SelectItem></SelectContent></Select></div>
        </Panel>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><Metric label="Rows before" value={numberFormat(prepResult.summary.beforeRows)} detail="Imported / generated" /><Metric label="Rows after" value={numberFormat(prepResult.summary.afterRows)} detail="Ready for analysis" /><Metric label="Missing values" value={`${numberFormat(prepResult.summary.beforeMissing)} → ${numberFormat(prepResult.summary.afterMissing)}`} detail="Before → after imputation" /><Metric label="Duplicates removed" value={numberFormat(prepResult.summary.duplicatesRemoved)} detail={preprocess.removeDuplicates ? "Exact matches" : "Removal disabled"} /></div>
        <Panel title="Prepared records" subtitle="Preview of the transformed data."><PreviewTable rows={prepResult.rows} /></Panel>
      </div>;
      case 3: return analysis ? <div className="space-y-5">
        <div className="grid gap-5 xl:grid-cols-[1.05fr_0.95fr]">
          <Panel title="Correlation with sales" subtitle="Pearson correlation across numeric features."><div className="space-y-3">{analysis.correlations.map(({ feature, value }) => <div key={feature} className="grid grid-cols-[112px_1fr_54px] items-center gap-3 text-xs"><span className="truncate text-foreground">{featureLabels[feature] ?? feature}</span><div className="h-2 overflow-hidden bg-muted"><div className={`h-full ${value >= 0 ? "bg-chart-1" : "bg-chart-3"}`} style={{ width: `${Math.abs(value) * 100}%` }} /></div><span className="text-right tabular-nums text-muted-foreground">{value.toFixed(3)}</span></div>)}</div></Panel>
          <Panel title="Correlation matrix" subtitle="Pairwise numeric feature relationships."><div className="overflow-x-auto"><table className="w-full min-w-[460px] text-center text-[10px]"><thead><tr><th className="p-1.5" />{numericColumns.map((column) => <th key={column} className="p-1.5 font-medium text-muted-foreground">{column}</th>)}</tr></thead><tbody>{analysis.heatmap.map((row) => <tr key={row.feature}><th className="max-w-24 truncate p-1.5 text-left font-medium text-muted-foreground">{row.feature}</th>{row.values.map(({ feature, value }) => <td key={feature} className="p-1"><div className="rounded-sm py-2 font-semibold tabular-nums" style={{ backgroundColor: `color-mix(in oklab, ${value >= 0 ? "var(--chart-1)" : "var(--chart-3)"} ${Math.max(10, Math.abs(value) * 75)}%, var(--muted))`, color: "var(--foreground)" }}>{value.toFixed(2)}</div></td>)}</tr>)}</tbody></table></div></Panel>
        </div>
        <Panel title="Features for both models" subtitle="Select the numeric inputs used by linear and KNN regression. KNN inputs are standardized with z-scores.">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{selectableFeatures.map((feature) => <label key={feature} className="flex cursor-pointer items-center gap-3 border border-border px-3 py-3 text-sm"><Checkbox checked={features.includes(feature)} onCheckedChange={(checked) => toggleFeature(feature, checked)} /><span className="font-medium">{featureLabels[feature] ?? feature}</span><span className="ml-auto text-xs tabular-nums text-muted-foreground">{(analysis.correlations.find((item) => item.feature === feature)?.value ?? 0).toFixed(2)}</span></label>)}</div>
          <div className="mt-4 text-xs text-muted-foreground">{features.length} features selected · normalizing KNN input values</div>
        </Panel>
      </div> : null;
      case 4: return <div className="space-y-5">
        <Panel title="Training configuration" subtitle="Fit the linear baseline and KNN model from scratch on your selected features.">
          <div className="grid gap-6 xl:grid-cols-2">
            <div><SliderField label="Training split" value={splitPercent} unit="%" min={60} max={90} onChange={(value) => { setSplitPercent(value); setResult(null); }} /><p className="mt-2 text-xs text-muted-foreground">{splitPercent}% training · {100 - splitPercent}% test</p></div>
            <div><SliderField label="K nearest neighbors" value={k} unit="" min={1} max={50} onChange={(value) => { setK(value); setResult(null); }} /><p className="mt-2 text-xs text-muted-foreground">{weighted ? "Inverse-distance weighted" : "Simple neighbor average"}</p></div>
            <div><p className="mb-2 text-xs font-medium text-muted-foreground">Distance metric</p><Select value={metric} onValueChange={(value: ModelOptions["metric"]) => { setMetric(value); setResult(null); }}><SelectTrigger aria-label="Distance metric"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="euclidean">Euclidean</SelectItem><SelectItem value="manhattan">Manhattan</SelectItem><SelectItem value="minkowski">Minkowski (p = 3)</SelectItem></SelectContent></Select></div>
            <div className="flex items-center justify-between border border-border px-3 py-2.5"><div><p className="text-sm font-medium">Distance weighting</p><p className="text-xs text-muted-foreground">Closer neighbors influence more.</p></div><Switch checked={weighted} onCheckedChange={(checked) => { setWeighted(checked); setResult(null); }} aria-label="Distance weighting" /></div>
          </div>
          <div className="mt-6 flex flex-wrap items-center gap-3"><Button disabled={training || effectiveRows.length < 8 || features.length === 0} onClick={() => runTraining()}><WandSparkles />{training ? "Training models…" : "Train both models"}</Button><span className="text-xs text-muted-foreground">{numberFormat(effectiveRows.length)} prepared rows · {features.length} features · KNN capped at 5,000 training observations</span></div>
        </Panel>
        {result && <>
          <Panel title="Linear regression equation" subtitle="Standardized input coefficients estimated with the normal equation."><p className="break-words font-mono text-xs leading-7 text-foreground sm:text-sm">Sales = {money(result.intercept)}{features.map((feature, index) => ` ${((result.coefficients[index + 1] ?? 0) < 0 ? "−" : "+")} ${money(Math.abs(result.coefficients[index + 1] ?? 0))} × z(${featureLabels[feature] ?? feature})`)}</p><div className="mt-4 flex flex-wrap gap-2">{features.map((feature, index) => <Badge variant="secondary" key={feature}>{feature}: {(result.coefficients[index + 1] ?? 0).toFixed(2)}</Badge>)}</div></Panel>
          <div className="grid gap-5 xl:grid-cols-2"><PredictionScatter title="Linear regression · held-out data" actual={result.actual} predicted={result.linear} color="var(--chart-1)" /><PredictionScatter title="KNN regression · held-out data" actual={result.actual} predicted={result.knn} color="var(--chart-3)" /><LineFitChart title="Sales vs customers" rows={effectiveRows} x="Customers" /><LineFitChart title="Sales vs promotion" rows={effectiveRows} x="Promo" /></div>
        </>}
        {!result && <EmptyState title="Ready to fit both models">Training applies the selected split, features, distance rule and neighbor count to the prepared rows.</EmptyState>}
      </div>;
      case 5: return result && metrics ? <div className="space-y-5">
        <Panel title="Model comparison" subtitle="Metrics on train and held-out observations · MAPE excludes zero-sales rows."><div className="overflow-x-auto"><table className="w-full min-w-[650px] text-left text-xs"><thead><tr className="border-b border-border text-muted-foreground">{["Model", "Train RMSE", "Test RMSE", "Train MAPE", "Test MAPE", "Test R²"].map((label) => <th key={label} className="px-3 py-3 font-medium">{label}</th>)}</tr></thead><tbody>{[["Linear regression", metrics.linearTrain, metrics.linearTest], ["KNN regression", metrics.knnTrain, metrics.knnTest]].map(([name, trainMetric, testMetric]) => <tr key={String(name)} className="border-b border-border last:border-0"><td className="px-3 py-3 font-semibold">{name}</td><td className="px-3 py-3 tabular-nums">{money((trainMetric as ReturnType<typeof getMetrics>).rmse)}</td><td className="px-3 py-3 tabular-nums">{money((testMetric as ReturnType<typeof getMetrics>).rmse)}</td><td className="px-3 py-3 tabular-nums">{((trainMetric as ReturnType<typeof getMetrics>).mape).toFixed(2)}%</td><td className="px-3 py-3 tabular-nums">{((testMetric as ReturnType<typeof getMetrics>).mape).toFixed(2)}%</td><td className="px-3 py-3 tabular-nums">{((testMetric as ReturnType<typeof getMetrics>).r2).toFixed(3)}</td></tr>)}</tbody></table></div></Panel>
        <div className="grid gap-4 sm:grid-cols-2"><Metric label="Linear test RMSE" value={money(metrics.linearTest.rmse)} detail={`R² ${metrics.linearTest.r2.toFixed(3)} · MAPE ${metrics.linearTest.mape.toFixed(1)}%`} /><Metric label="KNN test RMSE" value={money(metrics.knnTest.rmse)} detail={`R² ${metrics.knnTest.r2.toFixed(3)} · MAPE ${metrics.knnTest.mape.toFixed(1)}%`} /></div>
        <div className="grid gap-5 xl:grid-cols-2"><Panel title="Train vs test RMSE" subtitle="Lower values indicate lower prediction error."><div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={[{ model: "Linear", train: metrics.linearTrain.rmse, test: metrics.linearTest.rmse }, { model: "KNN", train: metrics.knnTrain.rmse, test: metrics.knnTest.rmse }]}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis dataKey="model" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} /><YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><Tooltip {...tooltipStyle} formatter={(value) => money(Number(value))} /><Bar dataKey="train" name="Train RMSE" fill="var(--chart-2)" radius={[3, 3, 0, 0]} /><Bar dataKey="test" name="Test RMSE" fill="var(--chart-1)" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></Panel>
          <ChartPanel title="KNN test error by K" subtitle="A smaller neighbor count can fit noise more closely." data={result.sweep} dataKey="rmse" xKey="k" color="var(--chart-3)" formatter={money} /></div>
        <Panel title="Residual distribution" subtitle="Held-out actual minus predicted sales."><ResidualChart actual={result.actual} predicted={result.linear} color="var(--chart-1)" /></Panel>
        <Panel title="10-fold cross-validation" subtitle="Manual folds, each evaluated against its held-out observations."><div className="overflow-x-auto"><table className="w-full min-w-[420px] text-left text-xs"><thead><tr className="border-b border-border text-muted-foreground"><th className="px-3 py-2.5 font-medium">Fold</th><th className="px-3 py-2.5 font-medium">Linear RMSE</th><th className="px-3 py-2.5 font-medium">KNN RMSE</th><th className="px-3 py-2.5 font-medium">Lower error</th></tr></thead><tbody>{result.cv.map((fold) => <tr key={fold.fold} className="border-b border-border/70"><td className="px-3 py-2.5">{String(fold.fold).padStart(2, "0")}</td><td className="px-3 py-2.5 tabular-nums">{money(fold.linearRmse)}</td><td className="px-3 py-2.5 tabular-nums">{money(fold.knnRmse)}</td><td className="px-3 py-2.5"><Badge variant="secondary">{fold.linearRmse <= fold.knnRmse ? "Linear" : "KNN"}</Badge></td></tr>)}</tbody></table></div></Panel>
        <Panel title="Generalization gap"><div className="grid gap-4 sm:grid-cols-2"><GapCard model="Linear regression" train={metrics.linearTrain.rmse} test={metrics.linearTest.rmse} /><GapCard model="KNN regression" train={metrics.knnTrain.rmse} test={metrics.knnTest.rmse} /></div>{Math.abs(metrics.knnTest.rmse - metrics.knnTrain.rmse) > Math.abs(metrics.linearTest.rmse - metrics.linearTrain.rmse) * 1.25 && <div className="mt-4"><Badge variant="destructive">KNN: overfitting signal</Badge></div>}</Panel>
        <ConclusionCard metrics={metrics} />
        <div className="flex flex-wrap gap-2"><Button onClick={downloadResults}><ArrowDownToLine />Download Results as CSV</Button><Button variant="outline" onClick={() => setStage(4)}><RotateCcw />Adjust and retrain</Button></div>
      </div> : <Panel title="Evaluation awaits a trained model"><p className="text-sm text-muted-foreground">Train both regressors in the previous step to calculate metrics, validation folds and generalization.</p><Button className="mt-4" onClick={() => setStage(4)}>Go to model training<ChevronRight /></Button></Panel>;
      case 6: return result ? <div className="grid gap-5 xl:grid-cols-[0.8fr_1.2fr]">
        <Panel title="Store-day inputs" subtitle="Create a scenario from customer traffic, promotions and calendar context.">
          <div className="space-y-4"><label className="block text-xs text-muted-foreground">Customers<Input aria-label="Prediction customers" className="mt-2" type="number" min={0} value={customersInput} onChange={(event) => setCustomersInput(Number(event.target.value))} /></label><label className="block text-xs text-muted-foreground">Day of week<Select value={String(dayInput)} onValueChange={(value) => setDayInput(Number(value))}><SelectTrigger aria-label="Prediction day of week" className="mt-2"><SelectValue /></SelectTrigger><SelectContent>{["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map((day, index) => <SelectItem key={day} value={String(index + 1)}>{day}</SelectItem>)}</SelectContent></Select></label><ToggleRow label="Promotion active" checked={promoInput} onChange={setPromoInput} /><ToggleRow label="School holiday" checked={schoolInput} onChange={setSchoolInput} /></div>
        </Panel>
        <PredictionResultView rows={trainingRows} result={result} features={features} customers={customersInput} day={dayInput} promo={promoInput} school={schoolInput} metric={metric} k={k} weighted={weighted} />
      </div> : <div className="space-y-4"><EmptyState title="Train the models to open the playground">The live sales estimates use the fitted model weights and the selected KNN settings.</EmptyState><Button onClick={() => setStage(4)}>Train both models<ChevronRight /></Button></div>;
      default: return null;
    }
  }

  return <main className={dark ? "dark min-h-screen bg-background text-foreground" : "min-h-screen bg-background text-foreground"}>
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-8">
          <div className="flex min-w-0 items-center gap-3"><div className="grid h-9 w-9 shrink-0 place-items-center bg-primary text-primary-foreground"><Activity className="h-5 w-5" /></div><div className="min-w-0"><p className="truncate text-sm font-semibold">Rossmann Sales Lab</p><p className="text-[11px] text-muted-foreground">Linear regression vs KNN regression</p></div></div>
          <div className="flex items-center gap-2"><Badge variant="outline" className="hidden sm:inline-flex">BROWSER-ONLY · NO BACKEND</Badge><Button aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} title={dark ? "Light appearance" : "Dark appearance"} size="icon" variant="ghost" onClick={() => setDark((value) => !value)}>{dark ? <Sun /> : <Moon />}</Button></div>
        </div>
      </header>
      <div className="sticky top-0 z-20 border-b border-border bg-background/95 backdrop-blur">
        <div className="mx-auto max-w-[1440px] px-4 sm:px-8"><nav aria-label="Analysis pipeline" className="flex overflow-x-auto">{stages.map(({ label, short, icon: Icon }, index) => <button key={label} aria-current={stage === index ? "step" : undefined} onClick={() => setStage(index)} className={`group flex min-w-[82px] flex-1 items-center gap-2 border-b-2 px-2 py-3 text-left transition-colors sm:min-w-0 sm:px-3 ${stage === index ? "border-primary text-primary" : index < stage ? "border-chart-2 text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}><span className={`grid h-6 w-6 shrink-0 place-items-center text-[10px] font-semibold ${stage === index ? "bg-primary text-primary-foreground" : index < stage ? "bg-chart-2/15 text-chart-2" : "bg-muted text-muted-foreground"}`}>{index < stage ? <Check className="h-3.5 w-3.5" /> : <Icon className="h-3.5 w-3.5" />}</span><span className="hidden truncate text-xs font-medium md:inline">{label}</span><span className="text-xs font-medium md:hidden">{short}</span></button>)}</nav></div>
      </div>
      <div className="simulator-grid border-b border-border"><div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-4 py-5 sm:px-8"><div><p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">Rossmann Store Sales · Research simulator</p><h1 className="mt-1 text-xl font-semibold sm:text-2xl">{stages[stage]?.label ?? "Data collection"}</h1><p className="mt-1 text-xs text-muted-foreground">Step {stage + 1} of {stages.length} <span className="px-1.5">·</span> {sourceName}</p></div><div className="flex flex-wrap items-center gap-2"><Button variant="outline" disabled={fullRun} onClick={runFullSimulation}><Play />{fullRun ? "Running…" : "Run full simulation"}</Button>{rows.length > 0 && <Button variant="ghost" onClick={createRows}><RotateCcw />Reset data</Button>}</div>{fullRun && <div className="w-full"><div className="mb-1.5 flex justify-between text-xs text-muted-foreground"><span>Preparing records, fitting models and evaluating results</span><span>{progress}%</span></div><Progress value={progress} /></div>}</div></div>
      <div className="mx-auto max-w-[1440px] px-4 py-6 sm:px-8 sm:py-8">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{rows.length ? `${numberFormat(rows.length)} observations` : "Waiting for data"} {processedRows.length > 0 ? `· ${numberFormat(processedRows.length)} prepared` : ""}</p></div><div className="flex items-center gap-2"><Button variant="outline" size="sm" disabled={stage === 0} onClick={() => setStage((value) => Math.max(0, value - 1))}><ChevronLeft />Previous</Button><Button size="sm" disabled={stage === stages.length - 1} onClick={() => setStage((value) => Math.min(stages.length - 1, value + 1))}>Next step<ChevronRight /></Button></div></div>
        {renderStage()}
        <footer className="mt-12 flex flex-wrap items-center justify-between gap-3 border-t border-border py-5 text-[11px] text-muted-foreground"><span>Rossmann Sales Prediction Simulator</span><span>Linear regression · KNN · manual 10-fold evaluation</span></footer>
      </div>
    </div>
  </main>;
}

function SliderField({ label, value, unit, min, max, onChange }: { label: string; value: number; unit: string; min: number; max: number; onChange: (value: number) => void }) {
  return <div><div className="mb-2 flex items-center justify-between text-xs"><label className="font-medium text-foreground">{label}</label><span className="tabular-nums text-muted-foreground">{value}{unit}</span></div><Slider aria-label={label} value={[value]} min={min} max={max} step={1} onValueChange={(value) => onChange(value[0] ?? min)} /></div>;
}

function ChartPanel({ title, subtitle, data, dataKey, xKey, color, formatter, secondaryKey, secondaryColor }: { title: string; subtitle?: string; data: ChartDatum[]; dataKey: string; xKey: string; color: string; formatter: (value: number) => string; secondaryKey?: string; secondaryColor?: string }) {
  return <Panel title={title} subtitle={subtitle}><div className="h-64"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 5, left: 0 }}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis dataKey={xKey} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} tickFormatter={(value) => numberFormat(Number(value))} /><Tooltip {...tooltipStyle} formatter={(value, name) => [formatter(Number(value)), name]} /><Bar dataKey={dataKey} name={dataKey === "sales" ? "Avg sales" : "Avg customers"} fill={color} radius={[3, 3, 0, 0]} />{secondaryKey && <Bar dataKey={secondaryKey} name="Avg customers" fill={secondaryColor ?? "var(--chart-3)"} radius={[3, 3, 0, 0]} />}</ComposedChart></ResponsiveContainer></div></Panel>;
}

function HistogramPanel({ title, bins, mean, sd, color }: { title: string; bins: { label: string; count: number }[]; mean: number; sd: number; color: string }) {
  return <Panel title={title} subtitle={`Mean ${numberFormat(mean)} · standard deviation ${numberFormat(sd)}`}><div className="h-56"><ResponsiveContainer width="100%" height="100%"><BarChart data={bins}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis dataKey="label" tick={{ fill: "var(--muted-foreground)", fontSize: 9 }} /><YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><Tooltip {...tooltipStyle} /><Bar dataKey="count" name="Records" fill={color} radius={[2, 2, 0, 0]} /></BarChart></ResponsiveContainer></div></Panel>;
}

function ToggleRow({ label, description, checked, onChange }: { label: string; description?: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <div className="flex items-center justify-between gap-4 py-3"><div><p className="text-sm font-medium text-foreground">{label}</p>{description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}</div><Switch checked={checked} onCheckedChange={onChange} aria-label={label} /></div>;
}

function PreviewTable({ rows }: { rows: RossmannRow[] }) {
  if (!rows.length) return <p className="py-8 text-center text-sm text-muted-foreground">No rows available.</p>;
  return <div className="overflow-x-auto"><table className="w-full min-w-[700px] text-left text-xs"><thead><tr className="border-b border-border text-muted-foreground">{Object.keys(rows[0] ?? {}).map((column) => <th key={column} className="px-3 py-2.5 font-medium">{column}</th>)}</tr></thead><tbody>{rows.slice(0, 12).map((row, index) => <tr key={`${row.Store}-${row.Date}-${index}`} className="border-b border-border/60 last:border-0">{Object.values(row).map((value, columnIndex) => <td key={columnIndex} className="px-3 py-2 tabular-nums">{value === null ? "—" : typeof value === "number" ? numberFormat(value) : value}</td>)}</tr>)}</tbody></table></div>;
}

function PredictionScatter({ title, actual, predicted, color }: { title: string; actual: number[]; predicted: number[]; color: string }) {
  const data = actual.map((value, index) => ({ actual: value, predicted: predicted[index] ?? 0 }));
  const max = Math.max(...actual, ...predicted, 1);
  return <Panel title={title} subtitle="Predicted against observed sales · dashed line is perfect fit."><div className="h-64"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 8 }}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis type="number" dataKey="actual" name="Actual" domain={[0, max]} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><YAxis type="number" dataKey="predicted" name="Predicted" domain={[0, max]} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><Tooltip {...tooltipStyle} /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="var(--muted-foreground)" strokeDasharray="4 4" /><Scatter data={data.slice(0, 1000)} fill={color} fillOpacity={0.55} /></ScatterChart></ResponsiveContainer></div></Panel>;
}

function LineFitChart({ title, rows, x }: { title: string; rows: RossmannRow[]; x: "Customers" | "Promo" }) {
  const bins = useMemo(() => {
    const groups = new Map<number, { sales: number; count: number }>();
    for (const row of rows) {
      if (typeof row[x] !== "number" || typeof row.Sales !== "number") continue;
      const key = x === "Customers" ? Math.floor(row.Customers! / 100) * 100 : row.Promo ?? 0;
      const group = groups.get(key) ?? { sales: 0, count: 0 }; group.sales += row.Sales; group.count += 1; groups.set(key, group);
    }
    return Array.from(groups, ([key, value]) => ({ x: x === "Promo" ? key ? "Promo" : "No promo" : String(key), sales: value.sales / value.count })).sort((a, b) => x === "Promo" ? String(a.x).localeCompare(String(b.x)) : Number(a.x) - Number(b.x));
  }, [rows, x]);
  return <Panel title={title} subtitle={x === "Promo" ? "Average daily sales by promotion status." : "Mean sales in customer-count bands."}><div className="h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={bins}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis dataKey="x" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><Tooltip {...tooltipStyle} formatter={(value) => money(Number(value))} /><Area type="monotone" dataKey="sales" name="Avg sales" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.12} /></AreaChart></ResponsiveContainer></div></Panel>;
}

function ResidualChart({ actual, predicted, color }: { actual: number[]; predicted: number[]; color: string }) {
  const bins = useMemo(() => {
    const residuals = actual.map((value, index) => value - (predicted[index] ?? 0));
    const min = Math.min(...residuals); const max = Math.max(...residuals); const range = max - min || 1;
    const result = Array.from({ length: 20 }, (_, index) => ({ bin: String(Math.round(min + range * index / 20)), count: 0 }));
    residuals.forEach((value) => { const index = Math.min(19, Math.floor((value - min) / range * 20)); result[index]!.count += 1; }); return result;
  }, [actual, predicted]);
  return <div className="h-56"><ResponsiveContainer width="100%" height="100%"><BarChart data={bins}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis dataKey="bin" tick={{ fill: "var(--muted-foreground)", fontSize: 9 }} /><YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} /><Tooltip {...tooltipStyle} /><ReferenceLine x="0" stroke="var(--muted-foreground)" strokeDasharray="4 4" /><Bar dataKey="count" name="Residuals" fill={color} radius={[2, 2, 0, 0]} /></BarChart></ResponsiveContainer></div>;
}

function GapCard({ model, train, test }: { model: string; train: number; test: number }) {
  const gap = test - train;
  return <div className="flex items-center justify-between gap-3 border border-border px-4 py-3"><div><p className="text-sm font-semibold">{model}</p><p className="mt-1 text-xs text-muted-foreground">Test RMSE − train RMSE</p></div><Badge variant={gap > 0 ? "outline" : "secondary"}>{gap >= 0 ? "+" : "−"}{money(Math.abs(gap))}</Badge></div>;
}

function ConclusionCard({ metrics }: { metrics: { linearTest: ReturnType<typeof getMetrics>; knnTest: ReturnType<typeof getMetrics>; linearTrain: ReturnType<typeof getMetrics>; knnTrain: ReturnType<typeof getMetrics> } }) {
  const linearWins = metrics.linearTest.rmse <= metrics.knnTest.rmse;
  const overfit = Math.abs(metrics.knnTest.rmse - metrics.knnTrain.rmse) > Math.abs(metrics.linearTest.rmse - metrics.linearTrain.rmse) * 1.25;
  return <Panel title="Conclusion" subtitle="Based on the current train/test evaluation."><div className="border-l-2 border-primary pl-4"><p className="text-sm leading-6 text-foreground"><strong>{linearWins ? "Linear regression" : "KNN regression"}</strong> has the lower held-out RMSE in this run ({money(Math.min(metrics.linearTest.rmse, metrics.knnTest.rmse))} versus {money(Math.max(metrics.linearTest.rmse, metrics.knnTest.rmse))}). {linearWins ? "Its compact equation is straightforward to interpret." : "The chosen feature space and distance settings better capture local patterns."} {overfit ? "KNN shows a larger train-to-test error gap, a sign that local neighborhoods may be fitting sample-specific variation." : "The train/test error gaps are comparable, with no pronounced KNN overfitting signal."} Results depend on the generated or uploaded data and selected settings.</p></div></Panel>;
}

function PredictionResultView({ rows, result, features, customers, day, promo, school, metric, k, weighted }: { rows: RossmannRow[]; result: PredictionResult; features: (keyof RossmannRow)[]; customers: number; day: number; promo: boolean; school: boolean; metric: ModelOptions["metric"]; k: number; weighted: boolean }) {
  const prediction = useMemo(() => predictOne({ Store: 1, DayOfWeek: day, Date: "2025-01-01", Sales: null, Customers: customers, Open: 1, Promo: promo ? 1 : 0, StateHoliday: "0", SchoolHoliday: school ? 1 : 0 }, features, result, rows, metric, k, weighted), [rows, result, features, customers, day, promo, school, metric, k, weighted]);
  return <Panel title="Predicted sales" subtitle="Both models use the same scenario inputs."><div className="grid gap-5 sm:grid-cols-2"><div className="border-l-2 border-chart-1 pl-4"><p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">Linear regression</p><p className="mt-2 text-2xl font-semibold tabular-nums">{money(prediction.linear)}</p></div><div className="border-l-2 border-chart-3 pl-4"><p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">KNN regression · K = {k}</p><p className="mt-2 text-2xl font-semibold tabular-nums">{money(prediction.knn)}</p></div></div><div className="mt-8 border-t border-border pt-4"><p className="text-xs font-semibold uppercase tracking-[0.12em] text-primary">Inventory planning note</p><p className="mt-2 text-sm leading-6 text-muted-foreground">Plan for approximately {money(Math.max(prediction.linear, prediction.knn))} in sales for this scenario; the difference between model estimates ({money(Math.abs(prediction.linear - prediction.knn))}) can guide a flexible stock buffer.</p></div><div className="mt-4 flex flex-wrap gap-2"><Badge variant="secondary">{numberFormat(customers)} customers</Badge><Badge variant="secondary">{promo ? "Promo" : "No promo"}</Badge><Badge variant="secondary">Day {day}</Badge><Badge variant="secondary">{school ? "School holiday" : "Regular school day"}</Badge></div></Panel>;
}