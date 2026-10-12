export type RossmannRow = {
  Store: number | null;
  DayOfWeek: number | null;
  Date: string;
  Sales: number | null;
  Customers: number | null;
  Open: number | null;
  Promo: number | null;
  StateHoliday: string;
  SchoolHoliday: number | null;
};

export type PreprocessOptions = {
  removeDuplicates: boolean;
  dropClosed: boolean;
  imputation: "mean" | "median";
  clipOutliers: boolean;
};

export type ModelOptions = {
  features: (keyof RossmannRow)[];
  splitPercent: number;
  k: number;
  metric: "euclidean" | "manhattan" | "minkowski";
  weighted: boolean;
  seed?: number;
};

export type PredictionResult = {
  linear: number[];
  knn: number[];
  actual: number[];
  trainLinear: number[];
  trainKnn: number[];
  trainActual: number[];
  coefficients: number[];
  intercept: number;
  testIndices: number[];
  trainIndices: number[];
  cv: { fold: number; linearRmse: number; knnRmse: number }[];
  sweep: { k: number; rmse: number }[];
};

export const numericColumns: (keyof RossmannRow)[] = [
  "Store", "DayOfWeek", "Sales", "Customers", "Open", "Promo", "SchoolHoliday",
];
export const selectableFeatures: (keyof RossmannRow)[] = [
  "Customers", "Promo", "DayOfWeek", "SchoolHoliday", "Store", "Open",
];

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(random: () => number) {
  const first = Math.max(random(), Number.EPSILON);
  const second = random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

export function generateSyntheticData(options: {
  stores: number;
  days: number;
  noise: number;
  missingPercent: number;
  outlierPercent: number;
  seed?: number;
}): RossmannRow[] {
  const random = seededRandom(options.seed ?? 42);
  const rows: RossmannRow[] = [];
  const start = new Date(Date.UTC(2013, 0, 1));
  const stores = Math.max(1, Math.min(200, Math.round(options.stores)));
  const days = Math.max(7, Math.min(730, Math.round(options.days)));
  for (let store = 1; store <= stores; store += 1) {
    const storeFactor = 0.78 + random() * 0.55;
    for (let offset = 0; offset < days; offset += 1) {
      const date = new Date(start.getTime() + offset * 86_400_000);
      const dayOfWeek = (date.getUTCDay() + 6) % 7 + 1;
      const closed = dayOfWeek === 7 || random() < 0.018;
      const promo = !closed && random() < 0.42 ? 1 : 0;
      const schoolHoliday = random() < 0.12 ? 1 : 0;
      const quarterFactor = date.getUTCMonth() >= 9 ? 1.14 : date.getUTCMonth() < 3 ? 0.94 : 1;
      const weekdayFactor = dayOfWeek === 1 || dayOfWeek === 5 ? 1.12 : dayOfWeek === 7 ? 0.65 : 1;
      const customers = closed ? 0 : Math.max(90, Math.round((430 + random() * 280 + (promo ? 65 : 0)) * storeFactor * weekdayFactor));
      const baseSales = closed ? 0 : (customers * 8.1 + 2100 + promo * customers * 1.95) * quarterFactor * weekdayFactor;
      let sales = closed ? 0 : Math.max(0, baseSales + gaussian(random) * baseSales * options.noise / 100);
      if (!closed && random() < options.outlierPercent / 100) sales *= random() < 0.5 ? 2.2 : 0.35;
      rows.push({
        Store: store, DayOfWeek: dayOfWeek, Date: date.toISOString().slice(0, 10),
        Sales: Math.round(sales), Customers: customers, Open: closed ? 0 : 1,
        Promo: promo, StateHoliday: "0", SchoolHoliday: schoolHoliday,
      });
    }
  }
  const missingCount = Math.floor(rows.length * Math.max(0, Math.min(80, options.missingPercent)) / 100);
  const mutableColumns: (keyof RossmannRow)[] = ["Sales", "Customers", "Promo", "SchoolHoliday"];
  for (let index = 0; index < missingCount; index += 1) {
    const row = rows[Math.floor(random() * rows.length)];
    const colKey = mutableColumns[Math.floor(random() * mutableColumns.length)];
    if (row && colKey) {
      (row as Record<string, unknown>)[colKey] = null;
    }
  }
  return rows;
}

function parseNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const number = typeof value === "number" ? value : Number(String(value).replaceAll(",", ""));
  return Number.isFinite(number) ? number : null;
}

export function normalizeUploadedRow(record: Record<string, unknown>): RossmannRow {
  return {
    Store: parseNumber(record["Store"]),
    DayOfWeek: parseNumber(record["DayOfWeek"]),
    Date: String(record["Date"] ?? ""),
    Sales: parseNumber(record["Sales"]),
    Customers: parseNumber(record["Customers"]),
    Open: parseNumber(record["Open"]),
    Promo: parseNumber(record["Promo"]),
    StateHoliday: String(record["StateHoliday"] ?? ""),
    SchoolHoliday: parseNumber(record["SchoolHoliday"]),
  };
}

export function countMissing(rows: RossmannRow[]) {
  return rows.reduce((total, row) => total + Object.values(row).filter((value) => value === null || value === "").length, 0);
}

export function preprocessData(rows: RossmannRow[], options: PreprocessOptions) {
  let result = rows.map((row) => ({ ...row }));
  const beforeRows = result.length;
  const beforeMissing = countMissing(result);
  let duplicatesRemoved = 0;
  if (options.removeDuplicates) {
    const seen = new Set<string>();
    result = result.filter((row) => {
      const key = JSON.stringify(row);
      if (seen.has(key)) { duplicatesRemoved += 1; return false; }
      seen.add(key);
      return true;
    });
  }
  if (options.dropClosed) result = result.filter((row) => row.Open !== 0);
  for (const column of numericColumns) {
    const values = result.map((row) => row[column]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    const fill = options.imputation === "median"
      ? sorted.length === 0 ? 0 : sorted.length % 2 ? sorted[middle] ?? 0 : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
      : values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
    for (const row of result) if (row[column] === null) (row[column] as number | null) = fill;
  }
  if (options.clipOutliers) {
    for (const column of ["Sales", "Customers"] as const) {
      const values = result.map((row) => row[column] ?? 0).sort((a, b) => a - b);
      if (!values.length) continue;
      const q1 = values[Math.floor((values.length - 1) * 0.25)] ?? 0;
      const q3 = values[Math.floor((values.length - 1) * 0.75)] ?? 0;
      const spread = q3 - q1;
      for (const row of result) row[column] = Math.min(q3 + spread * 1.5, Math.max(q1 - spread * 1.5, row[column] ?? 0));
    }
  }
  return {
    rows: result,
    summary: { beforeRows, afterRows: result.length, beforeMissing, afterMissing: countMissing(result), duplicatesRemoved },
  };
}

export function correlation(a: number[], b: number[]) {
  if (a.length < 2 || a.length !== b.length) return 0;
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
  let numerator = 0; let squaredA = 0; let squaredB = 0;
  for (let index = 0; index < a.length; index += 1) {
    const dx = (a[index] ?? 0) - meanA; const dy = (b[index] ?? 0) - meanB;
    numerator += dx * dy; squaredA += dx * dx; squaredB += dy * dy;
  }
  return squaredA === 0 || squaredB === 0 ? 0 : numerator / Math.sqrt(squaredA * squaredB);
}

export function getMetrics(actual: number[], predicted: number[]) {
  const count = Math.min(actual.length, predicted.length);
  if (!count) return { rmse: 0, mape: 0, r2: 0 };
  const mean = actual.slice(0, count).reduce((sum, value) => sum + value, 0) / count;
  let squaredError = 0; let absolutePercent = 0; let usable = 0; let totalVariation = 0;
  for (let index = 0; index < count; index += 1) {
    const target = actual[index] ?? 0; const estimate = predicted[index] ?? 0;
    squaredError += (target - estimate) ** 2;
    totalVariation += (target - mean) ** 2;
    if (target !== 0) { absolutePercent += Math.abs((target - estimate) / target); usable += 1; }
  }
  return { rmse: Math.sqrt(squaredError / count), mape: usable ? absolutePercent / usable * 100 : 0, r2: totalVariation ? 1 - squaredError / totalVariation : 0 };
}

function solve(matrix: number[][], vector: number[]) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index] ?? 0]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) if (Math.abs(augmented[row]?.[column] ?? 0) > Math.abs(augmented[pivot]?.[column] ?? 0)) pivot = row;
    [augmented[column], augmented[pivot]] = [augmented[pivot] ?? [], augmented[column] ?? []];
    const divisor = augmented[column]?.[column] ?? 0;
    if (Math.abs(divisor) < 1e-12) continue;
    for (let j = column; j <= size; j += 1) augmented[column]![j] = (augmented[column]?.[j] ?? 0) / divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row]?.[column] ?? 0;
      for (let j = column; j <= size; j += 1) augmented[row]![j] = (augmented[row]?.[j] ?? 0) - factor * (augmented[column]?.[j] ?? 0);
    }
  }
  return augmented.map((row, index) => row[size] ?? (index === 0 ? 0 : 0));
}

type Prepared = { values: number[]; target: number };

function fitLinear(rows: Prepared[], dimensions: number) {
  const size = dimensions + 1;
  const matrix = Array.from({ length: size }, () => Array<number>(size).fill(0));
  const vector = Array<number>(size).fill(0);
  for (const row of rows) {
    const input = [1, ...row.values];
    for (let i = 0; i < size; i += 1) {
      vector[i] = (vector[i] ?? 0) + (input[i] ?? 0) * row.target;
      for (let j = 0; j < size; j += 1) matrix[i]![j] = (matrix[i]?.[j] ?? 0) + (input[i] ?? 0) * (input[j] ?? 0);
    }
  }
  for (let i = 1; i < size; i += 1) matrix[i]![i] = (matrix[i]?.[i] ?? 0) + 1e-7;
  return solve(matrix, vector);
}

function predictLinear(rows: Prepared[], coefficients: number[]) {
  return rows.map((row) => coefficients[0]! + row.values.reduce((sum, value, index) => sum + value * (coefficients[index + 1] ?? 0), 0));
}

function predictKnn(training: Prepared[], queries: Prepared[], options: Pick<ModelOptions, "k" | "metric" | "weighted">) {
  const p = 3;
  return queries.map((query) => {
    const neighbors = training.map((row) => {
      let sum = 0;
      for (let index = 0; index < query.values.length; index += 1) {
        const difference = Math.abs((query.values[index] ?? 0) - (row.values[index] ?? 0));
        sum += options.metric === "manhattan" ? difference : options.metric === "minkowski" ? difference ** p : difference * difference;
      }
      return { distance: options.metric === "manhattan" ? sum : options.metric === "minkowski" ? sum ** (1 / p) : Math.sqrt(sum), target: row.target };
    }).sort((a, b) => a.distance - b.distance).slice(0, Math.max(1, options.k));
    const exact = neighbors.filter((neighbor) => neighbor.distance < 1e-12);
    const chosen = exact.length ? exact : neighbors;
    const totalWeight = chosen.reduce((sum, neighbor) => sum + (options.weighted ? 1 / Math.max(neighbor.distance, 1e-9) : 1), 0);
    return totalWeight ? chosen.reduce((sum, neighbor) => sum + neighbor.target * (options.weighted ? 1 / Math.max(neighbor.distance, 1e-9) : 1), 0) / totalWeight : 0;
  });
}

function metricsRmse(actual: number[], predicted: number[]) {
  return getMetrics(actual, predicted).rmse;
}

export function trainAndEvaluate(rows: RossmannRow[], options: ModelOptions): PredictionResult {
  const eligible = rows.filter((row) => typeof row.Sales === "number" && options.features.every((feature) => typeof row[feature] === "number"));
  if (eligible.length < 8 || !options.features.length) throw new Error("Select at least one feature and provide eight or more complete rows.");
  const fitRows = eligible.length > 5000 ? eligible.filter((_, index) => index % Math.ceil(eligible.length / 5000) === 0) : eligible;
  const means = options.features.map((feature) => fitRows.reduce((sum, row) => sum + (row[feature] as number), 0) / fitRows.length);
  const deviations = options.features.map((feature, index) => {
    const mean = means[index] ?? 0;
    return Math.sqrt(fitRows.reduce((sum, row) => sum + ((row[feature] as number) - mean) ** 2, 0) / fitRows.length) || 1;
  });
  const prepared: Prepared[] = fitRows.map((row) => ({
    values: options.features.map((feature, index) => ((row[feature] as number) - (means[index] ?? 0)) / (deviations[index] ?? 1)),
    target: row.Sales ?? 0,
  }));
  const random = seededRandom(options.seed ?? 13);
  const indices = prepared.map((_, index) => index);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [indices[index], indices[other]] = [indices[other] ?? 0, indices[index] ?? 0];
  }
  const trainCount = Math.max(4, Math.min(prepared.length - 2, Math.round(prepared.length * options.splitPercent / 100)));
  const trainIndices = indices.slice(0, trainCount);
  const testIndices = indices.slice(trainCount);
  const training = trainIndices.map((index) => prepared[index]).filter((row): row is Prepared => row !== undefined);
  const testing = testIndices.map((index) => prepared[index]).filter((row): row is Prepared => row !== undefined);
  const coefficients = fitLinear(training, options.features.length);
  const linear = predictLinear(testing, coefficients);
  const knn = predictKnn(training, testing, options);
  const trainLinear = predictLinear(training, coefficients);
  const trainKnn = training.map((query, index) => {
    const withoutSelf = training.filter((_, candidateIndex) => candidateIndex !== index);
    return predictKnn(withoutSelf, [query], options)[0] ?? 0;
  });

  const cvSample = prepared.length > 1200 ? prepared.filter((_, index) => index % Math.ceil(prepared.length / 1200) === 0) : prepared;
  const foldRows = Array.from({ length: 10 }, (_, index) => ({ fold: index + 1, rows: cvSample.filter((_, rowIndex) => rowIndex % 10 === index) }));
  const cv = foldRows.map(({ fold, rows: validation }) => {
    const foldTraining = cvSample.filter((_, index) => index % 10 !== fold - 1);
    const linearFit = fitLinear(foldTraining, options.features.length);
    return {
      fold,
      linearRmse: metricsRmse(validation.map((row) => row.target), predictLinear(validation, linearFit)),
      knnRmse: metricsRmse(validation.map((row) => row.target), predictKnn(foldTraining, validation, options)),
    };
  });

  const sweepTraining = training.length > 900 ? training.filter((_, index) => index % Math.ceil(training.length / 900) === 0) : training;
  const sweepTesting = testing.length > 300 ? testing.filter((_, index) => index % Math.ceil(testing.length / 300) === 0) : testing;
  const sweep = Array.from({ length: 30 }, (_, index) => {
    const candidateK = index + 1;
    const predictions = predictKnn(sweepTraining, sweepTesting, { ...options, k: candidateK });
    return { k: candidateK, rmse: metricsRmse(sweepTesting.map((row) => row.target), predictions) };
  });

  return {
    linear, knn, actual: testing.map((row) => row.target),
    trainLinear, trainKnn, trainActual: training.map((row) => row.target),
    coefficients, intercept: coefficients[0] ?? 0, testIndices, trainIndices, cv, sweep,
  };
}

export function predictOne(row: RossmannRow, features: (keyof RossmannRow)[], result: PredictionResult, trainingRows: RossmannRow[], metric: ModelOptions["metric"], k: number, weighted: boolean) {
  const values = features.map((feature) => typeof row[feature] === "number" ? row[feature] as number : 0);
  const trainingRaw = result.trainIndices.map((index) => trainingRows[index]).filter((item): item is RossmannRow => item !== undefined);
  const trainingPrepared: Prepared[] = trainingRaw.map((item) => ({
    values: features.map((feature, featureIndex) => {
      const rawValues = trainingRaw.map((candidate) => candidate[feature]).filter((value): value is number => typeof value === "number");
      const average = rawValues.reduce((sum, value) => sum + value, 0) / (rawValues.length || 1);
      const sd = Math.sqrt(rawValues.reduce((sum, value) => sum + (value - average) ** 2, 0) / (rawValues.length || 1)) || 1;
      return (((item[feature] as number) - average) / sd);
    }), target: item.Sales ?? 0,
  }));
  const means = features.map((feature) => {
    const vals = trainingRaw.map((item) => item[feature]).filter((value): value is number => typeof value === "number");
    return vals.reduce((sum, value) => sum + value, 0) / (vals.length || 1);
  });
  const deviations = features.map((feature, index) => {
    const average = means[index] ?? 0;
    const vals = trainingRaw.map((item) => item[feature]).filter((value): value is number => typeof value === "number");
    return Math.sqrt(vals.reduce((sum, value) => sum + (value - average) ** 2, 0) / (vals.length || 1)) || 1;
  });
  const query: Prepared = { values: values.map((value, index) => (value - (means[index] ?? 0)) / (deviations[index] ?? 1)), target: 0 };
  const linearPrediction = result.intercept + query.values.reduce((sum, value, index) => sum + value * (result.coefficients[index + 1] ?? 0), 0);
  const knnPrediction = predictKnn(trainingPrepared, [query], { k, metric, weighted })[0] ?? 0;
  return { linear: Math.max(0, linearPrediction), knn: Math.max(0, knnPrediction) };
}
