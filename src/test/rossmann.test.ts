import { describe, expect, it } from "vitest";
import { countMissing, generateSyntheticData, getMetrics, preprocessData, type RossmannRow } from "@/lib/rossmann";

describe("Rossmann analysis rules", () => {
  it("generates the requested store-by-day number of rows", () => {
    expect(generateSyntheticData({ stores: 3, days: 14, noise: 0, missingPercent: 0, outlierPercent: 0 }).length).toBe(42);
  });

  it("drops closed rows and fills missing numeric values when selected", () => {
    const rows: RossmannRow[] = [
      { Store: 1, DayOfWeek: 1, Date: "2024-01-01", Sales: 100, Customers: 10, Open: 1, Promo: 0, StateHoliday: "0", SchoolHoliday: 0 },
      { Store: 1, DayOfWeek: 7, Date: "2024-01-07", Sales: 0, Customers: 0, Open: 0, Promo: 0, StateHoliday: "0", SchoolHoliday: 0 },
      { Store: 2, DayOfWeek: 2, Date: "2024-01-02", Sales: null, Customers: 20, Open: 1, Promo: 1, StateHoliday: "0", SchoolHoliday: 0 },
    ];
    const processed = preprocessData(rows, { removeDuplicates: false, dropClosed: true, imputation: "mean", clipOutliers: false });
    expect(processed.summary.afterRows).toBe(2);
    expect(processed.summary.afterMissing).toBe(0);
    expect(processed.rows[1]?.Sales).toBe(100);
  });

  it("excludes zero-sales observations from MAPE division", () => {
    expect(getMetrics([0, 100], [40, 80]).mape).toBe(20);
  });

  it("counts actual null fields as missing", () => {
    const row: RossmannRow = { Store: 1, DayOfWeek: 1, Date: "", Sales: null, Customers: 0, Open: 1, Promo: 0, StateHoliday: "0", SchoolHoliday: 0 };
    expect(countMissing([row])).toBe(2);
  });
});