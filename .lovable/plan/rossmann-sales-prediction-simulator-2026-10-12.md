# Rossmann Sales Prediction Simulator

Build the requested single-page, browser-only simulator at `/` with an interactive seven-step workflow, polished analytics-focused styling, and no backend.

## Experience
- Replace the starter page with a titled app shell, research abstract, light/dark toggle, sticky step navigation, progress display, and clear empty/loading states.
- Implement the ordered workflow: data generation/import and preview; exploratory charts and insights; applicable preprocessing controls and before/after summary; numeric correlations, heatmap, ranking, feature selection and normalization; configurable linear/KNN training and predictions; train/test metrics, manual 10-fold comparison, overfitting and conclusion; and live prediction inputs.
- Add full-simulation progress and results CSV download. Synthetic data will follow the specified Rossmann sales relationships and include missingness/outliers; CSV import will parse and cap the sample at 20,000 rows.

## Technical approach
- Keep data generation, preprocessing, correlations, normalization, linear regression, KNN, cross-validation, and metrics in browser-side TypeScript; write small adjacent tests for the algorithms and metric rules.
- Use Recharts for all requested visualizations and PapaParse for CSV import; add PapaParse only if it is not already installed.
- Organize the UI and pure calculation helpers into focused modules, keep state in the page/browser, and preserve the existing TanStack Start routing conventions. Add a root `AGENTS.md` architecture rule for the client-side calculation boundary.
- Use the existing semantic token system, add any needed tokens in `src/styles.css`, and give the home route its own app-specific metadata.

## Validation
- Run relevant simulator tests, inspect the preview at desktop and narrow widths, exercise synthetic generation through training/evaluation and playground predictions, verify CSV upload/download and full-run progress, then check the latest build and runtime logs.
