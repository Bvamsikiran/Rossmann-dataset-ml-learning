# Rossmann Sales ML Learning Lab

An interactive, production-grade Machine Learning laboratory and simulator for the Rossmann Store Sales dataset. This application features a React/TypeScript frontend and a FastAPI (Python + Pandas + Scikit-Learn) backend to integrate, explore, preprocess, train, and compare **Linear Regression** versus **K-Nearest Neighbors (KNN)** models.

## Key Features

- **Multi-Dataset Relational Integration**: Seamlessly loads and merges:
  - `train.csv` & `test.csv` (sales, customers, promotions, open status)
  - `store.csv` (store type, assortment, competition distance, Promo2)
  - `store_states.csv` & `state_names.csv` (geographic state mapping)
  - `weather.csv` (temperature, humidity, precipitation, weather events)
  - `googletrend.csv` (regional Google search interest)
- **7-Stage Workflow**:
  1. **Data Collection & Integration**: Multi-table merging and statistical previews.
  2. **Exploratory Data Analysis**: Histograms, scatter plots, and correlation matrix.
  3. **Preprocessing**: Missing value imputation, outlier management, and filtering.
  4. **Dynamic Feature Selection**: Select any merged feature from store, weather, and sales tables.
  5. **Model Training**: Scikit-Learn Linear Regression vs. KNN with k-fold cross validation.
  6. **Evaluation & Comparison**: RMSE, MAE, R², residual distributions, and coefficient analysis.
  7. **Live Prediction Playground**: Interactive simulator for predicting store revenue based on custom parameters.

## Architecture

```
Rossmann-dataset-ml-learning/
├── backend/                  # Python FastAPI Backend
│   ├── config.py             # Dataset paths & app settings
│   ├── data_loader.py        # Multi-CSV loader & validator
│   ├── pipeline.py           # Relational joins & feature engineering
│   ├── models.py             # Scikit-learn Linear Regression & KNN
│   ├── main.py               # FastAPI REST endpoints
│   └── requirements.txt      # Python dependencies
├── src/                      # React / TypeScript Frontend
│   ├── components/           # UI Components & RossmannSimulator
│   ├── routes/               # TanStack Router routes
│   └── lib/                  # Client-side algorithms & utilities
└── package.json              # Frontend scripts & dependencies
```

## Running the Application Locally

### 1. Start the Python Backend
Ensure Python 3.10+ is installed:

```bash
cd backend
pip install -r requirements.txt
python main.py
```
*The backend runs on `http://127.0.0.1:8000` (API documentation available at `/docs`).*

### 2. Start the Frontend
In another terminal:

```bash
npm install
npm run dev
```
*The frontend runs on `http://localhost:5173` and automatically proxies `/api` calls to the FastAPI backend.*
