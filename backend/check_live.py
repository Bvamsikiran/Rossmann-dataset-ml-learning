import urllib.request
import json

def test_live():
    print("=" * 65)
    print("  ROSSMANN SALES ML LEARNING LAB - SYSTEM VERIFICATION")
    print("=" * 65)
    
    # 1. Health
    with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as r:
        health = json.loads(r.read().decode())
        print(f"1. Backend Health:   {health['status'].upper()} (v{health['version']})")
        print(f"   Target Dir:       {health['dataset_dir']}")

    # 2. Files Status
    with urllib.request.urlopen("http://127.0.0.1:8000/api/dataset/status") as r:
        ds = json.loads(r.read().decode())
        print(f"\n2. Google Drive Dataset Status: {'[FOUND]' if ds['directory_exists'] else '[NOT FOUND]'}")
        for key, f in ds["files"].items():
            print(f"   - {f['name']:22}: {f['size_mb']:>6.2f} MB  (Exists: {f['exists']})")

    # 3. Pipeline Merge
    print("\n3. Testing Relational Multi-Dataset Pipeline...")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/dataset/load-merged",
        data=json.dumps({"sample_size": 2500}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        merged = json.loads(r.read().decode())
        print(f"   Successfully Joined {len(merged['tables_joined'])} Tables: {', '.join(merged['tables_joined'])}")
        print(f"   Merged Schema:     {merged['total_loaded_rows']:,} rows × {merged['total_columns']} features")

    # 4. Model Training
    print("\n4. Testing Supervised Model Training (Linear vs KNN)...")
    train_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/train",
        data=json.dumps({
            "features": ["Customers", "Promo", "DayOfWeek", "CompetitionDistance", "Mean_TemperatureC"],
            "split_percent": 80,
            "k": 7,
            "sample_size": 2500
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(train_req) as r:
        trained = json.loads(r.read().decode())
        print(f"   Linear Test RMSE:  {trained['linear_metrics']['test']['rmse']:,} (R² = {trained['linear_metrics']['test']['r2']})")
        print(f"   KNN Test RMSE:     {trained['knn_metrics']['test']['rmse']:,} (R² = {trained['knn_metrics']['test']['r2']})")
        print(f"   5-Fold CV Mean:    Linear RMSE = {trained['cv_mean']['mean_linear_rmse']:,} | KNN RMSE = {trained['cv_mean']['mean_knn_rmse']:,}")

    # 5. Live Single-Store Prediction
    print("\n5. Testing Live Store Turnover Prediction...")
    pred_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=json.dumps({
            "features": {
                "Customers": 720,
                "Promo": 1,
                "DayOfWeek": 5,
                "CompetitionDistance": 850,
                "Mean_TemperatureC": 18
            }
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(pred_req) as r:
        pred = json.loads(r.read().decode())
        print(f"   Linear Predicted Sales: {pred['linear_prediction']:,} kr")
        print(f"   KNN Predicted Sales:    {pred['knn_prediction']:,} kr")

    # 6. Frontend
    with urllib.request.urlopen("http://localhost:5173") as r:
        print(f"\n6. Vite Frontend Server: HTTP {r.status} OK (http://localhost:5173)")
    print("=" * 65)

if __name__ == "__main__":
    test_live()
