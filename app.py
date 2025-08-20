# app.py
# Simple ETA predictor: trains a model on data.csv, saves it, and does one demo prediction.

# app.py
# Simple ETA PREDICTOR: trains a model and does one emo prediction 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

DATA_FILE = "data.csv"
MODEL_FILE = "eta_model.pkl"

EXPECTED_COLS = ["distance_km", "prep_time_min", "traffic_index", "delivery_time_min"]

def main():
    # 1) Load data
    df = pd.read_csv(DATA_FILE)

    # 2) Basic validation
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {DATA_FILE}: {missing}\n"
                         f"Expected columns: {EXPECTED_COLS}")

    # 3) Split features/target
    X = df[["distance_km", "prep_time_min", "traffic_index"]]
    y = df["delivery_time_min"]

 # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 7) Save model
    Path(".").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("✅ Model trained and saved ->", MODEL_FILE)
    print(f"📏 Test MAE: {mae:.2f} minutes | R²: {r2:.3f}")

    # 8) One demo prediction
    demo = pd.DataFrame([[5, 15, 3]], columns=["distance_km", "prep_time_min", "traffic_index"])
    demo_eta = model.predict(demo)[0]
    print(f"🔮 Demo prediction for distance=5km, prep=15m, traffic=3 → ETA ≈ {demo_eta:.1f} minutes")

if _name_ == "_main_":
    main()
