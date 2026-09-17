# Getaround Pricing Model - MLflow Experiment Tracking
# ---------------------------------------------------------------------------

from pathlib import Path

import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "get_around_pricing_project.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "getaround_pricing_model.joblib"
)


# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("getaround-pricing-model")


# ---------------------------------------------------------------------------
# Load and prepare pricing data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Remove the exported row index
df = df.drop(columns=["Unnamed: 0"])

TARGET = "rental_price_per_day"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print(f"Training observations: {len(X_train):,}")
print(f"Test observations: {len(X_test):,}")


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

numeric_features = [
    "mileage",
    "engine_power",
]

categorical_features = [
    "model_key",
    "fuel",
    "paint_color",
    "car_type",
]

boolean_features = [
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("boolean", "passthrough", boolean_features),
    ]
)

print("Preprocessing pipeline ready.")


# ---------------------------------------------------------------------------
# Experiment 1 — Linear Regression
# ---------------------------------------------------------------------------

with mlflow.start_run(run_name="linear-regression"):

    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    linear_pipeline.fit(X_train, y_train)

    y_pred = linear_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "LinearRegression")

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print("\nLinear Regression")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")


# ---------------------------------------------------------------------------
# Experiment 2 — Ridge Regression
# ---------------------------------------------------------------------------

with mlflow.start_run(run_name="ridge-regression"):

    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
            ),
        ]
    )

    ridge_pipeline.fit(X_train, y_train)

    y_pred = ridge_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    selected_alpha = ridge_pipeline.named_steps["model"].alpha_

    mlflow.log_param("model", "RidgeCV")
    mlflow.log_param("selected_alpha", selected_alpha)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print("\nRidge Regression")
    print(f"Selected alpha: {selected_alpha}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")


# ---------------------------------------------------------------------------
# Experiment 3 — Random Forest Baseline
# ---------------------------------------------------------------------------

with mlflow.start_run(run_name="random-forest-baseline"):

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print("\nRandom Forest Baseline")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")


# ---------------------------------------------------------------------------
# Experiment 4 — Tuned Random Forest
# ---------------------------------------------------------------------------

with mlflow.start_run(run_name="random-forest-tuned"):

    tuned_rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=15,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    tuned_rf_pipeline.fit(X_train, y_train)

    y_pred = tuned_rf_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 400)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_leaf", 1)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.log_artifact(str(MODEL_PATH))

    print("\nTuned Random Forest")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")