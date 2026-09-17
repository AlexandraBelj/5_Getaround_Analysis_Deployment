# Getaround Analysis & Deployment

## Overview

This project analyzes rental delays on **Getaround** and develops a production-ready machine-learning service for rental-price prediction.

The project addresses two complementary business problems:

1. **Rental Delay Analysis** — determine how much time should be required between two consecutive rentals of the same car, and whether such a policy should apply to all cars or only Connect-enabled rentals.
2. **Rental Pricing Prediction** — build, track, and deploy a machine-learning model that predicts a vehicle's daily rental price from its characteristics and equipment.

The complete project combines **data analysis, machine learning, experiment tracking, API development, containerization, and cloud deployment**.

It was developed as part of the **Jedha Data Science certification**.

---

## Applications & Services

| Component | Access |
|---|---|
| **Rental Delay Dashboard** | https://alexbelj-getaround-delay-dashboard.hf.space/ |
| **Pricing API** | https://alexbelj-getaround-pricing-api.hf.space/ |
| **Interactive API Documentation** | https://alexbelj-getaround-pricing-api.hf.space/docs |
| **MLflow Experiment Tracking** | Local — run `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`, then open http://127.0.0.1:5000 |

The **Streamlit dashboard** and **FastAPI pricing service** are deployed publicly on Hugging Face Spaces.

**MLflow is intentionally run locally** for experiment tracking. The repository contains the reproducible tracking script, while the generated SQLite database and MLflow artifacts are excluded from Git.

---

## Business Problem

### Rental Delays

Getaround allows drivers to rent cars for specific time periods. When two rentals of the same vehicle are scheduled close together, a late return from the first driver may interfere with the next driver's booking.

Introducing a mandatory minimum delay between rentals could reduce this risk, but it also reduces scheduling flexibility and may affect marketplace activity.

The analysis therefore focuses on two product decisions:

> **How long should the minimum gap between consecutive rentals be, and should the policy apply to all cars or only Connect cars?**

A key distinction is made between a **late checkout** and actual **next-driver interference**.

A late checkout becomes problematic only when:

```text
previous checkout delay > scheduled gap before the next rental
```

The analysis therefore evaluates the trade-off between:

- protecting the next driver from interference;
- restricting historical consecutive-booking configurations.

A restricted booking is **not interpreted as a lost booking**. It could potentially be rescheduled or fulfilled with another vehicle.

### Rental Pricing

The second workstream develops a regression model to estimate:

```text
rental_price_per_day
```

from vehicle characteristics such as mileage, engine power, model, fuel type, car type, equipment, and Getaround Connect availability.

The selected model is exposed through a production FastAPI endpoint.

---

## Datasets

Two datasets are used independently.

### Rental Delay Dataset

`get_around_delay_analysis.xlsx`

The dataset contains:

- **21,310 rentals**;
- **8,143 unique cars**;
- rental state;
- checkout delay;
- check-in type;
- previous rental identifier;
- planned time between consecutive rentals.

The dataset only identifies a previous ended rental when the planned gap is below **12 hours**.

After linking consecutive rentals:

- **1,841 consecutive-rental situations** are available for policy simulation;
- **1,729 situations** have an observed previous checkout delay and can be used to measure interference.

Missing checkout delays are not imputed.

### Rental Pricing Dataset

`get_around_pricing_project.csv`

The pricing dataset contains **4,843 vehicles** and includes:

- model;
- mileage;
- engine power;
- fuel type;
- paint color;
- car type;
- private parking;
- GPS;
- air conditioning;
- automatic transmission;
- Getaround Connect;
- speed regulator;
- winter tires;
- daily rental price.

The exported `Unnamed: 0` index column is removed before modeling.

The raw datasets are intentionally **not committed to this repository**.

---

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Streamlit
- FastAPI
- Pydantic
- MLflow
- joblib
- Docker
- Hugging Face Spaces
- Git / GitHub
- Jupyter Notebook

---

## Project Structure

```text
5_Getaround_Analysis_Deployment/
│
├── api/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── dashboard/
│   ├── app.py
│   ├── Dockerfile
│   ├── getaround_logo.png
│   └── requirements.txt
│
├── data/
│   └── raw/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_Delay_Analysis.ipynb
│   └── 02_Pricing_Prediction.ipynb
│
├── outputs/
│   ├── figures/
│   └── models/
│       └── getaround_pricing_model.joblib
│
├── mlflow_tracking.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 01 — Delay Analysis

The first notebook investigates rental delays and consecutive-booking relationships.

The analysis covers:

- data quality and missing values;
- checkout-delay distributions;
- consecutive-rental reconstruction;
- late-return frequency;
- next-driver interference;
- interference severity;
- Connect versus mobile rental patterns;
- minimum-gap policy simulation;
- operational impact of different thresholds;
- limitations of estimating revenue exposure.

### 02 — Pricing Prediction

The second notebook develops the rental-price prediction pipeline.

It includes:

- pricing-data preparation;
- feature and target definition;
- preprocessing;
- train/test splitting;
- Linear Regression;
- Ridge Regression;
- Random Forest baseline;
- Random Forest tuning;
- model comparison;
- final model selection;
- serialization of the complete prediction pipeline.

---

## Rental Delay Findings

### Late Returns Are Common, but Not Every Late Return Affects the Next Driver

Among ended rentals with an observed checkout delay:

- **57.5% were returned late**.

However, among the **1,729 analyzable consecutive rentals**, only:

- **12.6% experienced next-driver interference**.

This distinction is important: a vehicle can be returned late without affecting the next booking if sufficient time was scheduled between rentals.

Among previous rentals that were late, approximately **25% actually produced interference** with the following rental.

### Most Interference Is Relatively Short, but Severe Cases Exist

Among the **218 observed problematic consecutive rentals**:

- median interference was **26.5 minutes**;
- just over half lasted **30 minutes or less**;
- approximately **18.8% lasted more than two hours**.

The distribution is strongly right-skewed, meaning a smaller number of cases involve substantially longer disruption.

### Connect Rentals Show Lower Observed Interference Rates

For analyzable consecutive rentals:

| Scope | Consecutive rentals | Problematic cases | Problematic rate |
|---|---:|---:|---:|
| Connect | 791 | 69 | 8.72% |
| Mobile | 938 | 149 | 15.88% |

These differences are **descriptive associations** and should not be interpreted as evidence that Connect technology causes lower delay or interference.

---

## Minimum-Gap Policy Simulation

A policy is simulated by identifying historical consecutive bookings whose planned gap was shorter than a proposed minimum.

The threshold rule is:

```text
planned gap < minimum required gap
```

Bookings exactly equal to the threshold remain allowed.

### All Cars

| Minimum gap | Bookings requiring different timing | Interference potentially addressed |
|---:|---:|---:|
| 30 min | 15.15% | 53.21% |
| 60 min | 21.78% | 66.97% |
| 90 min | 31.72% | 78.90% |
| 120 min | 36.18% | 82.57% |
| 180 min | 47.26% | 89.91% |

### Connect Cars Only

| Minimum gap | Bookings requiring different timing | Interference potentially addressed |
|---:|---:|---:|
| 30 min | 16.11% | 57.97% |
| 60 min | 22.26% | 69.57% |
| 90 min | 31.98% | 81.16% |
| 120 min | 36.29% | 85.51% |
| 180 min | 45.76% | 92.75% |

Increasing the minimum gap protects more historical problematic cases, but progressively restricts more consecutive-booking configurations.

The analysis therefore shows **diminishing operational returns** as the threshold increases.

---

## Product Recommendation

A **60-minute minimum gap for Connect cars** is proposed as the starting point for product testing.

Historically, this scenario would have:

- required different timing for approximately **22.3% of consecutive Connect bookings**;
- potentially addressed approximately **69.6% of observed Connect interference cases**.

A 30-minute threshold creates less scheduling friction but leaves more than 40% of observed problematic Connect cases unaddressed.

Moving beyond 60 minutes provides additional protection, but scheduling restrictions increase substantially.

The 60-minute Connect-only policy is therefore presented as a **practical product-testing starting point**, not as a mathematically optimal policy.

A real rollout should monitor:

- booking conversion;
- rescheduling;
- cancellations;
- owner revenue;
- next-driver incidents.

---

## Revenue Limitation

The project brief asks how much owner revenue could potentially be affected by a minimum-gap policy.

The available datasets do **not** allow this quantity to be calculated directly.

The delay dataset and pricing dataset do not share a common `rental_id` or `car_id`, so historical consecutive rentals cannot be matched to their rental prices.

For this reason:

> **The share of bookings requiring different timing is used as an operational exposure indicator, not as an estimate of revenue loss.**

A booking affected by the policy is also not necessarily lost: it may be rescheduled or fulfilled with another vehicle.

---

## Pricing Model

### Preprocessing

The machine-learning pipeline separates the predictors into three groups.

**Numerical features**

- mileage;
- engine power.

These features are standardized with `StandardScaler`.

**Categorical features**

- model;
- fuel;
- paint color;
- car type.

These are encoded using `OneHotEncoder(handle_unknown="ignore")`.

**Boolean features**

Vehicle-equipment indicators are passed directly to the model.

The complete preprocessing and regression workflow is implemented as a scikit-learn `Pipeline`, ensuring that training and production predictions use the same transformations.

---

## Model Comparison

The dataset is split into **80% training data and 20% test data** using `random_state=42`.

Four regression configurations are evaluated:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 12.12 | 17.96 | 0.694 |
| RidgeCV | 12.12 | 17.97 | 0.693 |
| Random Forest baseline | 10.63 | 16.71 | 0.735 |
| **Tuned Random Forest** | **10.59** | **16.57** | **0.739** |

The tuned Random Forest performs best **among the models tested**.

Its configuration is:

```text
n_estimators = 400
max_depth = 15
min_samples_leaf = 1
random_state = 42
```

The final pipeline is serialized as:

```text
outputs/models/getaround_pricing_model.joblib
```

This artifact contains both preprocessing and prediction logic and is loaded directly by the production API.

---

## MLflow Experiment Tracking

MLflow is used to track the model-development experiments.

The experiment:

```text
getaround-pricing-model
```

contains four runs:

1. Linear Regression
2. Ridge Regression
3. Random Forest baseline
4. Tuned Random Forest

For each experiment, evaluation metrics are calculated from actual test-set predictions and logged to MLflow:

- MAE;
- RMSE;
- R².

Relevant model parameters are also logged.

The selected production `.joblib` model is stored as an artifact of the tuned Random Forest run.

MLflow uses a local SQLite tracking database:

```text
mlflow.db
```

Generated MLflow databases and artifact directories are excluded from Git through `.gitignore`.

### Run the Experiments

From the repository root:

```bash
python mlflow_tracking.py
```

This creates the local tracking database and records the four model experiments.

### Open the MLflow Interface

Run:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

The MLflow interface provides access to the experiment runs, model parameters, evaluation metrics, and logged production-model artifact.

---

## Pricing Prediction API

The selected model is exposed through a FastAPI application.

### Production URLs

API root:

```text
https://alexbelj-getaround-pricing-api.hf.space/
```

Interactive Swagger documentation:

```text
https://alexbelj-getaround-pricing-api.hf.space/docs
```

### Prediction Endpoint

```text
POST /predict
```

The endpoint accepts one vehicle description and returns the predicted daily rental price.

### Example Request

```json
{
  "model_key": "Citroën",
  "mileage": 120000,
  "engine_power": 110,
  "fuel": "diesel",
  "paint_color": "black",
  "car_type": "sedan",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": true,
  "automatic_car": false,
  "has_getaround_connect": true,
  "has_speed_regulator": true,
  "winter_tires": false
}
```

### Example Response

```json
{
  "predicted_rental_price_per_day": 108.92
}
```

The descriptive response field makes the API contract explicit and self-documenting.

### curl Example

```bash
curl -X POST \
  "https://alexbelj-getaround-pricing-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "Citroën",
    "mileage": 120000,
    "engine_power": 110,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": true,
    "has_gps": true,
    "has_air_conditioning": true,
    "automatic_car": false,
    "has_getaround_connect": true,
    "has_speed_regulator": true,
    "winter_tires": false
  }'
```

### Python Example

```python
import requests

url = "https://alexbelj-getaround-pricing-api.hf.space/predict"

payload = {
    "model_key": "Citroën",
    "mileage": 120000,
    "engine_power": 110,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": False,
}

response = requests.post(url, json=payload, timeout=30)

print(response.json())
```

FastAPI automatically exposes the interactive OpenAPI documentation at `/docs`, where the prediction endpoint can also be tested directly from a browser.

---

## Deployment Architecture

The project uses a simple production architecture:

```text
GitHub repository
      │
      ├── Streamlit dashboard
      │       │
      │       └── Docker → Hugging Face Space
      │
      └── FastAPI pricing service
              │
              ├── trained sklearn pipeline
              │
              └── Docker → Hugging Face Space
```

The **GitHub repository is the source of truth** for project code.

Hugging Face Spaces contain the deployment copies required to run each service.

The two services use separate runtime requirements so that each container installs only the dependencies it needs.

---

## Docker

Both production services are containerized independently.

The API container includes:

- FastAPI application;
- pricing-model artifact;
- API runtime dependencies.

The dashboard container includes:

- Streamlit application;
- delay dataset required at runtime;
- dashboard runtime dependencies.

Both Hugging Face services listen on port `7860`.

When building locally, Docker commands should be executed from the **repository root** because the Dockerfiles copy files from multiple project directories.

### Build the API Image

```bash
docker build -f api/Dockerfile -t getaround-pricing-api .
```

### Build the Dashboard Image

```bash
docker build -f dashboard/Dockerfile -t getaround-delay-dashboard .
```

---

## Reproducibility

### 1. Clone the Repository

```bash
git clone https://github.com/AlexandraBelj/5_Getaround_Analysis_Deployment.git
cd 5_Getaround_Analysis_Deployment
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it using the command appropriate for your operating system.

### 3. Install Project Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the Raw Datasets

Place the original Jedha datasets in:

```text
data/raw/get_around_delay_analysis.xlsx
data/raw/get_around_pricing_project.csv
```

The raw datasets are excluded from Git.

### 5. Run the Dashboard Locally

```bash
streamlit run dashboard/app.py
```

The production version is available at:

```text
https://alexbelj-getaround-delay-dashboard.hf.space/
```

### 6. Run the API Locally

```bash
uvicorn api.app:app --reload
```

The production API and documentation are available at:

```text
https://alexbelj-getaround-pricing-api.hf.space/
https://alexbelj-getaround-pricing-api.hf.space/docs
```

### 7. Reproduce MLflow Experiments

```bash
python mlflow_tracking.py
```

Then launch the MLflow UI:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and open:

```text
http://127.0.0.1:5000
```

---

## Limitations

The project has several important limitations.

### Delay Analysis

- checkout delays are missing for some completed rentals and are not imputed;
- consecutive-rental information is available only when the planned gap is below 12 hours;
- policy simulations describe how historical booking configurations would have been affected;
- potentially addressed cases are not proof that future incidents would be prevented;
- restricted bookings should not be interpreted automatically as lost bookings;
- Connect/mobile comparisons are descriptive rather than causal.

### Revenue Analysis

- the delay and pricing datasets do not share a common rental or vehicle identifier;
- exact historical owner revenue exposure therefore cannot be calculated;
- operational booking exposure is not equivalent to financial loss.

### Pricing Model

- model performance is evaluated on the supplied historical pricing dataset;
- the tuned Random Forest is selected only among the models tested;
- predictions should not be interpreted as causal estimates of how changing an individual vehicle feature would change its market price;
- production performance may change if the vehicle population or pricing environment changes.

---

## Conclusion

The project combines **business analysis and production deployment** around two Getaround use cases.

For rental delays, the analysis shows that late returns are frequent but actual next-driver interference is much less common. Minimum-gap policies can reduce historical interference exposure, but larger thresholds increasingly restrict scheduling flexibility.

A **60-minute minimum gap on Connect cars** is proposed as a practical starting point for product testing, with marketplace and revenue outcomes monitored during rollout.

For rental pricing, a tuned Random Forest achieves the strongest predictive performance among the tested models, with **MAE 10.59**, **RMSE 16.57**, and **R² 0.739**. The complete prediction pipeline is tracked with MLflow, serialized with joblib, exposed through FastAPI, containerized with Docker, and deployed on Hugging Face Spaces.

Together, the two workstreams demonstrate an end-to-end workflow from **data analysis and business decision support to machine-learning experimentation and production deployment**.

---

## Logo Attribution

The Getaround logo used in the dashboard is sourced from **Wikimedia Commons** and attributed to **Getaround Europe**, under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license.

Source: https://commons.wikimedia.org/

License: https://creativecommons.org/licenses/by-sa/4.0/