# Industrial Time-Series Anomaly Detection for Oil Wells

This project demonstrates how machine learning can support predictive monitoring in industrial oil and gas operations by detecting abnormal behavior in oil-well sensor data to assist engineering decision-making.

The solution combines time-series preprocessing, temporal feature engineering, supervised classification, threshold-based alerting, explainability, FastAPI model serving, and a Streamlit monitoring dashboard.

## Why this project matters

Oil wells generate continuous sensor readings such as pressure, temperature, and gas-lift flow measurements. Abnormal behavior in these signals can indicate equipment issues, production instability, maintenance needs, or safety risks.

Early detection of abnormal operating conditions can help reduce downtime, improve safety, support preventive maintenance, and improve operational decision-making.

This project focuses on building a practical anomaly-detection workflow that connects:

- sensor data preparation
- time-series feature engineering
- model training and evaluation
- threshold-based alerting
- explainability
- API-based inference
- dashboard-based monitoring

## What I implemented

- Loaded and structured industrial time-series data from multiple oil-well sensor files
- Treated each well/source file as a separate time-series instance to reduce data leakage
- Cleaned and prepared sensor measurements for modeling
- Handled missing values using forward-fill and backward-fill strategies within each well file
- Engineered temporal features such as lag values and rolling-window statistics
- Used fold-based evaluation instead of random row splitting
- Trained a Random Forest classifier to detect abnormal operating states
- Tuned decision thresholds to balance false alarms and missed abnormal events
- Added persistent alert logic to reduce noisy single-point alerts
- Used feature importance and SHAP-based explainability for model interpretation
- Exported trained model artifacts from Colab for serving
- Packaged the trained model behind a FastAPI inference endpoint
- Added Swagger documentation for API testing
- Built a Streamlit dashboard for single prediction and batch CSV analysis

## Project highlights

- Industrial time-series anomaly detection
- Oil-well sensor monitoring use case
- Leakage-aware preprocessing
- Lag and rolling-window feature engineering
- Fold-based model evaluation
- Binary classification for normal vs abnormal behavior
- Threshold tuning for practical alert decisions
- Persistent alert logic to reduce noisy alerts
- Feature importance and SHAP explainability
- FastAPI model-serving layer
- Swagger API documentation
- Streamlit monitoring dashboard
- Batch CSV prediction support
- Clean repository structure for review and extension

## Repository structure

```text
industrial-time-series-anomaly-detection-3w/
├── app/
│   ├── main.py
│   ├── model_service.py
│   ├── schemas.py
│   ├── alerting.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── models/
│   ├── rf_3w_abnormal_detector.joblib
│   ├── chosen_threshold.json
│   ├── feature_columns.json
│   └── sample CSV/reporting artifacts
├── notebooks/
├── reports/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```
Dataset

The project was developed using the Petrobras 3W dataset for oil-well abnormal event detection. Place the dataset locally and point the training pipeline to the dataset path.

Sensor variables

The base sensor variables include pressure, temperature, and gas-lift flow measurements:

Feature	Description
P-PDG	Pressure at the Permanent Downhole Gauge
P-TPT	Pressure at the Temperature and Pressure Transducer
T-TPT	Temperature at the Temperature and Pressure Transducer
P-MON-CKP	Pressure around the production choke/manifold
T-JUS-CKP	Temperature downstream of the production choke
P-JUS-CKGL	Pressure downstream of the gas-lift choke
T-JUS-CKGL	Temperature downstream of the gas-lift choke
QGL	Gas-lift flow rate

From these raw sensor readings, the pipeline creates lag and rolling-window features to capture temporal behavior.

Setup
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
How to run the pipeline

Run the training pipeline with a dataset root or a folds CSV file:

python -m src.pipeline --dataset path/to/dataset_root --output-dir outputs

Model outputs and evaluation reports are saved to the selected output directory.

Model artifacts

The trained model artifacts are exported from the Colab training workflow and placed in the models/ directory.

Expected files:

models/
├── rf_3w_abnormal_detector.joblib
├── chosen_threshold.json
├── feature_columns.json
├── threshold_tuning_results.csv
├── operational_alert_summary.csv
└── sample batch/reporting CSV artifacts

Key files used by the API:

rf_3w_abnormal_detector.joblib — trained Random Forest model
chosen_threshold.json — JSON file containing the selected decision threshold using the key chosen_threshold
feature_columns.json — ordered list of feature names used during training

The API loads these artifacts and enforces the same feature order during inference.

API inference layer

A FastAPI inference layer is included to serve the trained anomaly detection model.

The API:

loads the trained Random Forest model
loads the selected threshold
loads the required feature columns
validates incoming sensor features
enforces feature ordering
calculates abnormal probability
applies threshold-based classification
returns prediction, alert status, and top model features
Run the API
uvicorn app.main:app --reload
Open Swagger documentation
http://127.0.0.1:8000/docs
Health endpoint
GET /health

The health endpoint confirms whether the model, threshold, and feature columns are loaded.

Prediction endpoint
POST /predict

The current prototype expects engineered feature values matching models/feature_columns.json.

Example request:

{
  "sensor_readings": {
    "P-PDG": 120.0,
    "P-TPT": 70.0,
    "T-TPT": 85.0,
    "P-MON-CKP": 40.0,
    "T-JUS-CKP": 50.0,
    "P-JUS-CKGL": 30.0,
    "T-JUS-CKGL": 60.0,
    "QGL": 8.0,
    "P-PDG_lag1": 118.0,
    "P-PDG_lag5": 115.0,
    "P-PDG_lag60": 110.0,
    "P-PDG_roll60_mean": 111.0,
    "P-PDG_roll60_std": 3.2,
    "P-PDG_roll60_max": 130.0
  }
}

Note: the full request should include all engineered features listed in models/feature_columns.json. The shortened example above is for readability.

Example response:

{
  "prediction": "normal",
  "abnormal_probability": 0.36,
  "threshold_used": 0.9,
  "alert_status": "normal",
  "top_features": [
    {
      "feature": "T-TPT_lag5",
      "importance": 0.118389
    },
    {
      "feature": "T-TPT",
      "importance": 0.1179
    },
    {
      "feature": "T-TPT_lag1",
      "importance": 0.08969
    }
  ],
  "message": "Sensor behavior appears normal based on the current threshold."
}
Dashboard

A Streamlit dashboard is included as a lightweight monitoring interface on top of the FastAPI model-serving endpoint.

The dashboard includes:

backend/model status overview
single prediction workflow
batch CSV analysis
model information tab
abnormal probability output
alert status display
top feature importance display
clean summary table for batch predictions
Run the dashboard

Start the FastAPI backend first:

uvicorn app.main:app --reload

Then run the dashboard:

streamlit run dashboard/app.py

The dashboard assumes the FastAPI backend is running at:

http://127.0.0.1:8000
CSV upload format

The current batch prediction prototype expects engineered features matching the model’s training feature list.

Required feature columns are stored in:

models/feature_columns.json

Uploaded CSV files should contain these engineered features, including:

raw sensor readings
lag features
rolling-window mean features
rolling-window standard deviation features
rolling-window maximum features

In a production deployment, raw sensor streams would be accepted and the backend would automatically generate lag and rolling-window features before inference.

Testing and validation

The solution was tested at multiple levels:

data preprocessing checks
missing-value handling validation
feature engineering consistency
fold-based model evaluation
threshold tuning comparison
alert summary analysis
FastAPI /health endpoint validation
FastAPI /predict endpoint testing through Swagger
dashboard single-prediction testing
batch CSV prediction testing
Engineering focus

This is a practical anomaly-detection pipeline that connects sensor data processing, model training, alert generation, explainability, API serving, and dashboard-based monitoring.

Key engineering aspects include:

treating each well/source file as a separate time-series instance to reduce data leakage
creating lag and rolling-window features to capture temporal behavior
using fold-based evaluation instead of random row splitting
tuning the decision threshold to balance false alarms and missed abnormal events
adding persistent alert logic to reduce noisy single-point alerts
using feature importance and SHAP to explain which sensor patterns contributed to abnormal predictions
exporting trained artifacts from Colab for model serving
enforcing feature consistency using feature_columns.json
exposing inference through FastAPI and Swagger
adding a Streamlit dashboard for user-facing monitoring
Future improvements
Add automatic feature generation from raw sensor streams inside the backend
Compare the Random Forest baseline with XGBoost or LightGBM
Add sequence-based models such as LSTM, GRU, or temporal CNNs
Add monitoring for data drift, missing sensors, and model performance degradation
Extend the binary normal-vs-abnormal task into multiclass event classification
Deploy API and dashboard to a cloud environment
Add authentication and role-based access control for enterprise use