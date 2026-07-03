# Industrial Time-Series Anomaly Detection for Oil Wells

This project demonstrates how machine learning can support predictive monitoring in industrial operations by detecting abnormal behavior in oil well sensor data. The workflow combines time-series preprocessing, feature engineering, supervised classification, and threshold-based alert logic to identify abnormal conditions that may warrant attention.

## Why this project matters

Early detection of abnormal operating conditions can help reduce equipment failure, improve safety, and support better operational decision-making. In this project, I focused on building a practical anomaly-detection workflow that could be explained clearly in an interview setting.

## What I implemented

- Loaded and structured industrial time-series data from multiple sensor files
- Cleaned and prepared sensor measurements for modeling
- Engineered temporal features such as lag values and rolling statistics
- Trained a Random Forest classifier to predict abnormal states
- Tuned decision thresholds and built a simple operational alerting logic
- Organized the workflow into a reproducible Python project structure

## Project highlights

- Time-series preprocessing for industrial sensor data
- Feature engineering for temporal behavior
- Binary classification for abnormal vs normal conditions
- Threshold tuning for practical alerting decisions
- A clean repository layout suitable for GitHub and interviews

## Repository structure

```text
industrial-time-series-anomaly-detection-3w/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
├── reports/
├── models/
├── src/
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run the pipeline

Run the training pipeline with a dataset root or a folds CSV file:

```bash
python -m src.pipeline --dataset path/to/dataset_root --output-dir outputs
```

## Notes

- The dataset is not included in this repository.
- Place your dataset locally and point the script to it.
- Model outputs and evaluation reports are saved to the output directory.

## Interview talking points

- This project shows end-to-end machine learning work, not only model training.
- It highlights practical thinking around industrial monitoring and operational alerts.
- It demonstrates how data science can be connected to real-world decision support.

## Next steps

- Add a more detailed notebook walkthrough
- Compare Random Forest with gradient boosting or temporal models
- Add explainability tools such as SHAP and feature-importance plots

