from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_SENSOR_COLS = [
    "P-PDG",
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
    "T-JUS-CKGL",
    "QGL",
]


def find_dataset_root(start_path: str | os.PathLike[str]) -> str | None:
    expected_items = set([str(i) for i in range(9)] + ["folds"])
    for root, dirs, files in os.walk(str(start_path)):
        if expected_items.issubset(set(dirs + files)):
            return root
    return None


def load_dataset(dataset_path: str | os.PathLike[str]) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if os.path.isdir(dataset_path):
        base_path = find_dataset_root(dataset_path)
        if base_path is None:
            raise FileNotFoundError("Could not locate the dataset root in the provided directory.")
        folds_path = os.path.join(base_path, "folds", "folds_clf_02.csv")
    else:
        base_path = str(Path(dataset_path).parent)
        folds_path = dataset_path

    folds_df = pd.read_csv(folds_path)
    all_instances: List[pd.DataFrame] = []

    for _, row in folds_df.iterrows():
        instance_path = row["instancia"]
        if not str(instance_path).endswith(".csv"):
            instance_path = f"{instance_path}.csv"
        full_path = os.path.join(base_path, instance_path)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            continue
        temp_df = pd.read_csv(full_path)
        temp_df["source_file"] = instance_path
        temp_df["fold"] = row["fold"]
        if "is_ova" in folds_df.columns:
            temp_df["is_ova"] = row["is_ova"]
        all_instances.append(temp_df)

    if not all_instances:
        raise ValueError("No dataset instances were loaded.")

    data = pd.concat(all_instances, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["source_file", "timestamp"]).reset_index(drop=True)
    data["time_diff_seconds"] = (
        data.groupby("source_file")["timestamp"].diff().dt.total_seconds()
    )
    return data


def build_features(data: pd.DataFrame, sensor_cols: List[str] | None = None) -> tuple[pd.DataFrame, List[str]]:
    sensor_cols = sensor_cols or DEFAULT_SENSOR_COLS
    sensor_cols = [col for col in sensor_cols if col in data.columns]

    usable_sensor_cols: List[str] = []
    for col in sensor_cols:
        non_missing_ratio = data[col].notna().mean()
        if non_missing_ratio > 0.20:
            usable_sensor_cols.append(col)

    for col in usable_sensor_cols:
        data[col] = data.groupby("source_file")[col].transform(lambda x: x.ffill().bfill())

    data = data.dropna(subset=["class"]).copy()
    data["target_abnormal"] = (data["class"] != 0).astype(int)

    for col in usable_sensor_cols:
        data[f"{col}_lag1"] = data.groupby("source_file")[col].shift(1)
        data[f"{col}_lag5"] = data.groupby("source_file")[col].shift(5)
        data[f"{col}_lag60"] = data.groupby("source_file")[col].shift(60)
        data[f"{col}_roll60_mean"] = data.groupby("source_file")[col].transform(
            lambda x: x.shift(1).rolling(window=60).mean()
        )
        data[f"{col}_roll60_std"] = data.groupby("source_file")[col].transform(
            lambda x: x.shift(1).rolling(window=60).std()
        )
        data[f"{col}_roll60_max"] = data.groupby("source_file")[col].transform(
            lambda x: x.shift(1).rolling(window=60).max()
        )

    engineered_feature_cols = [
        col for col in data.columns if col in usable_sensor_cols or "_lag" in col or "_roll" in col
    ]
    data_model = data.dropna(subset=["target_abnormal"]).copy()
    for col in engineered_feature_cols:
        data_model[col] = data_model[col].fillna(0)
    data_model = data_model.dropna(subset=engineered_feature_cols).copy()
    return data_model, usable_sensor_cols


def prepare_datasets(data: pd.DataFrame, test_fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    train_data = data[data["fold"] != test_fold].copy()
    test_data = data[data["fold"] == test_fold].copy()

    drop_cols = ["timestamp", "class", "target_abnormal", "source_file", "fold", "is_ova", "time_diff_seconds"]
    feature_cols = [col for col in train_data.columns if col not in drop_cols and train_data[col].dtype != "object"]

    X_train = train_data[feature_cols].astype("float32")
    y_train = train_data["target_abnormal"]
    X_test = test_data[feature_cols].astype("float32")
    y_test = test_data["target_abnormal"]
    return train_data, test_data, feature_cols, [X_train, y_train, X_test, y_test]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=10,
        max_samples=0.5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, thresholds: List[float] | None = None) -> Dict[str, Any]:
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = thresholds or [0.1 + i * 0.05 for i in range(17)]
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_abnormal": precision_score(y_test, y_pred, zero_division=0),
                "recall_abnormal": recall_score(y_test, y_pred, zero_division=0),
                "f1_abnormal": f1_score(y_test, y_pred, zero_division=0),
            }
        )
    return {"y_proba": y_proba, "threshold_results": pd.DataFrame(results)}


def run_pipeline(dataset_path: str, output_dir: str, test_fold: int = 0, chosen_threshold: float = 0.90) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path)
    data_model, usable_sensor_cols = build_features(data)
    _, test_data, feature_cols, split_data = prepare_datasets(data_model, test_fold=test_fold)
    X_train, y_train, X_test, y_test = split_data

    model = train_model(X_train, y_train)
    evaluation = evaluate_model(model, X_test, y_test)
    y_proba = evaluation["y_proba"]
    y_pred_tuned = (y_proba >= chosen_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "confusion_matrix": confusion_matrix(y_test, y_pred_tuned).tolist(),
        "classification_report": classification_report(y_test, y_pred_tuned, zero_division=0),
        "chosen_threshold": chosen_threshold,
    }

    alert_df = test_data.copy()
    alert_df["predicted_probability_abnormal"] = y_proba
    alert_df["predicted_label"] = y_pred_tuned
    alert_df["raw_alert"] = (alert_df["predicted_probability_abnormal"] >= chosen_threshold).astype(int)
    alert_df["persistent_alert"] = (
        alert_df.groupby("source_file")["raw_alert"].transform(lambda x: x.rolling(window=30, min_periods=30).sum())
    )
    alert_df["persistent_alert"] = (alert_df["persistent_alert"] >= 30).astype(int)

    alert_summary_df = pd.DataFrame([])
    for source_file, group in alert_df.groupby("source_file"):
        group = group.sort_values("timestamp")
        max_prob = group["predicted_probability_abnormal"].max()
        raw_alert_count = group["raw_alert"].sum()
        persistent_alert_count = group["persistent_alert"].sum()
        actual_file_abnormal = int(group["target_abnormal"].max())
        persistent_alert_rows = group[group["persistent_alert"] == 1]

        if len(persistent_alert_rows) > 0:
            status = "Persistent abnormal alert"
            predicted_file_alert = 1
        elif raw_alert_count > 0:
            status = "Raw warning only"
            predicted_file_alert = 1
        else:
            status = "Normal"
            predicted_file_alert = 0

        if actual_file_abnormal == 1 and predicted_file_alert == 1:
            alert_outcome = "True alert"
        elif actual_file_abnormal == 0 and predicted_file_alert == 1:
            alert_outcome = "False alert"
        elif actual_file_abnormal == 1 and predicted_file_alert == 0:
            alert_outcome = "Missed alert"
        else:
            alert_outcome = "Correct normal"

        alert_summary_df = pd.concat(
            [alert_summary_df, pd.DataFrame([{
                "source_file": source_file,
                "actual_file_abnormal": actual_file_abnormal,
                "predicted_file_alert": predicted_file_alert,
                "max_abnormal_probability": round(max_prob, 4),
                "raw_alert_count": int(raw_alert_count),
                "persistent_alert_count": int(persistent_alert_count),
                "status": status,
                "alert_outcome": alert_outcome,
            }])], ignore_index=True
        )

    joblib.dump(model, output_path / "model.joblib")
    with open(output_path / "feature_columns.json", "w", encoding="utf-8") as handle:
        json.dump(feature_cols, handle, indent=2)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    evaluation["threshold_results"].to_csv(output_path / "threshold_results.csv", index=False)
    alert_summary_df.to_csv(output_path / "alert_summary.csv", index=False)

    return {
        "metrics": metrics,
        "feature_cols": feature_cols,
        "usable_sensor_cols": usable_sensor_cols,
        "alert_summary": alert_summary_df,
        "model_path": str(output_path / "model.joblib"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an anomaly-detection model on industrial time-series data")
    parser.add_argument("--dataset", required=True, help="Path to the dataset root or folds CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save generated artifacts")
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    result = run_pipeline(args.dataset, args.output_dir, test_fold=args.test_fold, chosen_threshold=args.threshold)
    print(json.dumps({"metrics": result["metrics"], "feature_count": len(result["feature_cols"])}, indent=2))


if __name__ == "__main__":
    main()
