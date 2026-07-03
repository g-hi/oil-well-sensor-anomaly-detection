from pathlib import Path

import pandas as pd

from src.pipeline import build_features, load_dataset


def test_build_features_creates_target(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"]),
            "source_file": ["a.csv", "a.csv"],
            "fold": [0, 0],
            "class": [0, 1],
            "P-PDG": [10.0, 10.5],
            "P-TPT": [1.0, 1.1],
        }
    )

    data_model, usable_sensor_cols = build_features(data, sensor_cols=["P-PDG", "P-TPT"])

    assert "target_abnormal" in data_model.columns
    assert data_model["target_abnormal"].iloc[0] == 0
    assert data_model["target_abnormal"].iloc[1] == 1
    assert "P-PDG_lag1" in data_model.columns
    assert usable_sensor_cols == ["P-PDG", "P-TPT"]


def test_load_dataset_requires_existing_path():
    missing_path = Path("does_not_exist")
    try:
        load_dataset(missing_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
