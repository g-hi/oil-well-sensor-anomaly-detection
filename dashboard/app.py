import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Oil-Well Anomaly Monitoring", layout="wide")

st.title("Oil-Well Anomaly Monitoring Dashboard")
st.markdown(
    "A FastAPI-powered ML monitoring prototype for detecting abnormal oil-well sensor behavior."
)

BACKEND_DEFAULT = "http://127.0.0.1:8000"
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")
THRESHOLD_PATH = Path("models/chosen_threshold.json")

sensor_fields = [
    "P-PDG",
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
    "T-JUS-CKGL",
    "QGL",
]


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


feature_columns = load_json(FEATURE_COLUMNS_PATH)
threshold_payload = load_json(THRESHOLD_PATH)
threshold_value = None
if isinstance(threshold_payload, dict):
    threshold_value = threshold_payload.get("chosen_threshold")

backend_url = st.sidebar.text_input("Backend URL", value=BACKEND_DEFAULT)
if st.sidebar.button("Refresh backend status"):
    st.experimental_rerun()

backend_status = None
backend_status_text = "Offline"
backend_status_color = "error"
try:
    response = requests.get(f"{backend_url}/health", timeout=2)
    response.raise_for_status()
    backend_status = response.json()
    backend_status_text = "Online"
    backend_status_color = "success"
except requests.exceptions.RequestException:
    backend_status = None

main_status_cols = st.columns(4)
main_status_cols[0].metric("Backend Status", backend_status_text)
main_status_cols[1].metric(
    "Model Loaded",
    "Yes" if backend_status and backend_status.get("model_loaded") else "No",
)
main_status_cols[2].metric(
    "Threshold",
    f"{threshold_value:.2f}" if threshold_value is not None else "Unavailable",
)
main_status_cols[3].metric(
    "Number of Features",
    len(feature_columns) if isinstance(feature_columns, list) else "Unknown",
)

st.markdown("---")

single_tab, batch_tab, info_tab = st.tabs(["Single Prediction", "Batch CSV Analysis", "Model Information"])

with single_tab:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Single prediction inputs")
        raw_values: dict[str, float] = {}
        for field in sensor_fields:
            raw_values[field] = st.number_input(field, value=0.0, step=0.1, key=f"raw_{field}")

        with st.expander("Advanced: Engineered Features JSON", expanded=False):
            advanced_json = st.text_area(
                "Paste a JSON object with engineered lag and rolling-window features.",
                height=260,
                key="advanced_json",
            )
            st.caption(
                "Use this when you want to score the exact engineered features used by the trained model."
            )

        predict_button = st.button("Predict")

    prediction_response: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None

    if predict_button:
        if advanced_json:
            try:
                advanced_payload = json.loads(advanced_json)
                if not isinstance(advanced_payload, dict):
                    st.error("Advanced input must be a JSON object.")
                else:
                    payload = {"sensor_readings": advanced_payload}
            except json.JSONDecodeError as exc:
                st.error(f"JSON parse error: {exc}")
        else:
            payload = {"sensor_readings": raw_values}

        if payload is not None:
            if backend_status is None:
                st.error("FastAPI backend is not reachable. Verify the backend URL and refresh status.")
            else:
                try:
                    response = requests.post(f"{backend_url}/predict", json=payload, timeout=10)
                    response.raise_for_status()
                    prediction_response = response.json()
                except requests.exceptions.RequestException as exc:
                    st.error(f"Prediction request failed: {exc}")

    if prediction_response:
        with right_col:
            st.subheader("Prediction results")
            result_cols = st.columns(4)
            result_cols[0].metric("Prediction", prediction_response.get("prediction", "unknown"))
            result_cols[1].metric("Abnormal Probability", f"{prediction_response.get('abnormal_probability', 0):.3f}")
            result_cols[2].metric("Alert Status", prediction_response.get("alert_status", "unknown"))
            result_cols[3].metric("Threshold Used", f"{prediction_response.get('threshold_used', 0):.3f}")

            alert_status = prediction_response.get("alert_status", "normal")
            if alert_status == "normal":
                st.success(prediction_response.get("message", "Normal behavior."))
            elif alert_status == "watch":
                st.warning(prediction_response.get("message", "Watch this well."))
            else:
                st.error(prediction_response.get("message", "Abnormal behavior detected."))

            top_features = prediction_response.get("top_features", [])
            if top_features:
                st.subheader("Top contributing features")
                top_df = pd.DataFrame(top_features)
                st.table(top_df)
                if "importance" in top_df.columns:
                    st.bar_chart(top_df.set_index("feature")["importance"])
    else:
        with right_col:
            st.subheader("Prediction results")
            st.info("Enter values or paste engineered features, then press Predict.")

with batch_tab:
    st.subheader("Batch CSV Analysis")
    st.markdown(
        "This prototype expects engineered features matching `models/feature_columns.json`. "
        "In production, raw sensor streams would be converted automatically into lag and rolling-window features."
    )

    uploaded_file = st.file_uploader("Upload engineered feature CSV", type=["csv"], key="batch_csv")
    batch_results: pd.DataFrame | None = None

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown("**Uploaded data preview**")
            st.dataframe(data.head())

            if not isinstance(feature_columns, list):
                st.warning("Local feature_columns.json is unavailable, so column validation cannot be performed.")
            else:
                missing_columns = [col for col in feature_columns if col not in data.columns]
                if missing_columns:
                    st.warning(
                        "The uploaded CSV is missing the following engineered feature columns: "
                        + ", ".join(missing_columns)
                    )
                else:
                    if st.button("Run batch predictions"):
                        predictions = []
                        progress_bar = st.progress(0)
                        total = len(data)
                        for idx, row in data.iterrows():
                            payload = {"sensor_readings": row.to_dict()}
                            try:
                                response = requests.post(f"{backend_url}/predict", json=payload, timeout=10)
                                response.raise_for_status()
                                row_result = response.json()
                                predictions.append({**row.to_dict(), **row_result})
                            except requests.exceptions.RequestException as exc:
                                st.error(f"Row {idx} prediction failed: {exc}")
                                break
                            progress_bar.progress((idx + 1) / total)
                        if predictions:
                            batch_results = pd.DataFrame(predictions)

            if batch_results is not None:
                def summarize_top_features(top_features: list[dict[str, Any]] | None) -> str:
                    if not isinstance(top_features, list):
                        return ""
                    items = []
                    for feature in top_features:
                        name = feature.get("feature")
                        importance = feature.get("importance")
                        if name is None or importance is None:
                            continue
                        items.append(f"{name} ({importance:.3f})")
                    return ", ".join(items)

                st.markdown("**Batch prediction summary**")
                batch_results = batch_results.reset_index(drop=True)
                summary_df = pd.DataFrame(
                    {
                        "row_id": batch_results.index + 1,
                        "prediction": batch_results.get("prediction"),
                        "abnormal_probability": batch_results.get("abnormal_probability"),
                        "threshold_used": batch_results.get("threshold_used"),
                        "alert_status": batch_results.get("alert_status"),
                        "message": batch_results.get("message"),
                        "top_features_summary": batch_results.get("top_features").apply(summarize_top_features),
                    }
                )

                metrics_cols = st.columns(3)
                metrics_cols[0].metric("Total rows", len(summary_df))
                metrics_cols[1].metric("Normal count", int((summary_df["prediction"] == "normal").sum()))
                metrics_cols[2].metric("Abnormal count", int((summary_df["prediction"] == "abnormal").sum()))
                if "alert_status" in summary_df.columns:
                    st.metric("Alert count", int((summary_df["alert_status"] == "alert").sum()))

                st.dataframe(
                    summary_df[
                        [
                            "row_id",
                            "prediction",
                            "abnormal_probability",
                            "threshold_used",
                            "alert_status",
                            "message",
                            "top_features_summary",
                        ]
                    ],
                    use_container_width=True,
                )

                with st.expander("View full detailed results"):
                    st.dataframe(batch_results, use_container_width=True)

                if "abnormal_probability" in batch_results.columns:
                    st.line_chart(batch_results["abnormal_probability"])
        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")

with info_tab:
    st.subheader("Model information")
    st.markdown(
        """
- Model type: Random Forest Classifier
- Training environment: Google Colab
- Serving layer: FastAPI
- UI layer: Streamlit
- Threshold source: `chosen_threshold.json`
- Feature source: `feature_columns.json`
- Explainability: feature importance and SHAP-ready
        """
    )
    st.markdown("---")
    st.subheader("Notes")
    st.markdown(
        """
- The model is trained on engineered time-series features: raw sensor values are converted into lag and rolling-window statistics.
- The Single Prediction tab accepts raw sensor values for simple testing, while the advanced section accepts fully engineered feature JSON.
- The Batch CSV Analysis tab expects a CSV with the same engineered feature columns used during training.
        """
    )
