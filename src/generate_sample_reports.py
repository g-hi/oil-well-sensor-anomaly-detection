from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_sample_reports(output_dir: str | Path | None = None) -> None:
    output_dir = Path(output_dir or "reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [0.91, 0.88, 0.89, 0.88],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df["Metric"], metrics_df["Value"], color="#4C78A8")
    ax.set_title("Sample Model Evaluation Metrics")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    plt.tight_layout()
    fig.savefig(output_dir / "sample_evaluation_metrics.png", dpi=200)
    plt.close(fig)

    sample_alerts = pd.DataFrame(
        {
            "Time": ["T1", "T2", "T3", "T4"],
            "Alert": [0, 1, 1, 0],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sample_alerts["Time"], sample_alerts["Alert"], marker="o", color="#F58518")
    ax.set_title("Sample Operational Alert Timeline")
    ax.set_ylabel("Alert Flag")
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    fig.savefig(output_dir / "sample_alert_timeline.png", dpi=200)
    plt.close(fig)

    summary_text = """Sample report summary:\nThis repository includes a reproducible workflow for anomaly detection in oil well time-series data.\nThe current example demonstrates the structure of the evaluation and alerting outputs."""
    (output_dir / "sample_report_summary.txt").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    generate_sample_reports()
