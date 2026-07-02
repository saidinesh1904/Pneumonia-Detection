
import json
import os
from typing import Dict

import pandas as pd

from configs.config import (
    METRICS_DIR,
    METRICS_PATH,
    CLASSIFICATION_REPORT_PATH,
    HISTORY_PATH,
    BENCHMARK_PATH,
)


def create_metrics_directory() -> None:
    os.makedirs(
        METRICS_DIR,
        exist_ok=True,
    )

def save_metrics(
    metrics: Dict,
    filepath: str = METRICS_PATH,
) -> None:


    create_metrics_directory()

    with open(
        filepath,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Metrics saved to {filepath}"
    )


def save_classification_report(
    report: str,
    filepath: str = CLASSIFICATION_REPORT_PATH,
) -> None:


    create_metrics_directory()

    with open(
        filepath,
        "w",
    ) as file:

        file.write(report)

    print(
        f"Classification report saved to {filepath}"
    )


def save_training_history(
    history: Dict,
    filepath: str = HISTORY_PATH,
) -> None:


    history_df = pd.DataFrame(
        history
    )

    history_df.index += 1

    history_df.index.name = "epoch"

    history_df.to_csv(
        filepath,
        index=True,
    )

    print(
        f"Training history saved to {filepath}"
    )


def save_benchmark(
    benchmark: Dict,
    filepath: str = BENCHMARK_PATH,
) -> None:


    create_metrics_directory()

    with open(
        filepath,
        "w",
    ) as file:

        json.dump(
            benchmark,
            file,
            indent=4,
        )

    print(
        f"Benchmark saved to {filepath}"
    )


def print_metrics(
    metrics: Dict,
) -> None:


    print("\n" + "=" * 60)

    print("Evaluation Metrics")

    print("=" * 60)

    for key, value in metrics.items():

        print(
            f"{key:15s}: {value:.4f}"
        )

    print("=" * 60)