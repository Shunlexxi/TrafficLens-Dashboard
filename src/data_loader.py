"""
This module handles the source selection.
"""

from pathlib import Path

import pandas as pd

from src.validation import require_valid_dataset
from src.preprocessing import preprocess_dataset


# Change these settings when production data becomes available.
DATA_SOURCE = "synthetic"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATHS = {
    "synthetic": BASE_DIR / "data" / "synthetic" / "vehicle_detections.csv",
    "production": BASE_DIR / "data" / "production" / "vehicle_detections.csv",
}


def load_data():
    """Load, validate and preprocess vehicle detection data."""

    if DATA_SOURCE not in DATA_PATHS:
        raise ValueError(
            f"Unsupported data source: {DATA_SOURCE}"
        )

    file_path = DATA_PATHS[DATA_SOURCE]

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        ) from exc

    # Validate the raw dataset before processing.
    require_valid_dataset(df)

    # Prepare it for dashboard analysis.
    df = preprocess_dataset(df)

    return df
