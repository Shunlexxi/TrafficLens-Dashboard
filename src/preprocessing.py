
import pandas as pd


def preprocess_dataset(df):
    """Prepare validated detection records for analysis."""

    df = df.copy()

    # Convert timestamps to datetime.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Invalid or missing timestamps detected.")

    # The CSV stores local Lagos wall-clock time.
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            "Africa/Lagos"
        )
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(
            "Africa/Lagos"
        )

    # Convert confidence columns to numeric.
    confidence_columns = [
        "detection_confidence",
        "ocr_confidence",
        "plate_type_confidence",
    ]

    for column in confidence_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Administrative area mapping
    # --------------------------------------------------------

    admin_mapping = {
        "Abeokuta": ("Abeokuta South", "LGA"),
        "Ibadan": ("Ibadan North", "LGA"),
        "Jos": ("Jos North", "LGA"),
        "Benin City": ("Oredo", "LGA"),
        "Enugu": ("Enugu North", "LGA"),
        "Akure": ("Akure South", "LGA"),
        "Ilorin": ("Ilorin West", "LGA"),
        "Port Harcourt": ("Port Harcourt City", "LGA"),
        "Abuja Municipal": ("Abuja Municipal", "Area Council"),
        "Gwagwalada": ("Gwagwalada", "Area Council"),
        "Kano Municipal": ("Kano Municipal", "LGA"),
        "Ikeja": ("Ikeja", "LGA"),
        "Badagry": ("Badagry", "LGA"),
        "Epe": ("Epe", "LGA"),
        "Lagos Island": ("Lagos Island", "LGA"),
    }

    # Map existing registration labels to administrative areas.
    df["registration_admin_area"] = (
        df["registration_lga"]
        .map(lambda value: admin_mapping.get(value, (None, None))[0])
    )

    # Identify whether the administrative unit is an LGA or Area Council.
    df["registration_admin_type"] = (
        df["registration_lga"]
        .map(lambda value: admin_mapping.get(value, (None, None))[1])
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["day_type"] = df["day_of_week"].apply(
        lambda day: "Weekend" if day >= 5 else "Weekday"
    )

    # Useful for temporal trend analysis.
    df["year_month"] = df["timestamp"].dt.strftime("%Y-%m")

    return df
