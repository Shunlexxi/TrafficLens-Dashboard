
import pandas as pd

EXPECTED_COLUMNS = [
    "detection_id",
    "timestamp",
    "camera_id",
    "location",
    "plate_number",
    "registration_state",
    "registration_lga",
    "plate_type",
    "car_brand",
    "vehicle_category",
    "detection_confidence",
    "ocr_confidence",
    "plate_type_confidence",
    "image_path",
    "vehicle_hash",
    "processing_status",
    "data_source",
]

CAMERA_LOCATIONS = {
    "CAM_001": "Ikeja, Lagos",
    "CAM_002": "Lekki, Lagos",
    "CAM_003": "Yaba, Lagos",
    "CAM_004": "Victoria Island, Lagos",
    "CAM_005": "Wuse, FCT",
}

VALID_STATUSES = {"success", "partial", "failed"}

CONFIDENCE_COLUMNS = [
    "detection_confidence",
    "ocr_confidence",
    "plate_type_confidence",
]

VEHICLE_ATTRIBUTES = [
    "plate_number",
    "registration_state",
    "registration_lga",
    "plate_type",
    "car_brand",
    "vehicle_category",
]


def validate_dataset(df):
    """Validate the structure and basic integrity of a detection dataset."""

    errors = []

    # Check required columns and their order.
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(
            "Column names or order do not match the agreed 17-column schema."
        )
        return errors

    # Check event identifiers.
    if df["detection_id"].isna().any():
        errors.append("Some detection IDs are missing.")

    if df["detection_id"].duplicated().any():
        errors.append("Duplicate detection IDs found.")

    # Check processing statuses.
    invalid_statuses = set(
        df["processing_status"].dropna().unique()
    ) - VALID_STATUSES

    if invalid_statuses:
        errors.append(f"Invalid processing statuses: {invalid_statuses}")

    if df["processing_status"].isna().any():
        errors.append("Some processing statuses are missing.")

    # Check camera IDs and camera locations.
    for camera_id, location in CAMERA_LOCATIONS.items():
        camera_rows = df[df["camera_id"] == camera_id]

        if not camera_rows.empty:
            if not camera_rows["location"].eq(location).all():
                errors.append(
                    f"Location mismatch detected for {camera_id}."
                )

    unknown_cameras = set(df["camera_id"].dropna()) - set(CAMERA_LOCATIONS)

    if unknown_cameras:
        errors.append(f"Unknown camera IDs: {unknown_cameras}")

    # Check plate-number format.
    plate_values = df["plate_number"].dropna().astype(str)

    invalid_plates = ~plate_values.str.fullmatch(
        r"[A-Z]{3}-\d{3}-[A-Z]{2}"
    )

    if invalid_plates.any():
        errors.append(
            f"{invalid_plates.sum()} plate numbers have invalid formats."
        )

    # Failed events should not have OCR plate outputs.
    failed = df["processing_status"] == "failed"

    if df.loc[failed, "plate_number"].notna().any():
        errors.append(
            "Failed events contain plate numbers."
        )

    # Check confidence score ranges.
    for column in CONFIDENCE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")

        if (values.dropna() < 0).any() or (values.dropna() > 1).any():
            errors.append(
                f"{column} contains values outside the 0–1 range."
            )

    # Check repeat-vehicle consistency.
    for column in VEHICLE_ATTRIBUTES:
        inconsistent = (
            df.groupby("vehicle_hash")[column]
            .nunique(dropna=True)
            .gt(1)
        )

        if inconsistent.any():
            errors.append(
                f"Inconsistent {column} values across repeated vehicle IDs."
            )

    # Check data source.
    if df["data_source"].isna().any():
        errors.append("Some data_source values are missing.")

    return errors


def require_valid_dataset(df):
    """Raise an exception if the dataset fails validation."""

    errors = validate_dataset(df)

    if errors:
        raise ValueError(
            "Dataset validation failed:\n- " + "\n- ".join(errors)
        )

    return True
