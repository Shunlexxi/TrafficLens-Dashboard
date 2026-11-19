import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OOH Vehicle Analytics",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CONSTANTS AND CONFIGURATION
# ============================================================

APP_TITLE = "Vehicle Detection Analytics"
APP_SUBTITLE = "Nigerian Out-of-Home Intelligence Dashboard"

COLORS = {
    "primary": "#2563EB",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "secondary": "#64748B"
}


# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading vehicle detection data...")
def get_data():
    return load_data()


try:
    df = get_data()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def show_empty_message():
    st.info("No records match the selected filters.")


def format_number(value):
    return f"{value:,.0f}"


def show_kpi(label, value, help_text=None):
    st.metric(
        label=label,
        value=value,
        help=help_text
    )


def chart_has_data(data, column):
    return (
        not data.empty
        and column in data.columns
        and data[column].notna().any()
    )


# ============================================================
# 5. SIDEBAR AND GLOBAL FILTERS
# ============================================================

st.sidebar.title("🚘 OOH Analytics")
st.sidebar.caption("Vehicle Detection Intelligence")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Traffic Analytics",
        "Geographic Analytics",
        "Vehicle Analytics",
        "Camera Analytics",
        "Recognition Performance"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Global Filters")

# Date filter
minimum_date = df["timestamp"].min().date()
maximum_date = df["timestamp"].max().date()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date
)

# Handle incomplete date-range selection
if isinstance(selected_dates, tuple):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = selected_dates[0]
        end_date = selected_dates[0]
else:
    start_date = selected_dates
    end_date = selected_dates


# Camera filter
camera_options = sorted(df["camera_id"].dropna().unique())

selected_cameras = st.sidebar.multiselect(
    "Camera",
    options=camera_options,
    default=camera_options
)


# Registration state filter
state_options = sorted(
    df["registration_state"].dropna().unique()
)

selected_states = st.sidebar.multiselect(
    "Registration state",
    options=state_options,
    default=state_options
)


# Processing status filter
status_options = sorted(
    df["processing_status"].dropna().unique()
)

selected_statuses = st.sidebar.multiselect(
    "Processing status",
    options=status_options,
    default=status_options
)


# ============================================================
# 6. APPLY GLOBAL FILTERS
# ============================================================

filtered_df = df.copy()

filtered_df = filtered_df[
    (filtered_df["timestamp"].dt.date >= start_date)
    & (filtered_df["timestamp"].dt.date <= end_date)
]

filtered_df = filtered_df[
    filtered_df["camera_id"].isin(selected_cameras)
]

filtered_df = filtered_df[
    filtered_df["registration_state"].isin(selected_states)
    | filtered_df["registration_state"].isna()
]

filtered_df = filtered_df[
    filtered_df["processing_status"].isin(selected_statuses)
]


# ============================================================
# 7. HEADER
# ============================================================

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.caption(
    f"Showing data from {start_date.strftime('%d %b %Y')} "
    f"to {end_date.strftime('%d %b %Y')}"
)

st.divider()


# ============================================================
# 8. OVERVIEW KPI FUNCTION
# ============================================================

def display_overview_kpis(data):

    total_detections = len(data)

    unique_vehicles = data["vehicle_hash"].nunique()

    repeat_detections = max(
        total_detections - unique_vehicles,
        0
    )

    active_cameras = data["camera_id"].nunique()

    successful_events = (
        data["processing_status"] == "success"
    ).sum()

    success_rate = (
        successful_events / total_detections * 100
        if total_detections > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_kpi(
            "Detection Events",
            format_number(total_detections),
            "Total recorded detection events"
        )

    with col2:
        show_kpi(
            "Unique Vehicles",
            format_number(unique_vehicles),
            "Distinct vehicle identifiers"
        )

    with col3:
        show_kpi(
            "Active Cameras",
            format_number(active_cameras),
            "Cameras represented in filtered data"
        )

    with col4:
        show_kpi(
            "Processing Success Rate",
            f"{success_rate:.2f}%",
            "Successful events divided by all events"
        )

    st.write("")

    col5, col6 = st.columns(2)

    with col5:
        show_kpi(
            "Repeat Detection Events",
            format_number(repeat_detections),
            "Events beyond the first observation of each vehicle"
        )

    with col6:
        average_detections = (
            total_detections / unique_vehicles
            if unique_vehicles > 0
            else 0
        )

        show_kpi(
            "Average Detections per Vehicle",
            f"{average_detections:.2f}"
        )


# ============================================================
# 9. OVERVIEW PAGE
# ============================================================

def overview_page(data):

    st.subheader("Overview")

    display_overview_kpis(data)

    st.divider()

    if data.empty:
        show_empty_message()
        return

    col1, col2 = st.columns(2)

    # Daily detection trend
    with col1:

        st.markdown("#### Daily Detection Trend")

        daily_data = (
            data.groupby("date")
            .size()
            .reset_index(name="detections")
        )

        fig = px.line(
            daily_data,
            x="date",
            y="detections",
            markers=True,
            labels={
                "date": "Date",
                "detections": "Detection Events"
            }
        )

        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Camera distribution
    with col2:

        st.markdown("#### Detection Distribution by Camera")

        camera_data = (
            data.groupby("camera_id")
            .size()
            .reset_index(name="detections")
        )

        fig = px.bar(
            camera_data,
            x="camera_id",
            y="detections",
            color="camera_id",
            labels={
                "camera_id": "Camera",
                "detections": "Detection Events"
            }
        )

        fig.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 10. TRAFFIC ANALYTICS PAGE
# ============================================================

def traffic_analytics_page(data):

    st.subheader("Traffic Analytics")

    if data.empty:
        show_empty_message()
        return

    st.markdown("#### Weekday vs Weekend")
    
    day_data = (
            data.groupby("day_type")
            .agg(
                detections=("detection_id", "count"),
                observed_days=("date", "nunique")
            )
            .reset_index()
        )
    
    day_data["average_per_day"] = (
            day_data["detections"]
            / day_data["observed_days"].replace(0, pd.NA)
        )
    
    fig = px.bar(
            day_data,
            x="day_type",
            y="average_per_day",
            color="day_type",
            labels={
                "day_type": "Day Type",
                "average_per_day": "Average Detections per Day"
            }
        )
    
    fig.update_layout(
            height=350,
            showlegend=False
        )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 11. GEOGRAPHIC ANALYTICS PAGE
# ============================================================

def geographic_analytics_page(data):

    st.subheader("Geographic Analytics")

    if data.empty:
        show_empty_message()
        return

    st.caption(
        "Registration charts describe the vehicle's recorded "
        "registration origin, not where it was detected. "
        "Camera locations show where detection events occurred."
    )

    # --------------------------------------------------------
    # 1. REGISTRATION BY STATE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Registration by State")

        state_data = (
            data.dropna(subset=["registration_state"])
            .groupby("registration_state")
            .size()
            .reset_index(name="detections")
            .sort_values("detections", ascending=False)
        )

        if state_data.empty:
            st.info("No registration-state information available.")
        else:
            fig = px.bar(
                state_data,
                x="registration_state",
                y="detections",
                labels={
                    "registration_state": "Registration State",
                    "detections": "Detection Events"
                }
            )

            fig.update_layout(height=420)

            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # 2. REGISTRATION BY ADMINISTRATIVE AREA
    # --------------------------------------------------------

    with col2:

        st.markdown("#### Registration by Administrative Area")

        admin_columns = [
            "registration_admin_area",
            "registration_admin_type"
        ]

        if not all(column in data.columns for column in admin_columns):

            st.info(
                "Administrative-area fields are not available "
                "in this dataset."
            )

        else:

            admin_data = (
                data.dropna(subset=["registration_admin_area"])
                .groupby(
                    [
                        "registration_admin_area",
                        "registration_admin_type"
                    ]
                )
                .size()
                .reset_index(name="detections")
                .sort_values("detections", ascending=False)
            )

            if admin_data.empty:
                st.info("No administrative-area information available.")
            else:
                fig = px.bar(
                    admin_data,
                    x="detections",
                    y="registration_admin_area",
                    color="registration_admin_type",
                    orientation="h",
                    labels={
                        "registration_admin_area": "Administrative Area",
                        "detections": "Detection Events",
                        "registration_admin_type": "Administrative Type"
                    },
                    hover_data=["registration_admin_type"]
                )

                fig.update_layout(
                    height=500,
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------------
    # 3. ADMINISTRATIVE AREA TYPE DISTRIBUTION
    # --------------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("#### Administrative Area Type")

        if "registration_admin_type" not in data.columns:

            st.info("Administrative-type information is unavailable.")

        else:

            type_data = (
                data.dropna(subset=["registration_admin_type"])
                .groupby("registration_admin_type")
                .size()
                .reset_index(name="detections")
                .sort_values("detections", ascending=False)
            )

            if type_data.empty:
                st.info("No administrative-type information available.")
            else:
                fig = px.pie(
                    type_data,
                    names="registration_admin_type",
                    values="detections",
                    hole=0.45,
                    labels={
                        "registration_admin_type": "Administrative Type",
                        "detections": "Detection Events"
                    }
                )

                fig.update_layout(height=420)

                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # 4. DETECTIONS BY CAMERA OBSERVATION LOCATION
    # --------------------------------------------------------

    with col4:

        st.markdown("#### Detections by Camera Location")

        if "location" not in data.columns:

            st.info("Camera-location information is unavailable.")

        else:

            location_data = (
                data.dropna(subset=["location"])
                .groupby("location")
                .size()
                .reset_index(name="detections")
                .sort_values("detections", ascending=False)
            )

            if location_data.empty:
                st.info("No camera-location information available.")
            else:
                fig = px.bar(
                    location_data,
                    x="location",
                    y="detections",
                    labels={
                        "location": "Camera Observation Location",
                        "detections": "Detection Events"
                    }
                )

                fig.update_layout(
                    height=420,
                    xaxis_tickangle=-30
                )

                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 12. VEHICLE ANALYTICS PAGE
# ============================================================

def vehicle_analytics_page(data):

    st.subheader("Vehicle Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Vehicle Manufacturers")

        brand_data = (
            data.dropna(subset=["car_brand"])
            .groupby("car_brand")
            .size()
            .reset_index(name="detections")
            .sort_values("detections", ascending=False)
            .head(10)
        )

        if brand_data.empty:
            st.info("No vehicle manufacturer information available.")
        else:
            fig = px.bar(
                brand_data,
                x="detections",
                y="car_brand",
                orientation="h",
                labels={
                    "car_brand": "Manufacturer",
                    "detections": "Detection Events"
                }
            )

            fig.update_layout(
                height=450,
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("#### Vehicle Categories")

        category_data = (
            data.dropna(subset=["vehicle_category"])
            .groupby("vehicle_category")
            .size()
            .reset_index(name="detections")
        )

        if category_data.empty:
            st.info("No vehicle category information available.")
        else:
            fig = px.pie(
                category_data,
                names="vehicle_category",
                values="detections",
                hole=0.45
            )

            fig.update_layout(height=450)

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("#### Plate Type Distribution")

    plate_data = (
        data.dropna(subset=["plate_type"])
        .groupby("plate_type")
        .size()
        .reset_index(name="detections")
    )

    if plate_data.empty:
        st.info("No plate-type information available.")
    else:
        fig = px.bar(
            plate_data,
            x="plate_type",
            y="detections",
            color="plate_type",
            labels={
                "plate_type": "Plate Type",
                "detections": "Detection Events"
            }
        )

        fig.update_layout(
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 13. CAMERA ANALYTICS PAGE
# ============================================================

def camera_analytics_page(data):

    st.subheader("Camera Analytics")

    if data.empty:
        show_empty_message()
        return

    camera_data = (
        data.groupby(["camera_id", "location"])
        .size()
        .reset_index(name="detections")
        .sort_values("detections", ascending=False)
    )

    st.markdown("#### Camera Performance Summary")

    st.dataframe(
        camera_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("#### Hourly Activity by Camera")

    hourly_camera_data = (
        data.groupby(["hour", "camera_id"])
        .size()
        .reset_index(name="detections")
    )

    fig = px.line(
        hourly_camera_data,
        x="hour",
        y="detections",
        color="camera_id",
        markers=True,
        labels={
            "hour": "Hour of Day",
            "detections": "Detection Events",
            "camera_id": "Camera"
        }
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 14. RECOGNITION PERFORMANCE PAGE
# ============================================================

def recognition_performance_page(data):

    st.subheader("Recognition Performance")

    if data.empty:
        show_empty_message()
        return

    total = len(data)

    plate_extracted = data["plate_number"].notna().sum()

    extraction_rate = (
        plate_extracted / total * 100
        if total > 0
        else 0
    )

    average_detection = data["detection_confidence"].mean()
    average_ocr = data["ocr_confidence"].mean()
    average_plate_type = data["plate_type_confidence"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_kpi(
            "Plate Extraction Rate",
            f"{extraction_rate:.2f}%"
        )

    with col2:
        show_kpi(
            "Mean Detection Confidence",
            f"{average_detection:.3f}"
            if pd.notna(average_detection) else "N/A"
        )

    with col3:
        show_kpi(
            "Mean OCR Confidence",
            f"{average_ocr:.3f}"
            if pd.notna(average_ocr) else "N/A"
        )

    with col4:
        show_kpi(
            "Mean Plate-Type Confidence",
            f"{average_plate_type:.3f}"
            if pd.notna(average_plate_type) else "N/A"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Processing Status")

        status_data = (
            data.groupby("processing_status")
            .size()
            .reset_index(name="events")
        )

        fig = px.pie(
            status_data,
            names="processing_status",
            values="events",
            hole=0.45
        )

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("#### Confidence Score Distributions")

        confidence_columns = [
            "detection_confidence",
            "ocr_confidence",
            "plate_type_confidence"
        ]

        confidence_data = data[
            confidence_columns
        ].melt(
            var_name="model_component",
            value_name="confidence"
        ).dropna()

        fig = px.box(
            confidence_data,
            x="model_component",
            y="confidence",
            color="model_component",
            labels={
                "model_component": "Recognition Component",
                "confidence": "Confidence Score"
            }
        )

        fig.update_layout(
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Confidence scores measure model certainty, not verified "
        "recognition accuracy. Accuracy requires ground-truth labels."
    )


# ============================================================
# 15. PAGE ROUTING
# ============================================================

if page == "Overview":
    overview_page(filtered_df)

elif page == "Traffic Analytics":
    traffic_analytics_page(filtered_df)

elif page == "Geographic Analytics":
    geographic_analytics_page(filtered_df)

elif page == "Vehicle Analytics":
    vehicle_analytics_page(filtered_df)

elif page == "Camera Analytics":
    camera_analytics_page(filtered_df)

elif page == "Recognition Performance":
    recognition_performance_page(filtered_df)


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    "OOH Vehicle Analytics | Data source: Synthetic prototype "
    "| For demonstration purposes only"
)
