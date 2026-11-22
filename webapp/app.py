"""Interactive web app to test the model through the API.

    API_URL=http://localhost:8000 streamlit run webapp/app.py      # or: make webapp

1. Load rides: upload a parquet/CSV file (GBG500 format), or pick an example ride
   from the test split when it is available locally (EXAMPLES_PATH).
2. Look at the IMU signals of a ride.
3. Classify: every ride of the file goes to POST /predict/file in one call.
4. Optionally send the true label back (POST /feedback).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ride_classifier.client import ApiClient, read_rides
from ride_classifier.config import CHANNELS, CLASSES, RIDE_ID, ROOT, TARGET, TIME

API_URL = os.getenv("API_URL", "http://localhost:8000")
EXAMPLES_PATH = Path(os.getenv("EXAMPLES_PATH", ROOT / "data/processed/test.parquet"))

# Reference data-viz palette, categorical slots 1-3 (validated for all pairs, light and dark)
AXIS_COLORS = {
    "light": {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"},
    "dark": {"x": "#3987e5", "y": "#d95926", "z": "#199e70"},
}
GROUPS = {
    "Accelerometer (m/s²)": {"x": "ax", "y": "ay", "z": "az"},
    "Gyroscope": {"x": "rx", "y": "ry", "z": "rz"},
}

st.set_page_config(page_title="Ride classifier", page_icon="🚲", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(API_URL)


@st.cache_data
def load_examples(path: str) -> pd.DataFrame | None:
    """A few rides per class from the test split, with their true label (dev only: data is not in the image)."""
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    keep = df.drop_duplicates(RIDE_ID).groupby(TARGET).head(2)[RIDE_ID]
    return df[df[RIDE_ID].isin(keep)].reset_index(drop=True)


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def signal_chart(ride: pd.DataFrame, title: str, columns: dict[str, str]) -> go.Figure:
    """One chart per sensor (accelerometer and gyroscope have different units: never one shared axis)."""
    colors = AXIS_COLORS[theme()]
    seconds = (ride[TIME] - ride[TIME].min()) / 1000
    fig = go.Figure()
    for axis, column in columns.items():
        fig.add_scatter(
            x=seconds,
            y=ride[column],
            name=column,
            mode="lines",
            line={"width": 2, "color": colors[axis]},
            hovertemplate=f"{column}: %{{y:.2f}}<br>t = %{{x:.1f}} s<extra></extra>",
        )
    fig.update_layout(
        title={"text": title, "font": {"size": 14}},
        height=260,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.15, "x": 1, "xanchor": "right"},
        xaxis={"title": "time (s)", "showgrid": False},
        yaxis={"gridcolor": "rgba(128,128,128,0.15)", "zeroline": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def probability_chart(row: pd.Series) -> go.Figure:
    """Single series: one hue, no legend, values labelled (5 bars only)."""
    color = AXIS_COLORS[theme()]["x"]
    proba = row[CLASSES].astype(float)
    fig = go.Figure(
        go.Bar(
            x=proba.values,
            y=CLASSES,
            orientation="h",
            marker={"color": color, "cornerradius": 4},
            text=[f"{p:.0%}" for p in proba.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=230,
        margin={"l": 10, "r": 40, "t": 10, "b": 10},
        xaxis={"range": [0, 1.1], "tickformat": ".0%", "showgrid": False, "visible": False},
        yaxis={"autorange": "reversed"},
        bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --- Sidebar: API status -------------------------------------------------------------
client = get_client()
with st.sidebar:
    st.header("API")
    st.caption(API_URL)
    try:
        health = client.health()
        if health["model_loaded"]:
            st.success(f"Online · model v{health['model_version']}", icon="✅")
        else:
            st.warning("Online, but the model is not loaded", icon="⚠️")
    except Exception:
        health = None
        st.error("API unreachable. Start it with `make api`.", icon="🚫")
    st.divider()
    st.markdown(f"[Swagger UI]({API_URL}/docs) · [Prometheus metrics]({API_URL}/metrics)")

st.title("🚲 Ride classifier")
st.caption(
    "Classifies a bike ride from its IMU time series (10 Hz) into "
    + ", ".join(f"**{c}**" for c in CLASSES)
    + ". The prediction is the mean over 25.6 s windows of the ride."
)

# --- 1. Input ----------------------------------------------------------------------
examples = load_examples(str(EXAMPLES_PATH))
sources = ["Upload a file"] + (["Example rides (test split)"] if examples is not None else [])
source = st.radio("Rides", sources, horizontal=True, label_visibility="collapsed")

rides: pd.DataFrame | None = None
if source == "Upload a file":
    upload = st.file_uploader(
        "Parquet or CSV file with columns `time_index, ax, ay, az, rx, ry, rz` (+ optional `ride_id`, `label`)",
        type=["parquet", "csv"],
    )
    if upload is not None:
        try:
            rides = read_rides(upload.name, upload.getvalue())
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
else:
    rides = examples

if rides is None:
    st.info("Upload a ride to start, e.g. a file in the GBG500 format.")
    st.stop()

ride_ids = rides[RIDE_ID].unique().tolist()
st.write(f"**{len(ride_ids)}** ride(s), **{len(rides):,}** samples.")

# --- 2. Classify (all rides of the file, one API call) ---------------------------------
key = (source, len(rides), tuple(ride_ids[:5]))
if st.session_state.get("pred_key") != key:
    st.session_state.pop("predictions", None)
if st.button("Classify", type="primary", disabled=not (health and health["model_loaded"])):
    with st.spinner("Calling the API..."):
        try:
            st.session_state["predictions"] = client.predict(rides)
            st.session_state["pred_key"] = key
        except Exception as exc:
            st.error(str(exc))
predictions: pd.DataFrame | None = st.session_state.get("predictions")

if predictions is not None and len(predictions) > 1:
    table = predictions.set_index(RIDE_ID)
    if TARGET in rides:
        truth = rides.drop_duplicates(RIDE_ID).set_index(RIDE_ID)[TARGET]
        table.insert(1, "true label", truth.reindex(table.index))
        st.metric("Correct predictions", f"{(table['prediction'] == table['true label']).mean():.0%}")
    st.dataframe(
        table.drop(columns="model_version"),
        column_config={
            c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1) for c in CLASSES
        },
    )

# --- 3. One ride in detail ---------------------------------------------------------
st.subheader("Ride details")
ride_id = st.selectbox("Ride", ride_ids, format_func=lambda r: r if len(r) <= 24 else f"{r[:20]}…")
ride = rides[rides[RIDE_ID] == ride_id]

left, right = st.columns([3, 2])
with left:
    for title, columns in GROUPS.items():
        st.plotly_chart(signal_chart(ride, title, columns), width="stretch")
    duration = (ride[TIME].max() - ride[TIME].min()) / 1000
    st.caption(
        f"{len(ride):,} samples · {duration / 60:.1f} min · {ride[CHANNELS].isna().any(axis=1).mean():.1%} missing"
    )

with right:
    if predictions is None:
        st.info("Click **Classify** to get a prediction.")
    else:
        row = predictions.set_index(RIDE_ID).loc[ride_id]
        st.metric("Prediction", row["prediction"], help=f"Model version {row['model_version']}")
        if TARGET in ride:
            actual = ride[TARGET].iloc[0]
            if actual == row["prediction"]:
                st.success(f"True label: {actual}", icon="✅")
            else:
                st.warning(f"True label: {actual}", icon="❌")
        st.plotly_chart(probability_chart(row), width="stretch")
        with st.expander("Probabilities as a table"):
            st.dataframe(row[CLASSES].astype(float).rename("probability").to_frame().style.format("{:.1%}"))

        with st.form("feedback"):
            st.write("Know the true label? Send it back to the API (feedback metrics).")
            default = CLASSES.index(ride[TARGET].iloc[0]) if TARGET in ride else 0
            actual = st.selectbox("True label", CLASSES, index=default)
            if st.form_submit_button("Send feedback"):
                try:
                    client.feedback(ride_id, row["prediction"], actual)
                    st.toast("Feedback sent", icon="✅")
                except Exception as exc:
                    st.error(f"Feedback failed: {exc}")
