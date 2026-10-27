"""Streamlit dashboard for the used-car price regression project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from train_model import (
    ARTIFACT_PATH,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    VEHICLES,
)


st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #172a2a; --muted: #627473; --paper: #f3f6f1; --green: #0c6652; --lime: #d7ed9f; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { color-scheme: light; color: var(--ink) !important; background: radial-gradient(ellipse at 92% 0%, #e2edcf 0, transparent 34%), var(--paper); }
    .stApp h1, .stApp h2, .stApp h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stWidgetLabel"] { color: var(--ink) !important; }
    .hero { border-bottom: 1px solid #d5dfd6; padding: 1.15rem 0 1.5rem; margin-bottom: 1.2rem; }
    .eyebrow { color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { margin: .35rem 0; font-size: 2.45rem; color: var(--ink) !important; }
    .hero p { color: var(--muted); font-size: 1.02rem; margin: 0; max-width: 760px; }
    .stButton > button[kind="primary"] { background: var(--green); border: 0; min-height: 2.8rem; font-weight: 700; }
    [data-testid="stMetric"] { background: #fff; padding: 1rem; border: 1px solid #e0e7df; border-radius: 6px; }
    @media (max-width: 640px) { .hero h1 { font-size: 1.8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact(path: str) -> dict[str, Any]:
    return joblib.load(path)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_vehicle_options() -> tuple[list[str], dict[str, list[str]]]:
    """Use dataset values when present; otherwise use generator's catalog."""
    if DATA_PATH.exists():
        data = load_dataset(str(DATA_PATH))
        company_models = (
            data[["company", "model"]]
            .drop_duplicates()
            .sort_values(["company", "model"])
        )
        mapping = {
            company: group["model"].tolist()
            for company, group in company_models.groupby("company")
        }
        return list(mapping), mapping

    mapping: dict[str, list[str]] = {}
    for company, model, *_ in VEHICLES:
        mapping.setdefault(company, []).append(model)
    return sorted(mapping), mapping


def inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine learning mini-project</div>
      <h1>Used Car Price Predictor</h1>
      <p>Explore how a car's age, distance travelled, and specifications relate to its estimated resale price.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not Path(ARTIFACT_PATH).exists():
    st.warning(
        "The trained models are not available yet. Open the VS Code terminal in "
        "the project folder and run `python train_model.py`, then refresh this page."
    )
    st.stop()

try:
    artifact = load_artifact(str(ARTIFACT_PATH))
except Exception as error:
    st.error(f"Could not load the trained models: {error}")
    st.stop()

if artifact.get("is_synthetic"):
    st.info(
        "Demonstration mode: the model was trained on synthetic sample data. "
        "Its estimates and scores do not represent real market prices."
    )

with st.sidebar:
    st.header("Vehicle details")
    companies, company_models = get_vehicle_options()
    company = st.selectbox("Company", companies)
    model_names = company_models.get(company, [])
    model_name = st.selectbox("Model", model_names)
    year = st.number_input("Year", min_value=1990, max_value=2026, value=2021, step=1)
    km_driven = st.number_input(
        "Kilometers driven", min_value=0, max_value=1_000_000, value=35_000, step=1_000
    )
    fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox(
        "Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"]
    )

    catalog_entry = next(
        (vehicle for vehicle in VEHICLES if vehicle[0] == company and vehicle[1] == model_name),
        None,
    )
    fallback_engine = catalog_entry[3] if catalog_entry else 1200
    fallback_power = catalog_entry[4] if catalog_entry else 90
    fallback_seats = catalog_entry[5] if catalog_entry else 5
    engine_cc = st.number_input(
        "Engine (cc)", min_value=500, max_value=8000, value=int(fallback_engine), step=50
    )
    max_power_bhp = st.number_input(
        "Maximum power (bhp)", min_value=20, max_value=1000, value=int(fallback_power), step=5
    )
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=[2, 4, 5, 6, 7, 8, 9].index(fallback_seats))
    chosen_model = st.selectbox(
        "Prediction model",
        ["Best test-set model", *artifact["models"].keys()],
        help="The best model is selected by lower test-set RMSE.",
    )
    predict_clicked = st.button("Predict price", type="primary", use_container_width=True)

input_row = {
    "company": company,
    "model": model_name,
    "year": int(year),
    "km_driven": int(km_driven),
    "fuel": fuel,
    "transmission": transmission,
    "owner": owner,
    "engine_cc": int(engine_cc),
    "max_power_bhp": int(max_power_bhp),
    "seats": int(seats),
}

if predict_clicked:
    current_year = 2026
    if year > current_year:
        st.error("Year cannot be later than 2026.")
    elif km_driven < 0 or engine_cc <= 0 or max_power_bhp <= 0:
        st.error("Distance, engine size, and power must be positive values.")
    else:
        model_key = (
            artifact["best_model_name"]
            if chosen_model == "Best test-set model"
            else chosen_model
        )
        prediction = float(
            artifact["models"][model_key].predict(pd.DataFrame([input_row], columns=FEATURES))[0]
        )
        st.session_state["prediction"] = (max(0.0, prediction), model_key)

if "prediction" in st.session_state:
    prediction, prediction_model = st.session_state["prediction"]
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Estimated resale price")
        st.markdown(f"## {inr(prediction)}")
        st.caption(f"Predicted with {prediction_model}. This is a classroom estimate, not a quotation.")
    with right:
        best_scores = artifact["metrics"][artifact["best_model_name"]]
        st.metric("Best test-set model", artifact["best_model_name"])
        st.caption(f"RMSE: {inr(best_scores['rmse'])} | R²: {best_scores['r2']:.3f}")

st.divider()
st.header("Model comparison")
st.caption("Scores are calculated on the held-out test set; lower MAE, MSE, and RMSE are better. R² closer to 1 is better.")
metric_rows = []
for model_name_key, scores in artifact["metrics"].items():
    metric_rows.append(
        {
            "Model": model_name_key,
            "MAE (₹)": scores["mae"],
            "MSE (₹²)": scores["mse"],
            "RMSE (₹)": scores["rmse"],
            "R²": scores["r2"],
        }
    )
st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), use_container_width=True)
st.success(
    f"On this test split, **{artifact['best_model_name']}** has the lower RMSE. "
    "That result describes this dataset and split only."
)

if DATA_PATH.exists():
    dataset = load_dataset(str(DATA_PATH))
    available = dataset.dropna(subset=[TARGET])
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Price distribution")
        figure, axis = plt.subplots(figsize=(7, 4))
        sns.histplot(data=available, x=TARGET, bins=24, color="#0c6652", ax=axis)
        axis.set_xlabel("Selling price (INR)")
        axis.set_ylabel("Number of cars")
        axis.ticklabel_format(style="plain", axis="x")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Actual vs predicted")
    plot_data = pd.DataFrame({"Actual": artifact["test_actual"]})
    for model_name_key, values in artifact["test_predictions"].items():
        plot_data[model_name_key] = values
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=plot_data,
        x="Actual",
        y=artifact["best_model_name"],
        color="#d56e3f",
        alpha=0.72,
        ax=axis,
    )
    low = min(plot_data["Actual"].min(), plot_data[artifact["best_model_name"]].min())
    high = max(plot_data["Actual"].max(), plot_data[artifact["best_model_name"]].max())
    axis.plot([low, high], [low, high], color="#172a2a", linestyle="--", linewidth=1)
    axis.set_xlabel("Actual price (INR)")
    axis.set_ylabel("Predicted price (INR)")
    axis.ticklabel_format(style="plain")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with chart_right:
    st.subheader("Price by company")
    if DATA_PATH.exists():
        figure, axis = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=dataset, x="company", y=TARGET, color="#d7ed9f", ax=axis)
        axis.set_xlabel("")
        axis.set_ylabel("Selling price (INR)")
        axis.tick_params(axis="x", rotation=30)
        axis.ticklabel_format(style="plain", axis="y")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    else:
        st.caption("Train the model to generate the demonstration dataset and charts.")

st.caption("Inputs: " + ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES))