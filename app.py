from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from oil_forecast_core import (
    BASE_DIR,
    BEST_MODEL_FORECAST_COLUMN,
    BEST_MODEL_NAME,
    LAG_COUNT,
    TEST_START_DATE,
    build_results,
    get_default_scenario_window,
    load_weekly_data,
)


st.set_page_config(
    page_title="Oil Price Forecasting Dashboard",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(236, 196, 115, 0.18), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(10, 56, 76, 0.18), transparent 25%),
                    linear-gradient(135deg, #f3efe3 0%, #f7f4eb 40%, #eef3f5 100%);
            }
            [data-testid="stMainBlockContainer"] h1,
            [data-testid="stMainBlockContainer"] h2,
            [data-testid="stMainBlockContainer"] h3,
            [data-testid="stMainBlockContainer"] h4,
            [data-testid="stMainBlockContainer"] p,
            [data-testid="stMainBlockContainer"] label,
            [data-testid="stMainBlockContainer"] .stMarkdown,
            [data-testid="stMainBlockContainer"] .stCaption {
                color: #102b39;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f2d3a 0%, #164356 100%);
            }
            [data-testid="stSidebar"] * {
                color: #f6f0e5;
            }
            .hero-card {
                padding: 1.4rem 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #102b39 0%, #1d4e63 55%, #d59b35 100%);
                color: #f8f4ea;
                box-shadow: 0 24px 48px rgba(16, 43, 57, 0.18);
                margin-bottom: 1rem;
            }
            .hero-title {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 2.35rem;
                font-weight: 700;
                line-height: 1.05;
                margin-bottom: 0.45rem;
            }
            .hero-subtitle {
                font-size: 1rem;
                color: rgba(248, 244, 234, 0.88);
                max-width: 56rem;
            }
            .metric-card {
                background: rgba(255, 252, 245, 0.92);
                border: 1px solid rgba(16, 43, 57, 0.08);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 26px rgba(16, 43, 57, 0.08);
            }
            .metric-label {
                color: #49616d;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                color: #102b39;
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.8rem;
                font-weight: 700;
                line-height: 1.1;
            }
            .metric-note {
                color: #5c717b;
                font-size: 0.9rem;
                margin-top: 0.25rem;
            }
            .section-caption {
                color: #5c717b;
                margin-top: -0.35rem;
                margin-bottom: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def draw_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


@st.cache_data(show_spinner=False)
def get_initial_window() -> list[float]:
    return get_default_scenario_window(load_weekly_data())


@st.cache_data(show_spinner=False)
def get_dashboard_results(horizon: int, scenario_window: tuple[float, ...]):
    return build_results(horizon=horizon, scenario_window=list(scenario_window))


def plot_history_and_forecast(history: pd.DataFrame, future: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.patch.set_facecolor("#fdfaf3")
    ax.set_facecolor("#fdfaf3")

    recent_history = history.tail(26)

    ax.plot(
        recent_history["Date"],
        recent_history["Oilprice"],
        color="#0f2d3a",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Recent observed price",
    )
    ax.plot(
        future["Date"],
        future["Predicted_Oilprice"],
        color="#cf8b17",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Forecast",
    )
    ax.fill_between(
        future["Date"],
        future["Lower_Bound"],
        future["Upper_Bound"],
        color="#d6a03f",
        alpha=0.2,
        label="Heuristic band (+/- RMSE)",
    )

    ax.set_title("Forward Oil Price Forecast", fontsize=15, fontweight="bold", color="#102b39")
    ax.set_ylabel("Oil price (USD)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    return fig


def plot_backtest(reconstructed_forecasts: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.patch.set_facecolor("#fdfaf3")
    ax.set_facecolor("#fdfaf3")

    ann_backtest = reconstructed_forecasts.dropna(subset=[BEST_MODEL_FORECAST_COLUMN]).copy()

    ax.plot(
        ann_backtest["Date"],
        ann_backtest["Actual_Oilprice"],
        color="#0f2d3a",
        linewidth=2.2,
        label="Actual oil price",
    )
    ax.plot(
        ann_backtest["Date"],
        ann_backtest[BEST_MODEL_FORECAST_COLUMN],
        color="#cf8b17",
        linewidth=2.2,
        linestyle="--",
        label="ANN + oil prediction",
    )

    ax.set_title("Held-Out Test Performance", fontsize=15, fontweight="bold", color="#102b39")
    ax.set_ylabel("Oil price (USD)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    return fig


def plot_full_history(history: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.patch.set_facecolor("#fdfaf3")
    ax.set_facecolor("#fdfaf3")

    ax.plot(history["Date"], history["Oilprice"], color="#1d4e63", linewidth=2.2)
    ax.axvspan(TEST_START_DATE, history["Date"].max(), color="#ecc473", alpha=0.22)

    ax.set_title("Weekly Oil Price History", fontsize=15, fontweight="bold", color="#102b39")
    ax.set_ylabel("Oil price (USD)")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        TEST_START_DATE,
        history["Oilprice"].max() + 1.5,
        "Test window",
        color="#8e5c0a",
        fontsize=10,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    return fig


def show_method_snapshot() -> None:
    st.markdown("### Model snapshot")
    st.caption("This web deployment uses the same best model configuration reported in your notebook results.")

    method_df = pd.DataFrame(
        [
            {
                "Deployed model": BEST_MODEL_NAME,
                "Input design": "4 lagged weekly oil prices",
                "Training window": "2005-02-04 to 2024-04-12",
                "Inference style": "Recursive multi-week forecasting",
                "Why deployed": "Lowest RMSE, MAE, and MAPE among all tested models",
            }
        ]
    )

    st.dataframe(method_df, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Applied ML Forecasting in Oil Prices</div>
            <div class="hero-subtitle">
                A simple deployment-ready dashboard for your final project. The app retrains the best-performing
                ANN model from the notebook, visualizes its backtest performance, and produces forward weekly oil
                price forecasts from the latest observed data or a custom what-if scenario.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_window = get_initial_window()

    with st.sidebar:
        st.title("Forecast Controls")
        forecast_horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=12, value=8)
        scenario_mode = st.toggle("Enable what-if scenario", value=False)

        st.markdown("Use the last four weekly oil prices as the model input window.")

        if scenario_mode:
            scenario_window = [
                st.number_input(
                    f"Oil price {lag} week(s) ago",
                    value=float(latest_window[lag - 1]),
                    step=0.5,
                    format="%.3f",
                )
                for lag in range(1, LAG_COUNT + 1)
            ]
        else:
            scenario_window = latest_window

        st.caption(
            "The uncertainty band shown in the forecast chart uses the ANN test RMSE as a simple deployment-friendly reference range."
        )

    results = get_dashboard_results(forecast_horizon, tuple(float(value) for value in scenario_window))
    best_model_metrics = results.performance_table.loc[
        results.performance_table["Model"] == BEST_MODEL_NAME
    ].iloc[0]

    next_week_forecast = float(results.future_forecast["Predicted_Oilprice"].iloc[0])
    last_observed = float(results.history["Oilprice"].iloc[-1])
    horizon_average = float(results.future_forecast["Predicted_Oilprice"].mean())

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        draw_metric_card("Deployed model", BEST_MODEL_NAME, "Best overall model from the comparison table")
    with metric_col2:
        draw_metric_card("Latest observed price", format_currency(last_observed), "Most recent weekly value in the cleaned dataset")
    with metric_col3:
        draw_metric_card("Next-week forecast", format_currency(next_week_forecast), f"Forecast for {results.future_forecast['Date'].iloc[0].date()}")
    with metric_col4:
        draw_metric_card("Test RMSE", f"{best_model_metrics['RMSE']:.3f}", f"Average {forecast_horizon}-week projection: {format_currency(horizon_average)}")

    chart_col1, chart_col2 = st.columns([1.1, 1.0])
    with chart_col1:
        st.markdown("### Future outlook")
        st.markdown('<div class="section-caption">Forecast generated from the 4-week ANN input window.</div>', unsafe_allow_html=True)
        st.pyplot(plot_history_and_forecast(results.history, results.future_forecast), use_container_width=True)

    with chart_col2:
        st.markdown("### Backtest fit")
        st.markdown('<div class="section-caption">Historical test predictions from the saved project output.</div>', unsafe_allow_html=True)
        st.pyplot(plot_backtest(results.reconstructed_forecasts), use_container_width=True)

    history_col, table_col = st.columns([1.2, 0.8])
    with history_col:
        st.markdown("### Historical context")
        st.markdown('<div class="section-caption">The shaded region marks the ANN test period used for evaluation.</div>', unsafe_allow_html=True)
        st.pyplot(plot_full_history(results.history), use_container_width=True)

    with table_col:
        st.markdown("### Model leaderboard")
        st.dataframe(
            results.performance_table.style.format(
                {"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE": "{:.3f}", "R^2": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Forecast table")
        forecast_table = results.future_forecast.copy()
        forecast_table["Date"] = forecast_table["Date"].dt.date
        st.dataframe(
            forecast_table.style.format(
                {
                    "Predicted_Oilprice": "${:,.2f}",
                    "Lower_Bound": "${:,.2f}",
                    "Upper_Bound": "${:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    show_method_snapshot()

    if scenario_mode:
        st.info(
            "Scenario mode is on. The forecast uses your manual four-week input window instead of the latest observed prices."
        )

    figure_path = BASE_DIR / "outputs" / "figures" / "figure_11_best_model_actual_vs_predicted.png"
    if figure_path.exists():
        with st.expander("View saved notebook figure"):
            st.image(str(figure_path), caption="Saved best-model figure exported from the project notebook.")


if __name__ == "__main__":
    main()
