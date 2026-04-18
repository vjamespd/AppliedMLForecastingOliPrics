from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
WEEKLY_DATA_PATH = BASE_DIR / "outputs" / "processed_data" / "weekly_data_after_outlier_treatment.csv"
RECONSTRUCTED_FORECASTS_PATH = BASE_DIR / "outputs" / "processed_data" / "model_forecasts_reconstructed.csv"
PERFORMANCE_TABLE_PATH = BASE_DIR / "outputs" / "tables" / "table_8_final_model_performance_comparison.csv"

BEST_MODEL_NAME = "ANN + oil"
BEST_MODEL_FORECAST_COLUMN = "ANN + oil"
LAG_COUNT = 4
TEST_START_DATE = pd.Timestamp("2020-06-12")
FEATURE_COLUMNS = [f"Oilprice_lag_{lag}" for lag in range(1, LAG_COUNT + 1)]


@dataclass(frozen=True)
class ForecastResults:
    history: pd.DataFrame
    future_forecast: pd.DataFrame
    performance_table: pd.DataFrame
    reconstructed_forecasts: pd.DataFrame
    scenario_window: list[float]


def load_weekly_data() -> pd.DataFrame:
    weekly_data = pd.read_csv(WEEKLY_DATA_PATH, parse_dates=["Date"])
    return weekly_data.sort_values("Date").reset_index(drop=True)


def load_reconstructed_forecasts() -> pd.DataFrame:
    forecasts = pd.read_csv(RECONSTRUCTED_FORECASTS_PATH, parse_dates=["Date"])
    return forecasts.sort_values("Date").reset_index(drop=True)


def load_performance_table() -> pd.DataFrame:
    performance = pd.read_csv(PERFORMANCE_TABLE_PATH)
    return performance.sort_values("RMSE").reset_index(drop=True)


def build_supervised_oil_only_dataset(weekly_data: pd.DataFrame, lag_count: int = LAG_COUNT) -> pd.DataFrame:
    supervised = weekly_data[["Date", "Oilprice"]].copy()

    for lag in range(1, lag_count + 1):
        supervised[f"Oilprice_lag_{lag}"] = supervised["Oilprice"].shift(lag)

    supervised = supervised.dropna().reset_index(drop=True)
    return supervised


def create_best_ann_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    learning_rate_init=0.001,
                    batch_size=8,
                    max_iter=2000,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def train_best_ann_model(weekly_data: pd.DataFrame) -> Pipeline:
    supervised = build_supervised_oil_only_dataset(weekly_data)

    model = create_best_ann_pipeline()
    model.fit(supervised[FEATURE_COLUMNS], supervised["Oilprice"])
    return model


def recursive_forecast(
    model: Pipeline,
    oil_history: pd.Series,
    horizon: int,
    last_observed_date: pd.Timestamp,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 week.")

    if len(oil_history) < LAG_COUNT:
        raise ValueError(f"Need at least {LAG_COUNT} historical values to forecast.")

    rolling_window = oil_history.astype(float).tolist()[-LAG_COUNT:]
    forecast_rows: list[dict[str, float | pd.Timestamp]] = []

    for step in range(1, horizon + 1):
        model_input = pd.DataFrame([rolling_window[::-1]], columns=FEATURE_COLUMNS)
        prediction = float(model.predict(model_input)[0])
        forecast_date = last_observed_date + pd.Timedelta(weeks=step)

        forecast_rows.append(
            {
                "Date": forecast_date,
                "Predicted_Oilprice": prediction,
            }
        )

        rolling_window = rolling_window[1:] + [prediction]

    return pd.DataFrame(forecast_rows)


def get_default_scenario_window(weekly_data: pd.DataFrame) -> list[float]:
    latest_values = weekly_data["Oilprice"].tail(LAG_COUNT).tolist()
    return [float(value) for value in latest_values[::-1]]


def build_results(horizon: int, scenario_window: list[float] | None = None) -> ForecastResults:
    history = load_weekly_data()
    performance_table = load_performance_table()
    reconstructed_forecasts = load_reconstructed_forecasts()
    model = train_best_ann_model(history)

    if scenario_window is None:
        effective_window = get_default_scenario_window(history)
    else:
        if len(scenario_window) != LAG_COUNT:
            raise ValueError(f"Scenario window must contain {LAG_COUNT} values.")
        effective_window = [float(value) for value in scenario_window]

    scenario_series = pd.Series(effective_window[::-1], name="Oilprice")
    future_forecast = recursive_forecast(
        model=model,
        oil_history=scenario_series,
        horizon=horizon,
        last_observed_date=history["Date"].iloc[-1],
    )

    rmse = float(
        performance_table.loc[performance_table["Model"] == BEST_MODEL_NAME, "RMSE"].iloc[0]
    )
    future_forecast["Lower_Bound"] = future_forecast["Predicted_Oilprice"] - rmse
    future_forecast["Upper_Bound"] = future_forecast["Predicted_Oilprice"] + rmse

    return ForecastResults(
        history=history,
        future_forecast=future_forecast,
        performance_table=performance_table,
        reconstructed_forecasts=reconstructed_forecasts,
        scenario_window=effective_window,
    )
