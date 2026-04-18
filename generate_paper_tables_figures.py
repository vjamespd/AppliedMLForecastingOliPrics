from __future__ import annotations

import ast
import json
import math
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore", category=ConvergenceWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "APPLIED-ML-FORECASTING-OIL-PRICE.ipynb"
DATA_PATH = PROJECT_ROOT / "final_oil-us-indexv1.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed_data"

TRAIN_RATIO = 0.8
ANN_LAGS = 4
LAGGED_CORR_MAX_LAG = 12


def configure_style() -> None:
    """Set a clean, thesis-friendly plotting style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_directories() -> None:
    """Create the requested output folders."""
    for directory in (OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def date_string(value: pd.Timestamp) -> str:
    """Return a stable ISO date string."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def wrap_text(value: object, width: int = 28) -> str:
    """Wrap long table strings to keep rendered PNG tables readable."""
    if pd.isna(value):
        return ""
    text = str(value)
    if len(text) <= width:
        return text
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def format_numeric(value: float, decimals: int = 4) -> str:
    """Format numbers for displayed tables."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.{decimals}f}"


def render_table_png(df: pd.DataFrame, title: str, png_path: Path, decimals: int = 4) -> None:
    """Render a dataframe as a PNG image using matplotlib."""
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda x: format_numeric(x, decimals=decimals))
        else:
            display_df[column] = display_df[column].map(wrap_text)

    rows, cols = display_df.shape
    fig_width = max(8.5, min(18, cols * 2.1))
    fig_height = max(2.4, min(18, (rows + 1) * 0.5))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=12)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f6f8fa")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table(
    df: pd.DataFrame,
    stem: str,
    title: str,
    created: dict[str, list[Path]],
    decimals: int = 4,
) -> None:
    """Save a table to both CSV and PNG."""
    csv_path = TABLES_DIR / f"{stem}.csv"
    png_path = TABLES_DIR / f"{stem}.png"
    df.to_csv(csv_path, index=False)
    render_table_png(df, title, png_path, decimals=decimals)
    created["tables"].extend([csv_path, png_path])


def save_dataframe_csv(df: pd.DataFrame, filename: str, created: dict[str, list[Path]]) -> Path:
    """Save a dataframe in the processed_data folder and record it."""
    path = PROCESSED_DIR / filename
    df.to_csv(path)
    created["processed_data"].append(path)
    return path


def save_figure(fig: plt.Figure, filename: str, created: dict[str, list[Path]]) -> None:
    """Save a figure as a 300 dpi PNG."""
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created["figures"].append(path)


def percent_series(counts: pd.Series, denominator: int) -> pd.Series:
    """Convert counts to percentages."""
    return (counts / denominator) * 100.0


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute mean absolute percentage error in percent."""
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    safe_mask = y_true != 0
    if not safe_mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[safe_mask] - y_pred[safe_mask]) / y_true[safe_mask])) * 100.0)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics used in the notebook."""
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mape(y_true, y_pred)),
        "R^2": float(r2_score(y_true, y_pred)),
    }


def create_supervised_features(
    df: pd.DataFrame,
    target_col: str = "Oilprice",
    exog_cols: list[str] | None = None,
    lags: int = 4,
    include_current_exog: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Recreate the notebook's ANN supervised feature transformation."""
    if exog_cols is None:
        exog_cols = []

    supervised = pd.DataFrame(index=df.index)
    supervised[target_col] = df[target_col]

    for lag in range(1, lags + 1):
        supervised[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    for exog_col in exog_cols:
        for lag in range(1, lags + 1):
            supervised[f"{exog_col}_lag_{lag}"] = df[exog_col].shift(lag)
        if include_current_exog:
            supervised[f"{exog_col}_current"] = df[exog_col]

    supervised = supervised.dropna()
    X = supervised.drop(columns=[target_col])
    y = supervised[target_col]
    return supervised, X, y


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Recreate the notebook's time-ordered train/test split."""
    split_index = int(len(X) * train_ratio)
    return (
        X.iloc[:split_index].copy(),
        X.iloc[split_index:].copy(),
        y.iloc[:split_index].copy(),
        y.iloc[split_index:].copy(),
    )


def build_summary_rows(pairs: list[tuple[str, object]]) -> pd.DataFrame:
    """Create a two-column summary dataframe."""
    return pd.DataFrame(pairs, columns=["Statistic", "Value"])


def draw_annotated_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    """Draw a compact correlation heatmap with text annotations."""
    im = ax.imshow(matrix.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def extract_metric_dict(text: str) -> dict[str, float] | None:
    """Extract the notebook's printed metric dictionary from stdout text."""
    match = re.search(
        r"\{'RMSE': ([0-9eE\.\-]+), 'MAE': ([0-9eE\.\-]+), 'MAPE': np\.float64\(([0-9eE\.\-]+)\), 'R2': ([0-9eE\.\-]+)\}",
        text,
    )
    if not match:
        return None
    return {
        "RMSE": float(match.group(1)),
        "MAE": float(match.group(2)),
        "MAPE": float(match.group(3)),
        "R^2": float(match.group(4)),
    }


def parse_tuple_from_output(text: str, label: str) -> tuple[int, ...] | None:
    """Parse a tuple such as an ARIMA order from notebook stdout."""
    match = re.search(rf"{re.escape(label)}: (\([^)]+\))", text)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def parse_param_dict(text: str) -> dict[str, object] | None:
    """Parse the printed ANN parameter dictionary from notebook stdout."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    return ast.literal_eval(match.group(0))


def inspect_notebook(notebook_path: Path) -> dict[str, object]:
    """Inspect the notebook to verify model settings and printed metrics."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells: list[dict[str, str]] = []

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        output_text = ""
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                output_text += "".join(text)
            elif isinstance(text, str):
                output_text += text
        cells.append({"source": source, "output_text": output_text})

    inspection: dict[str, object] = {
        "configs": {},
        "metrics": {},
        "notes": [],
    }

    for cell in cells:
        source = cell["source"]
        output_text = cell["output_text"]

        if "best_arima_forecast = best_arima_model.forecast" in source:
            order = parse_tuple_from_output(output_text, "Best ARIMA order")
            metrics = extract_metric_dict(output_text)
            if order is not None:
                inspection["configs"]["arima_order"] = order
            if metrics is not None:
                inspection["metrics"]["ARIMA baseline"] = metrics

        if "best_sarimax_forecast = best_sarimax_model.forecast" in source:
            order = parse_tuple_from_output(output_text, "Best SARIMAX order")
            seasonal = parse_tuple_from_output(output_text, "Best SARIMAX seasonal order")
            metrics = extract_metric_dict(output_text)
            if order is not None:
                inspection["configs"]["sarimax_order"] = order
            if seasonal is not None:
                inspection["configs"]["sarimax_seasonal_order"] = seasonal
            if metrics is not None:
                inspection["metrics"]["SARIMAX + oil + us-index"] = metrics

        if "random_search_oil.fit(X_train_oil, y_train_oil)" in source:
            params = parse_param_dict(output_text)
            if params is not None:
                inspection["configs"]["ann_oil_params"] = params

        if "best_ann_oil = random_search_oil.best_estimator_" in source:
            metrics = extract_metric_dict(output_text)
            if metrics is not None:
                inspection["metrics"]["ANN + oil"] = metrics

        if "random_search_oil_us.fit(X_train_oil_us, y_train_oil_us)" in source:
            params = parse_param_dict(output_text)
            if params is not None:
                inspection["configs"]["ann_oil_us_params"] = params

        if "best_ann_oil_us = random_search_oil_us.best_estimator_" in source:
            metrics = extract_metric_dict(output_text)
            if metrics is not None:
                inspection["metrics"]["ANN + oil + us-index"] = metrics

    if all(
        key in inspection["configs"]
        for key in ("arima_order", "sarimax_order", "sarimax_seasonal_order", "ann_oil_params", "ann_oil_us_params")
    ):
        inspection["notes"].append("Verified model configurations from notebook source/output cells.")
    else:
        inspection["notes"].append("Some model configurations could not be extracted automatically from the notebook.")

    inspection["notes"].append(
        "Forecast variables are defined in notebook source cells, but explicit forecast arrays are not persisted in notebook outputs."
    )
    inspection["notes"].append(
        "Figures requiring forecasts will therefore use faithfully reconstructed prediction series from the verified workflow."
    )
    return inspection


def prepare_datasets(created: dict[str, list[Path]]) -> dict[str, object]:
    """Load the CSV and reproduce the preprocessing pipeline exactly."""
    raw = pd.read_csv(DATA_PATH)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").set_index("Date")

    raw_missing_counts = raw.isna().sum()
    raw_missing_summary = pd.DataFrame(
        {
            "Column": raw_missing_counts.index,
            "Missing count": raw_missing_counts.values,
            "Missing percentage": percent_series(raw_missing_counts, len(raw)).values,
        }
    )

    daily_clean = raw.copy()
    daily_clean["Oilprice"] = daily_clean["Oilprice"].interpolate(method="linear")
    daily_clean["us-index"] = daily_clean["us-index"].interpolate(method="linear")
    daily_clean = daily_clean.ffill().bfill()

    clean_missing_counts = daily_clean.isna().sum()
    clean_missing_summary = pd.DataFrame(
        {
            "Column": clean_missing_counts.index,
            "Missing count": clean_missing_counts.values,
            "Missing percentage": percent_series(clean_missing_counts, len(daily_clean)).values,
        }
    )

    weekly_before = daily_clean.resample("W-FRI").mean()

    q1 = float(weekly_before["Oilprice"].quantile(0.25))
    q3 = float(weekly_before["Oilprice"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (weekly_before["Oilprice"] < lower_bound) | (weekly_before["Oilprice"] > upper_bound)
    weekly_after = weekly_before.copy()
    weekly_after["Oilprice"] = weekly_after["Oilprice"].clip(lower=lower_bound, upper=upper_bound)
    capped_values = int((weekly_before["Oilprice"] != weekly_after["Oilprice"]).sum())
    outlier_count = int(outlier_mask.sum())

    save_dataframe_csv(daily_clean, "daily_data_cleaned.csv", created)
    save_dataframe_csv(weekly_before, "weekly_data_before_outlier_treatment.csv", created)
    save_dataframe_csv(weekly_after, "weekly_data_after_outlier_treatment.csv", created)

    return {
        "raw": raw,
        "daily_clean": daily_clean,
        "weekly_before": weekly_before,
        "weekly_after": weekly_after,
        "raw_missing_summary": raw_missing_summary,
        "clean_missing_summary": clean_missing_summary,
        "outlier_stats": {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower bound": lower_bound,
            "Upper bound": upper_bound,
            "Number of outliers detected": outlier_count,
            "Number of values capped": capped_values,
            "Capped minimum": float(weekly_after["Oilprice"].min()),
            "Capped maximum": float(weekly_after["Oilprice"].max()),
        },
    }


def fit_models(
    weekly_after: pd.DataFrame,
    notebook_info: dict[str, object],
    issues: list[str],
) -> dict[str, object]:
    """Reconstruct the model predictions used for the requested figures."""
    configs = notebook_info.get("configs", {})

    # ARIMA baseline
    series = weekly_after["Oilprice"].dropna().copy()
    classical_split = int(len(series) * TRAIN_RATIO)
    train_arima = series.iloc[:classical_split].copy()
    test_arima = series.iloc[classical_split:].copy()

    arima_order = configs.get("arima_order")
    if arima_order is None:
        raise RuntimeError("Could not verify the best ARIMA order from the notebook.")
    arima_model = ARIMA(train_arima, order=arima_order).fit()
    arima_forecast = pd.Series(arima_model.forecast(steps=len(test_arima)), index=test_arima.index, name="ARIMA baseline")
    arima_metrics = regression_metrics(test_arima, arima_forecast)

    # SARIMAX with exogenous US index
    endog = weekly_after["Oilprice"].dropna().copy()
    exog = weekly_after[["us-index"]].copy().loc[endog.index]
    classical_split = int(len(endog) * TRAIN_RATIO)
    train_endog = endog.iloc[:classical_split].copy()
    test_endog = endog.iloc[classical_split:].copy()
    train_exog = exog.iloc[:classical_split].copy()
    test_exog = exog.iloc[classical_split:].copy()

    sarimax_order = configs.get("sarimax_order")
    sarimax_seasonal = configs.get("sarimax_seasonal_order")
    if sarimax_order is None or sarimax_seasonal is None:
        raise RuntimeError("Could not verify the best SARIMAX orders from the notebook.")
    sarimax_model = SARIMAX(
        train_endog,
        exog=train_exog,
        order=sarimax_order,
        seasonal_order=sarimax_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarimax_forecast = pd.Series(
        sarimax_model.forecast(steps=len(test_endog), exog=test_exog),
        index=test_endog.index,
        name="SARIMAX + oil + us-index",
    )
    sarimax_metrics = regression_metrics(test_endog, sarimax_forecast)

    # ANN + oil only
    supervised_oil, X_oil, y_oil = create_supervised_features(
        weekly_after,
        target_col="Oilprice",
        exog_cols=[],
        lags=ANN_LAGS,
    )
    X_train_oil, X_test_oil, y_train_oil, y_test_oil = time_series_split(X_oil, y_oil, train_ratio=TRAIN_RATIO)

    ann_oil_params = configs.get("ann_oil_params")
    if ann_oil_params is None:
        raise RuntimeError("Could not verify the best ANN + oil parameters from the notebook.")
    ann_oil_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "mlp",
                MLPRegressor(
                    max_iter=2000,
                    random_state=42,
                    early_stopping=True,
                    hidden_layer_sizes=ann_oil_params["mlp__hidden_layer_sizes"],
                    activation=ann_oil_params["mlp__activation"],
                    solver=ann_oil_params["mlp__solver"],
                    alpha=ann_oil_params["mlp__alpha"],
                    learning_rate_init=ann_oil_params["mlp__learning_rate_init"],
                    batch_size=ann_oil_params["mlp__batch_size"],
                ),
            ),
        ]
    )
    ann_oil_pipeline.fit(X_train_oil, y_train_oil)
    ann_oil_forecast = pd.Series(ann_oil_pipeline.predict(X_test_oil), index=y_test_oil.index, name="ANN + oil")
    ann_oil_metrics = regression_metrics(y_test_oil, ann_oil_forecast)

    # ANN + oil + US index
    supervised_oil_us, X_oil_us, y_oil_us = create_supervised_features(
        weekly_after,
        target_col="Oilprice",
        exog_cols=["us-index"],
        lags=ANN_LAGS,
    )
    X_train_oil_us, X_test_oil_us, y_train_oil_us, y_test_oil_us = time_series_split(
        X_oil_us,
        y_oil_us,
        train_ratio=TRAIN_RATIO,
    )

    ann_oil_us_params = configs.get("ann_oil_us_params")
    if ann_oil_us_params is None:
        raise RuntimeError("Could not verify the best ANN + oil + us-index parameters from the notebook.")
    ann_oil_us_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "mlp",
                MLPRegressor(
                    max_iter=2000,
                    random_state=42,
                    early_stopping=True,
                    hidden_layer_sizes=ann_oil_us_params["mlp__hidden_layer_sizes"],
                    activation=ann_oil_us_params["mlp__activation"],
                    solver=ann_oil_us_params["mlp__solver"],
                    alpha=ann_oil_us_params["mlp__alpha"],
                    learning_rate_init=ann_oil_us_params["mlp__learning_rate_init"],
                    batch_size=ann_oil_us_params["mlp__batch_size"],
                ),
            ),
        ]
    )
    ann_oil_us_pipeline.fit(X_train_oil_us, y_train_oil_us)
    ann_oil_us_forecast = pd.Series(
        ann_oil_us_pipeline.predict(X_test_oil_us),
        index=y_test_oil_us.index,
        name="ANN + oil + us-index",
    )
    ann_oil_us_metrics = regression_metrics(y_test_oil_us, ann_oil_us_forecast)

    # Compare reconstructed metrics to notebook metrics when available.
    notebook_metrics = notebook_info.get("metrics", {})
    reconstructed_metrics = {
        "ANN + oil": ann_oil_metrics,
        "ANN + oil + us-index": ann_oil_us_metrics,
        "SARIMAX + oil + us-index": sarimax_metrics,
        "ARIMA baseline": arima_metrics,
    }
    for model_name, metrics in reconstructed_metrics.items():
        verified = notebook_metrics.get(model_name)
        if verified is None:
            issues.append(f"{model_name}: notebook metrics were not extracted; using reconstructed metrics.")
            continue
        for metric_name, metric_value in metrics.items():
            if abs(metric_value - verified[metric_name]) > 1e-3:
                issues.append(
                    f"{model_name}: reconstructed {metric_name} ({metric_value:.4f}) differs from notebook value ({verified[metric_name]:.4f})."
                )
                break

    forecast_frame = pd.DataFrame(index=weekly_after.index)
    forecast_frame["Actual_Oilprice"] = weekly_after["Oilprice"]
    forecast_frame["ARIMA baseline"] = arima_forecast.reindex(forecast_frame.index)
    forecast_frame["SARIMAX + oil + us-index"] = sarimax_forecast.reindex(forecast_frame.index)
    forecast_frame["ANN + oil"] = ann_oil_forecast.reindex(forecast_frame.index)
    forecast_frame["ANN + oil + us-index"] = ann_oil_us_forecast.reindex(forecast_frame.index)

    return {
        "train_arima": train_arima,
        "test_arima": test_arima,
        "arima_forecast": arima_forecast,
        "train_endog": train_endog,
        "test_endog": test_endog,
        "sarimax_forecast": sarimax_forecast,
        "supervised_oil": supervised_oil,
        "X_oil": X_oil,
        "y_oil": y_oil,
        "X_train_oil": X_train_oil,
        "X_test_oil": X_test_oil,
        "y_train_oil": y_train_oil,
        "y_test_oil": y_test_oil,
        "ann_oil_forecast": ann_oil_forecast,
        "supervised_oil_us": supervised_oil_us,
        "X_oil_us": X_oil_us,
        "y_oil_us": y_oil_us,
        "X_train_oil_us": X_train_oil_us,
        "X_test_oil_us": X_test_oil_us,
        "y_train_oil_us": y_train_oil_us,
        "y_test_oil_us": y_test_oil_us,
        "ann_oil_us_forecast": ann_oil_us_forecast,
        "metrics_reconstructed": reconstructed_metrics,
        "forecast_frame": forecast_frame,
    }


def create_tables(
    datasets: dict[str, object],
    models: dict[str, object],
    notebook_info: dict[str, object],
    created: dict[str, list[Path]],
) -> None:
    """Build and save all requested paper tables."""
    raw = datasets["raw"]
    weekly_before = datasets["weekly_before"]
    weekly_after = datasets["weekly_after"]
    outlier_stats = datasets["outlier_stats"]

    table_1 = build_summary_rows(
        [
            ("Number of rows", len(raw)),
            ("Number of columns", raw.shape[1]),
            ("Column names", ", ".join(raw.columns.tolist())),
            ("Start date", date_string(raw.index.min())),
            ("End date", date_string(raw.index.max())),
            ("Frequency before resampling", "Daily"),
            ("Target variable", "Oilprice"),
            ("Exogenous variable", "us-index"),
        ]
    )
    save_table(table_1, "table_1_raw_dataset_overview", "Table 1. Raw Dataset Overview", created, decimals=4)

    save_table(
        datasets["raw_missing_summary"],
        "table_2_missing_value_summary_before_treatment",
        "Table 2. Missing-Value Summary Before Treatment",
        created,
        decimals=2,
    )

    save_table(
        datasets["clean_missing_summary"],
        "table_3_missing_value_summary_after_treatment",
        "Table 3. Missing-Value Summary After Treatment",
        created,
        decimals=2,
    )

    table_4 = build_summary_rows(
        [
            ("Original frequency", "Daily"),
            ("Resampled frequency", "Weekly"),
            ("Raw daily row count", len(raw)),
            ("Weekly row count", len(weekly_before)),
            ("Aggregation rule used", "W-FRI mean"),
        ]
    )
    save_table(table_4, "table_4_weekly_aggregation_summary", "Table 4. Weekly Aggregation Summary", created, decimals=4)

    table_5 = build_summary_rows([(key, value) for key, value in outlier_stats.items()])
    save_table(
        table_5,
        "table_5_outlier_treatment_summary_weekly_oilprice",
        "Table 5. Outlier Treatment Summary for Weekly Oilprice",
        created,
        decimals=6,
    )

    arima_train = models["train_arima"]
    arima_test = models["test_arima"]
    ann_train = models["y_train_oil"]
    ann_test = models["y_test_oil"]
    table_6 = pd.DataFrame(
        [
            {
                "Modeling family": "ARIMA/SARIMAX",
                "Train start date": date_string(arima_train.index.min()),
                "Train end date": date_string(arima_train.index.max()),
                "Train observations": len(arima_train),
                "Test start date": date_string(arima_test.index.min()),
                "Test end date": date_string(arima_test.index.max()),
                "Test observations": len(arima_test),
            },
            {
                "Modeling family": "ANN",
                "Train start date": date_string(ann_train.index.min()),
                "Train end date": date_string(ann_train.index.max()),
                "Train observations": len(ann_train),
                "Test start date": date_string(ann_test.index.min()),
                "Test end date": date_string(ann_test.index.max()),
                "Test observations": len(ann_test),
            },
        ]
    )
    save_table(table_6, "table_6_train_test_split_summary", "Table 6. Train-Test Split Summary", created, decimals=4)

    table_7 = pd.DataFrame(
        [
            {
                "Model": "ARIMA baseline",
                "Model type": "ARIMA",
                "Inputs used": "Weekly Oilprice only",
                "Forecasting strategy": "Direct multi-step forecast over held-out test horizon",
                "Tuning approach": "Notebook-verified grid search over p,d,q with RMSE ranking",
                "Evaluation metrics": "RMSE, MAE, MAPE, R^2",
            },
            {
                "Model": "SARIMAX + oil + us-index",
                "Model type": "SARIMAX",
                "Inputs used": "Weekly Oilprice with weekly us-index exogenous input",
                "Forecasting strategy": "Direct multi-step forecast over held-out test horizon",
                "Tuning approach": "Notebook-verified search over order and seasonal_order with RMSE ranking",
                "Evaluation metrics": "RMSE, MAE, MAPE, R^2",
            },
            {
                "Model": "ANN + oil only",
                "Model type": "MLPRegressor in Pipeline",
                "Inputs used": "4 lagged Oilprice features",
                "Forecasting strategy": "Supervised one-step-ahead prediction on time-ordered test set",
                "Tuning approach": "RandomizedSearchCV with TimeSeriesSplit(n_splits=5)",
                "Evaluation metrics": "RMSE, MAE, MAPE, R^2",
            },
            {
                "Model": "ANN + oil + us-index",
                "Model type": "MLPRegressor in Pipeline",
                "Inputs used": "4 lagged Oilprice and 4 lagged us-index features",
                "Forecasting strategy": "Supervised one-step-ahead prediction on time-ordered test set",
                "Tuning approach": "RandomizedSearchCV with TimeSeriesSplit(n_splits=5)",
                "Evaluation metrics": "RMSE, MAE, MAPE, R^2",
            },
        ]
    )
    save_table(table_7, "table_7_model_configuration_summary", "Table 7. Model Configuration Summary", created, decimals=4)

    notebook_metrics = notebook_info.get("metrics", {})
    performance_rows = []
    model_order = [
        "ANN + oil",
        "ANN + oil + us-index",
        "SARIMAX + oil + us-index",
        "ARIMA baseline",
    ]
    for model_name in model_order:
        metrics = notebook_metrics.get(model_name, models["metrics_reconstructed"][model_name])
        performance_rows.append({"Model": model_name, **metrics})
    table_8 = pd.DataFrame(performance_rows)
    save_table(
        table_8,
        "table_8_final_model_performance_comparison",
        "Table 8. Final Model Performance Comparison",
        created,
        decimals=4,
    )

    save_dataframe_csv(models["forecast_frame"], "model_forecasts_reconstructed.csv", created)


def create_figures(
    datasets: dict[str, object],
    models: dict[str, object],
    notebook_info: dict[str, object],
    created: dict[str, list[Path]],
    issues: list[str],
) -> None:
    """Build and save all requested figures."""
    raw = datasets["raw"]
    weekly_before = datasets["weekly_before"]
    weekly_after = datasets["weekly_after"]

    # Figure 1
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    line_1 = ax1.plot(raw.index, raw["Oilprice"], color="#1f4e79", linewidth=1.2, label="Oilprice")
    line_2 = ax2.plot(raw.index, raw["us-index"], color="#c44e52", linewidth=1.2, label="us-index")
    ax1.set_title("Figure 1. Raw Daily Time-Series Plot")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Oilprice")
    ax2.set_ylabel("us-index")
    lines = line_1 + line_2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    save_figure(fig, "figure_1_raw_daily_timeseries.png", created)

    # Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(raw["Oilprice"].dropna(), bins=35, color="#1f4e79", edgecolor="white", alpha=0.9)
    axes[0].set_title("Oilprice Distribution")
    axes[0].set_xlabel("Oilprice")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(raw["us-index"].dropna(), bins=35, color="#c44e52", edgecolor="white", alpha=0.9)
    axes[1].set_title("us-index Distribution")
    axes[1].set_xlabel("us-index")
    axes[1].set_ylabel("Frequency")
    save_figure(fig, "figure_2_raw_distributions.png", created)

    # Figure 3
    missing_counts = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(missing_counts.index, missing_counts.values, color=["#1f4e79", "#c44e52"])
    ax.set_title("Figure 3. Missing Values Before Treatment")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Missing count")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}", ha="center", va="bottom")
    save_figure(fig, "figure_3_missing_values_before_treatment.png", created)

    # Figure 4
    daily_corr = raw[["Oilprice", "us-index"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_annotated_heatmap(ax, daily_corr, "Figure 4. Daily Correlation Heatmap")
    save_figure(fig, "figure_4_daily_correlation_heatmap.png", created)

    # Figure 5
    lagged_corr = pd.DataFrame(
        {
            "Lag": list(range(LAGGED_CORR_MAX_LAG + 1)),
            "Correlation": [raw["Oilprice"].corr(raw["us-index"].shift(lag)) for lag in range(LAGGED_CORR_MAX_LAG + 1)],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lagged_corr["Lag"], lagged_corr["Correlation"], color="#1f4e79", marker="o", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Figure 5. Lagged Correlation Between Oilprice and Lagged us-index")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.set_xticks(lagged_corr["Lag"])
    save_figure(fig, "figure_5_lagged_correlation.png", created)

    # Figure 6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(
        weekly_before["Oilprice"].dropna(),
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#8fbcd4"},
        medianprops={"color": "#2f2f2f"},
    )
    ax.set_title("Figure 6. Weekly Oilprice Boxplot Before Outlier Treatment")
    ax.set_xlabel("Oilprice")
    save_figure(fig, "figure_6_weekly_oilprice_boxplot_before.png", created)

    # Figure 7
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(
        weekly_after["Oilprice"].dropna(),
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#9bc59d"},
        medianprops={"color": "#2f2f2f"},
    )
    ax.set_title("Figure 7. Weekly Oilprice Boxplot After Outlier Treatment")
    ax.set_xlabel("Oilprice")
    save_figure(fig, "figure_7_weekly_oilprice_boxplot_after.png", created)

    # Figure 8
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    line_1 = ax1.plot(weekly_after.index, weekly_after["Oilprice"], color="#1f4e79", linewidth=1.3, label="Weekly Oilprice")
    line_2 = ax2.plot(weekly_after.index, weekly_after["us-index"], color="#c44e52", linewidth=1.3, label="Weekly us-index")
    ax1.set_title("Figure 8. Weekly Time Series After Preprocessing")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Oilprice")
    ax2.set_ylabel("us-index")
    lines = line_1 + line_2
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    save_figure(fig, "figure_8_weekly_timeseries_after_preprocessing.png", created)

    # Figure 9
    weekly_corr = weekly_after[["Oilprice", "us-index"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_annotated_heatmap(ax, weekly_corr, "Figure 9. Weekly Correlation Heatmap After Preprocessing")
    save_figure(fig, "figure_9_weekly_correlation_heatmap.png", created)

    # Figure 10
    notebook_metrics = notebook_info.get("metrics", {})
    performance_df = pd.DataFrame(
        [
            {"Model": "ANN + oil", **notebook_metrics.get("ANN + oil", models["metrics_reconstructed"]["ANN + oil"])},
            {
                "Model": "ANN + oil + us-index",
                **notebook_metrics.get("ANN + oil + us-index", models["metrics_reconstructed"]["ANN + oil + us-index"]),
            },
            {
                "Model": "SARIMAX + oil + us-index",
                **notebook_metrics.get(
                    "SARIMAX + oil + us-index",
                    models["metrics_reconstructed"]["SARIMAX + oil + us-index"],
                ),
            },
            {"Model": "ARIMA baseline", **notebook_metrics.get("ARIMA baseline", models["metrics_reconstructed"]["ARIMA baseline"])},
        ]
    )
    metric_names = ["RMSE", "MAE", "MAPE", "R^2"]
    colors = ["#1f4e79", "#c44e52", "#7a8b99", "#6aa56a"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric_name in zip(axes.flatten(), metric_names):
        ax.bar(performance_df["Model"], performance_df[metric_name], color=colors)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Figure 10. Final Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    save_figure(fig, "figure_10_model_performance_comparison.png", created)

    # Figure 11
    try:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(models["y_train_oil"].index, models["y_train_oil"], label="Train", color="#7a8b99", linewidth=1.1)
        ax.plot(models["y_test_oil"].index, models["y_test_oil"], label="Actual test", color="#1f1f1f", linewidth=1.4)
        ax.plot(
            models["ann_oil_forecast"].index,
            models["ann_oil_forecast"],
            label="ANN + oil predicted",
            color="#1f4e79",
            linewidth=1.6,
        )
        ax.set_title("Figure 11. Actual vs Predicted for the Best Model (ANN + oil)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Oilprice")
        ax.legend()
        save_figure(fig, "figure_11_best_model_actual_vs_predicted.png", created)
    except Exception as exc:
        issues.append(f"Figure 11 could not be created: {exc}")

    # Figure 12
    try:
        comparison_start = min(
            models["test_arima"].index.min(),
            models["y_test_oil"].index.min(),
            models["y_test_oil_us"].index.min(),
        )
        comparison_actual = weekly_after.loc[comparison_start:, "Oilprice"]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(comparison_actual.index, comparison_actual, label="Actual weekly Oilprice", color="#1f1f1f", linewidth=1.5)
        ax.plot(models["arima_forecast"].index, models["arima_forecast"], label="ARIMA baseline", color="#7a8b99", linewidth=1.3)
        ax.plot(
            models["sarimax_forecast"].index,
            models["sarimax_forecast"],
            label="SARIMAX + oil + us-index",
            color="#c44e52",
            linewidth=1.4,
        )
        ax.plot(models["ann_oil_forecast"].index, models["ann_oil_forecast"], label="ANN + oil", color="#1f4e79", linewidth=1.4)
        ax.plot(
            models["ann_oil_us_forecast"].index,
            models["ann_oil_us_forecast"],
            label="ANN + oil + us-index",
            color="#6aa56a",
            linewidth=1.4,
        )
        ax.set_title("Figure 12. Actual vs Predicted Comparison Across All Models")
        ax.set_xlabel("Date")
        ax.set_ylabel("Oilprice")
        ax.legend(ncol=2)
        save_figure(fig, "figure_12_all_models_forecast_comparison.png", created)
    except Exception as exc:
        issues.append(f"Figure 12 could not be created: {exc}")


def print_summary(
    created: dict[str, list[Path]],
    issues: list[str],
    notebook_info: dict[str, object],
) -> None:
    """Print a short terminal log of generated outputs."""
    print("Created files:")
    for category in ("processed_data", "tables", "figures"):
        for path in created[category]:
            print(f"  [{category.upper()}] {path.relative_to(PROJECT_ROOT)}")

    print()
    print(f"Number of tables generated: {len(created['tables']) // 2}")
    print(f"Number of figures generated: {len(created['figures'])}")
    print(f"Processed data files generated: {len(created['processed_data'])}")
    print(f"Tables path: {TABLES_DIR}")
    print(f"Figures path: {FIGURES_DIR}")
    print(f"Processed data path: {PROCESSED_DIR}")
    print()
    print("Notebook inspection notes:")
    for note in notebook_info.get("notes", []):
        print(f"  - {note}")
    print()
    if issues:
        print("Items that could not be reproduced exactly from the notebook:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Items that could not be reproduced exactly from the notebook:")
        print("  - Forecast arrays were not stored explicitly in notebook outputs; reconstructed from verified notebook configurations.")


def main() -> None:
    """Run the full table and figure generation workflow."""
    configure_style()
    ensure_directories()

    created: dict[str, list[Path]] = {
        "tables": [],
        "figures": [],
        "processed_data": [],
    }
    issues: list[str] = []

    notebook_info = inspect_notebook(NOTEBOOK_PATH)
    datasets = prepare_datasets(created)
    models = fit_models(datasets["weekly_after"], notebook_info, issues)

    create_tables(datasets, models, notebook_info, created)
    create_figures(datasets, models, notebook_info, created, issues)

    print_summary(created, issues, notebook_info)


if __name__ == "__main__":
    main()
