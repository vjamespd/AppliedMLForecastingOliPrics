from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

DATA_FILE = Path("final_oil-us-indexv1.csv")
OUTPUT_DIR = Path("methodology_outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

MAX_LAG = 8
TRAIN_RATIO = 0.80
LAGS = 4


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str, title: str, decimals: int = 4) -> None:
    csv_path = TABLE_DIR / f"{name}.csv"
    png_path = TABLE_DIR / f"{name}.png"

    export_df = df.copy()
    export_df.to_csv(csv_path, index=False)

    display_df = export_df.copy()
    numeric_cols = display_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        display_df[numeric_cols] = display_df[numeric_cols].round(decimals)

    rows, cols = display_df.shape
    fig_height = max(2.2, 0.55 * (rows + 1))
    fig_width = max(8, 1.6 * cols)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=display_df.astype(str).values,
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "#ffffff")

    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_supervised_features(
    df: pd.DataFrame,
    target_col: str = "Oilprice",
    exog_cols: list[str] | None = None,
    lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if exog_cols is None:
        exog_cols = []

    supervised = pd.DataFrame(index=df.index)
    supervised[target_col] = df[target_col]

    for lag in range(1, lags + 1):
        supervised[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    for exog_col in exog_cols:
        for lag in range(1, lags + 1):
            supervised[f"{exog_col}_lag_{lag}"] = df[exog_col].shift(lag)

    supervised = supervised.dropna()
    X = supervised.drop(columns=[target_col])
    y = supervised[target_col]
    return supervised, X, y


def train_test_split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test


def save_raw_time_series(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(data.index, data["Oilprice"], color="#0f4c81", linewidth=1.3)
    axes[0].set_title("Raw Daily Oil Price")
    axes[0].set_ylabel("Oil Price")

    axes[1].plot(data.index, data["us-index"], color="#d1495b", linewidth=1.3)
    axes[1].set_title("Raw Daily US Index")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("US Index")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_raw_daily_time_series.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_raw_distributions(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(data["Oilprice"].dropna(), bins=30, color="#4e79a7", alpha=0.85, edgecolor="white")
    axes[0].set_title("Distribution of Oil Prices")
    axes[0].set_xlabel("Oil Price")

    axes[1].hist(data["us-index"].dropna(), bins=30, color="#e15759", alpha=0.85, edgecolor="white")
    axes[1].set_title("Distribution of US Index")
    axes[1].set_xlabel("US Index")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02_raw_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_lagged_correlation(data: pd.DataFrame) -> pd.DataFrame:
    lag_corr = []
    for lag in range(0, MAX_LAG + 1):
        corr_val = data["Oilprice"].corr(data["us-index"].shift(lag))
        lag_corr.append({"Lag_Weeks_or_Days": lag, "Correlation": corr_val})

    lag_corr_df = pd.DataFrame(lag_corr)
    save_table(
        lag_corr_df,
        "04_lagged_correlation_table",
        "Lagged Correlation Table: Oil Price vs Shifted US Index",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(lag_corr_df["Lag_Weeks_or_Days"], lag_corr_df["Correlation"], color="#59a14f")
    ax.set_title("Lagged Correlation: Oil Price vs Shifted US Index")
    ax.set_xlabel("Lag of US Index")
    ax.set_ylabel("Correlation")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_lagged_correlation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return lag_corr_df


def save_outlier_boxplots(weekly_data: pd.DataFrame, weekly_data_clean: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].boxplot(weekly_data["Oilprice"].dropna(), vert=False, patch_artist=True, boxprops={"facecolor": "skyblue"})
    axes[0].set_title("Oil Price Before IQR Capping")

    axes[1].boxplot(
        weekly_data_clean["Oilprice"].dropna(),
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "lightgreen"},
    )
    axes[1].set_title("Oil Price After IQR Capping")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_outlier_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_weekly_correlation_plots(weekly_data_clean: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = weekly_data_clean[["Oilprice", "us-index"]].corr()
    corr_weekly = corr_matrix.reset_index()
    corr_weekly.columns = ["Variable", "Oilprice", "us-index"]
    save_table(corr_weekly, "08_weekly_correlation_matrix", "Weekly Correlation Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}", ha="center", va="center", color="black")
    ax.set_title("Weekly Correlation Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "06_weekly_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(weekly_data_clean.index, weekly_data_clean["Oilprice"], color="#0f4c81", linewidth=1.3)
    axes[0].set_title("Weekly Oil Price (Cleaned)")
    axes[0].set_ylabel("Oil Price")

    axes[1].plot(weekly_data_clean.index, weekly_data_clean["us-index"], color="#d1495b", linewidth=1.3)
    axes[1].set_title("Weekly US Index")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("US Index")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "07_weekly_time_series_cleaned.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return corr_weekly


def save_train_test_split_plot(
    y_train: pd.Series,
    y_test: pd.Series,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_train.index, y_train, label="Train", color="#4e79a7", linewidth=1.5)
    ax.plot(y_test.index, y_test, label="Test", color="#e15759", linewidth=1.5)
    ax.axvline(y_test.index.min(), linestyle="--", color="black", linewidth=1, label="Split Point")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    ensure_output_dirs()

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").set_index("Date")

    raw_overview = pd.DataFrame(
        [
            {
                "Rows": len(data),
                "Columns": data.shape[1],
                "Start_Date": data.index.min().date(),
                "End_Date": data.index.max().date(),
                "Oilprice_Min": data["Oilprice"].min(),
                "Oilprice_Max": data["Oilprice"].max(),
                "US_Index_Min": data["us-index"].min(),
                "US_Index_Max": data["us-index"].max(),
            }
        ]
    )
    save_table(raw_overview, "01_raw_data_overview", "Raw Data Overview")

    raw_missing = data.isna().sum().rename("Missing_Count").reset_index()
    raw_missing.columns = ["Column", "Missing_Count"]
    save_table(raw_missing, "02_raw_missing_values", "Missing Values Before Imputation")

    raw_describe = data.describe().T.reset_index()
    raw_describe.columns = ["Variable", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    save_table(raw_describe, "03_raw_descriptive_statistics", "Raw Descriptive Statistics")

    save_raw_time_series(data)
    save_raw_distributions(data)
    save_lagged_correlation(data)

    data_imputed = data.copy()
    data_imputed["Oilprice"] = data_imputed["Oilprice"].interpolate(method="linear")
    data_imputed["us-index"] = data_imputed["us-index"].interpolate(method="linear")
    data_imputed = data_imputed.ffill().bfill()

    imputed_missing = data_imputed.isna().sum().rename("Missing_Count").reset_index()
    imputed_missing.columns = ["Column", "Missing_Count"]
    save_table(imputed_missing, "05_missing_values_after_imputation", "Missing Values After Imputation")

    weekly_data = data_imputed.resample("W-FRI").mean()
    weekly_preview = weekly_data.head().reset_index()
    save_table(weekly_preview, "06_weekly_preview", "Weekly Aggregation Preview")

    q1 = weekly_data["Oilprice"].quantile(0.25)
    q3 = weekly_data["Oilprice"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = weekly_data[
        (weekly_data["Oilprice"] < lower_bound) | (weekly_data["Oilprice"] > upper_bound)
    ].copy()

    outlier_stats = pd.DataFrame(
        [
            {
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
                "Number_of_Outliers": len(outliers),
            }
        ]
    )
    save_table(outlier_stats, "07_outlier_statistics", "Weekly Oil Price Outlier Statistics")

    outlier_rows = outliers.reset_index()
    if outlier_rows.empty:
        outlier_rows = pd.DataFrame([{"Date": "None", "Oilprice": np.nan, "us-index": np.nan}])
    save_table(outlier_rows, "07b_outlier_rows", "Detected Weekly Oil Price Outliers")

    weekly_data_clean = weekly_data.copy()
    weekly_data_clean["Oilprice"] = np.clip(weekly_data_clean["Oilprice"], lower_bound, upper_bound)

    save_outlier_boxplots(weekly_data, weekly_data_clean)
    save_weekly_correlation_plots(weekly_data_clean)

    supervised_oil_only, X_oil_only, y_oil_only = create_supervised_features(
        weekly_data_clean,
        target_col="Oilprice",
        exog_cols=[],
        lags=LAGS,
    )
    X_train_oil, X_test_oil, y_train_oil, y_test_oil = train_test_split_time_series(
        X_oil_only,
        y_oil_only,
        train_ratio=TRAIN_RATIO,
    )

    supervised_oil_us, X_oil_us, y_oil_us = create_supervised_features(
        weekly_data_clean,
        target_col="Oilprice",
        exog_cols=["us-index"],
        lags=LAGS,
    )
    X_train_oil_us, X_test_oil_us, y_train_oil_us, y_test_oil_us = train_test_split_time_series(
        X_oil_us,
        y_oil_us,
        train_ratio=TRAIN_RATIO,
    )

    supervised_summary = pd.DataFrame(
        [
            {
                "Experiment": "ANN + oil only",
                "Rows": len(supervised_oil_only),
                "Feature_Count": X_oil_only.shape[1],
                "Train_Rows": len(X_train_oil),
                "Test_Rows": len(X_test_oil),
                "Lag_Count": LAGS,
                "Exogenous_Input": "None",
            },
            {
                "Experiment": "ANN + oil + US Index",
                "Rows": len(supervised_oil_us),
                "Feature_Count": X_oil_us.shape[1],
                "Train_Rows": len(X_train_oil_us),
                "Test_Rows": len(X_test_oil_us),
                "Lag_Count": LAGS,
                "Exogenous_Input": "us-index",
            },
        ]
    )
    save_table(
        supervised_summary,
        "09_supervised_dataset_summary",
        "Supervised Dataset Summary for ANN Experiments",
    )

    save_table(
        supervised_oil_only.head(8).reset_index(),
        "10_oil_only_supervised_preview",
        "Oil-Only Supervised Dataset Preview",
    )
    save_table(
        supervised_oil_us.head(8).reset_index(),
        "11_oil_us_supervised_preview",
        "Oil + US Index Supervised Dataset Preview",
    )

    save_train_test_split_plot(
        y_train_oil,
        y_test_oil,
        "Time-Aware Train/Test Split: Oil-Only Experiment",
        "08_train_test_split_oil_only.png",
    )
    save_train_test_split_plot(
        y_train_oil_us,
        y_test_oil_us,
        "Time-Aware Train/Test Split: Oil + US Index Experiment",
        "09_train_test_split_oil_us.png",
    )

    print(f"Methodology tables saved to: {TABLE_DIR.resolve()}")
    print(f"Methodology figures saved to: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
