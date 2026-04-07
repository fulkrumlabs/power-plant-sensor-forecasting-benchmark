from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

WINDOW_LABELS = {"3": "15m", "6": "30m", "12": "60m"}
TIME_FEATURE_LABELS = {
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "day": "Day of month",
    "month": "Month",
    "hour_sin": "Hour cyclical sine",
    "hour_cos": "Hour cyclical cosine",
    "dow_sin": "Weekday cyclical sine",
    "dow_cos": "Weekday cyclical cosine",
}


def handle_outliers(series: pd.Series, multiplier: float = 3) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)


def create_features(df: pd.DataFrame, target_col: str = "target", max_lag: int = 6) -> pd.DataFrame:
    df_feat = df.copy()

    for lag in range(1, max_lag + 1):
        df_feat[f"{target_col}_lag_{lag}"] = df_feat[target_col].shift(lag)

    for window in [3, 6, 12]:
        df_feat[f"{target_col}_roll_mean_{window}"] = df_feat[target_col].shift(1).rolling(window).mean()
        df_feat[f"{target_col}_roll_std_{window}"] = df_feat[target_col].shift(1).rolling(window).std()
        df_feat[f"{target_col}_roll_min_{window}"] = df_feat[target_col].shift(1).rolling(window).min()
        df_feat[f"{target_col}_roll_max_{window}"] = df_feat[target_col].shift(1).rolling(window).max()

    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["day"] = df_feat.index.day
    df_feat["month"] = df_feat.index.month

    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["dow_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)

    return df_feat


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2), "MAPE": mape}


def classify_trend(current: float, future: float, threshold: float = 0.02) -> int:
    pct_change = (future - current) / (current + 1e-10)
    if pct_change > threshold:
        return 1
    if pct_change < -threshold:
        return 0
    return 2


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def class_distribution(series: pd.Series) -> dict[str, int]:
    label_map = {0: "Down", 1: "Up", 2: "Steady"}
    counts = series.value_counts().sort_index()
    return {label_map.get(int(key), str(key)): int(value) for key, value in counts.items()}


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_public_aliases(
    heuristic_target: str,
    target_variable: str,
    related_vars: list[str],
    analysis_candidates: list[str],
) -> dict[str, str]:
    alias_map = {
        heuristic_target: "Sparse candidate signal",
        target_variable: "Selected target sensor",
    }

    support_index = 0
    for name in related_vars:
        if name not in alias_map:
            alias_map[name] = f"Support sensor {chr(65 + support_index)}"
            support_index += 1

    candidate_index = 1
    for name in analysis_candidates:
        if name not in alias_map:
            alias_map[name] = f"Candidate sensor {candidate_index}"
            candidate_index += 1

    return alias_map


def feature_display_label(name: str, alias_map: dict[str, str], public_safe: bool) -> str:
    if not public_safe:
        return name

    if name in alias_map:
        return alias_map[name]

    if name.startswith("target_lag_"):
        lag = name.split("_")[-1]
        return f"Target lag t-{lag}"

    if name.startswith("target_roll_"):
        parts = name.split("_")
        stat = parts[2]
        window = WINDOW_LABELS.get(parts[3], f"{parts[3]} steps")
        return f"Target rolling {stat} ({window})"

    return TIME_FEATURE_LABELS.get(name, name.replace("_", " ").title())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated charts and metrics")
    parser.add_argument("--timestamp-column", required=True, help="Private timestamp column name")
    parser.add_argument("--target", help="Optional target column override")
    parser.add_argument("--public-safe", action="store_true", help="Mask private sensor headers in outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column])
    df = df.rename(columns={args.timestamp_column: "timestamp"}).set_index("timestamp").sort_index()

    analysis_results: list[dict[str, object]] = []
    for col in df.columns:
        if df[col].std() == 0:
            continue

        mean_val = df[col].mean()
        std_dev = df[col].std()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
        outlier_pct = (outliers / len(df)) * 100

        try:
            adf_result = adfuller(df[col].dropna())
            adf_pvalue = float(adf_result[1])
            is_stationary = adf_pvalue < 0.05
        except Exception:
            adf_pvalue = float("nan")
            is_stationary = False

        analysis_results.append(
            {
                "Variable": col,
                "Mean": float(mean_val),
                "Std_Dev": float(std_dev),
                "CV": float((std_dev / mean_val) if mean_val != 0 else 0),
                "ACF_Lag1": float(df[col].autocorr(lag=1)),
                "ACF_Lag2": float(df[col].autocorr(lag=2)),
                "Outlier_%": float(outlier_pct),
                "ADF_pvalue": adf_pvalue,
                "Stationary": bool(is_stationary),
            }
        )

    analysis_df = pd.DataFrame(analysis_results).sort_values("ACF_Lag1", ascending=False)
    top_candidates = analysis_df[
        (analysis_df["ACF_Lag1"] > 0.8)
        & (analysis_df["CV"] > 0.01)
        & (analysis_df["Outlier_%"] < 10)
    ].head(5)

    if top_candidates.empty:
        raise RuntimeError("No target candidates satisfied the notebook criteria.")

    heuristic_target = str(top_candidates.iloc[0]["Variable"])
    target_variable = args.target or heuristic_target
    if target_variable not in df.columns:
        raise RuntimeError(f"Target column not found: {target_variable}")
    target_series = df[target_variable]
    target_label = "Selected target sensor" if args.public_safe else target_variable

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].plot(target_series.index, target_series.values, linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title(f"{target_label} - Complete Time Series", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Value")

    week_sample = target_series.loc["2005-02-01":"2005-02-07"]
    axes[0, 1].plot(week_sample.index, week_sample.values, linewidth=1.5, marker="o", markersize=2)
    axes[0, 1].set_title("One Week Detail (Feb 1-7, 2005)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Value")

    axes[0, 2].hist(target_series.dropna(), bins=50, edgecolor="black", alpha=0.7)
    axes[0, 2].axvline(target_series.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {target_series.mean():.2f}")
    axes[0, 2].axvline(target_series.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {target_series.median():.2f}")
    axes[0, 2].set_title("Distribution", fontsize=12, fontweight="bold")
    axes[0, 2].set_xlabel("Value")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend()

    axes[1, 0].boxplot(target_series.dropna(), vert=True)
    axes[1, 0].set_title("Box Plot (Outlier Detection)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Value")

    plot_acf(target_series.dropna(), lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("Autocorrelation Function (ACF)", fontsize=12, fontweight="bold")

    plot_pacf(target_series.dropna(), lags=40, ax=axes[1, 2], alpha=0.05)
    axes[1, 2].set_title("Partial Autocorrelation (PACF)", fontsize=12, fontweight="bold")
    save_figure(output_dir / "target-series-overview.png")

    target_df = pd.DataFrame(
        {
            "value": target_series,
            "hour": target_series.index.hour,
            "day_of_week": target_series.index.dayofweek,
            "month": target_series.index.month,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hourly_avg = target_df.groupby("hour")["value"].agg(["mean", "std"])
    axes[0].plot(hourly_avg.index, hourly_avg["mean"], marker="o", linewidth=2, markersize=6)
    axes[0].fill_between(
        hourly_avg.index,
        hourly_avg["mean"] - hourly_avg["std"],
        hourly_avg["mean"] + hourly_avg["std"],
        alpha=0.3,
    )
    axes[0].set_title("Average by Hour of Day", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Average Value")

    dow_avg = target_df.groupby("day_of_week")["value"].agg(["mean", "std"])
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(range(7), dow_avg["mean"], yerr=dow_avg["std"], capsize=5, alpha=0.7, edgecolor="black")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_names)
    axes[1].set_title("Average by Day of Week", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Average Value")

    monthly_avg = target_df.groupby("month")["value"].agg(["mean", "std"])
    axes[2].plot(monthly_avg.index, monthly_avg["mean"], marker="s", linewidth=2, markersize=8)
    axes[2].fill_between(
        monthly_avg.index,
        monthly_avg["mean"] - monthly_avg["std"],
        monthly_avg["mean"] + monthly_avg["std"],
        alpha=0.3,
    )
    axes[2].set_title("Average by Month", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Average Value")
    axes[2].set_xticks(range(1, 7))
    axes[2].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
    save_figure(output_dir / "temporal-patterns.png")

    correlation_matrix = df.corr(numeric_only=True)
    target_correlations = correlation_matrix[target_variable].sort_values(ascending=False)
    top_corr_vars = target_correlations.head(6).index.tolist()[1:]
    public_aliases = build_public_aliases(
        heuristic_target,
        target_variable,
        target_correlations.head(10).index.tolist(),
        top_candidates["Variable"].tolist(),
    )

    fig, axes = plt.subplots(1, min(5, len(top_corr_vars)), figsize=(16, 3))
    if len(top_corr_vars) == 1:
        axes = [axes]
    for idx, var in enumerate(top_corr_vars[:5]):
        axes[idx].scatter(df[var], df[target_variable], alpha=0.2, s=1)
        axes[idx].set_xlabel(public_aliases.get(var, var) if args.public_safe else var, fontsize=9)
        axes[idx].set_ylabel(target_label, fontsize=9)
        axes[idx].set_title(f"Corr: {correlation_matrix.loc[target_variable, var]:.3f}", fontsize=10, fontweight="bold")
    save_figure(output_dir / "sensor-correlations.png")

    target_clean = handle_outliers(target_series.copy())
    df_model = pd.DataFrame({"target": target_clean})
    for var in target_correlations.head(6).index[1:6]:
        df_model[var] = handle_outliers(df[var].copy())

    df_features = create_features(df_model, target_col="target", max_lag=6).dropna()
    df_features["target_5min"] = df_features["target"].shift(-1)
    df_features["target_10min"] = df_features["target"].shift(-2)
    df_features = df_features.dropna()

    n = len(df_features)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)

    train_data = df_features.iloc[:train_size].copy()
    val_data = df_features.iloc[train_size : train_size + val_size].copy()
    test_data = df_features.iloc[train_size + val_size :].copy()

    plt.figure(figsize=(16, 4))
    plt.plot(train_data.index, train_data["target"], label="Train", alpha=0.7, linewidth=0.5)
    plt.plot(val_data.index, val_data["target"], label="Validation", alpha=0.7, linewidth=0.5)
    plt.plot(test_data.index, test_data["target"], label="Test", alpha=0.7, linewidth=0.5)
    plt.xlabel("Date")
    plt.ylabel(target_label)
    plt.title("Train/Validation/Test Split", fontsize=14, fontweight="bold")
    plt.legend()
    save_figure(output_dir / "dataset-split.png")

    feature_cols = [col for col in df_features.columns if col not in ["target", "target_5min", "target_10min"]]
    X_train_5 = train_data[feature_cols]
    y_train_5 = train_data["target_5min"]
    X_val_5 = val_data[feature_cols]
    y_val_5 = val_data["target_5min"]
    X_test_5 = test_data[feature_cols]
    y_test_5 = test_data["target_5min"]

    X_train_10 = train_data[feature_cols]
    y_train_10 = train_data["target_10min"]
    X_val_10 = val_data[feature_cols]
    y_val_10 = val_data["target_10min"]
    X_test_10 = test_data[feature_cols]
    y_test_10 = test_data["target_10min"]

    scaler_5 = StandardScaler()
    X_train_5_scaled = scaler_5.fit_transform(X_train_5)
    X_val_5_scaled = scaler_5.transform(X_val_5)
    X_test_5_scaled = scaler_5.transform(X_test_5)

    scaler_10 = StandardScaler()
    X_train_10_scaled = scaler_10.fit_transform(X_train_10)
    X_val_10_scaled = scaler_10.transform(X_val_10)
    X_test_10_scaled = scaler_10.transform(X_test_10)

    results_reg = {"5min": {}, "10min": {}}
    y_baseline_5 = test_data["target"].values
    y_baseline_10 = test_data["target"].values
    results_reg["5min"]["baseline"] = evaluate_regression(y_test_5, y_baseline_5[: len(y_test_5)])
    results_reg["10min"]["baseline"] = evaluate_regression(y_test_10, y_baseline_10[: len(y_test_10)])

    rf_5 = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_5.fit(X_train_5_scaled, y_train_5)
    y_pred_rf_5_val = rf_5.predict(X_val_5_scaled)
    y_pred_rf_5_test = rf_5.predict(X_test_5_scaled)
    results_reg["5min"]["rf_val"] = evaluate_regression(y_val_5, y_pred_rf_5_val)
    results_reg["5min"]["rf_test"] = evaluate_regression(y_test_5, y_pred_rf_5_test)

    rf_10 = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_10.fit(X_train_10_scaled, y_train_10)
    y_pred_rf_10_val = rf_10.predict(X_val_10_scaled)
    y_pred_rf_10_test = rf_10.predict(X_test_10_scaled)
    results_reg["10min"]["rf_val"] = evaluate_regression(y_val_10, y_pred_rf_10_val)
    results_reg["10min"]["rf_test"] = evaluate_regression(y_test_10, y_pred_rf_10_test)

    xgb_5 = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    xgb_5.fit(X_train_5_scaled, y_train_5)
    y_pred_xgb_5_val = xgb_5.predict(X_val_5_scaled)
    y_pred_xgb_5_test = xgb_5.predict(X_test_5_scaled)
    results_reg["5min"]["xgb_val"] = evaluate_regression(y_val_5, y_pred_xgb_5_val)
    results_reg["5min"]["xgb_test"] = evaluate_regression(y_test_5, y_pred_xgb_5_test)

    xgb_10 = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    xgb_10.fit(X_train_10_scaled, y_train_10)
    y_pred_xgb_10_val = xgb_10.predict(X_val_10_scaled)
    y_pred_xgb_10_test = xgb_10.predict(X_test_10_scaled)
    results_reg["10min"]["xgb_val"] = evaluate_regression(y_val_10, y_pred_xgb_10_val)
    results_reg["10min"]["xgb_test"] = evaluate_regression(y_test_10, y_pred_xgb_10_test)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sample = slice(0, 500)
    axes[0, 0].plot(y_val_5.values[sample], label="Actual", linewidth=2, alpha=0.8)
    axes[0, 0].plot(y_pred_rf_5_val[sample], label="Random Forest", linewidth=1.5, alpha=0.7)
    axes[0, 0].plot(y_pred_xgb_5_val[sample], label="XGBoost", linewidth=1.5, alpha=0.7)
    axes[0, 0].set_title("5-Min Forecast - Validation", fontsize=12, fontweight="bold")
    axes[0, 0].legend()

    axes[0, 1].plot(y_test_5.values[sample], label="Actual", linewidth=2, alpha=0.8)
    axes[0, 1].plot(y_pred_rf_5_test[sample], label="Random Forest", linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(y_pred_xgb_5_test[sample], label="XGBoost", linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title("5-Min Forecast - Test", fontsize=12, fontweight="bold")
    axes[0, 1].legend()

    axes[1, 0].plot(y_val_10.values[sample], label="Actual", linewidth=2, alpha=0.8)
    axes[1, 0].plot(y_pred_rf_10_val[sample], label="Random Forest", linewidth=1.5, alpha=0.7)
    axes[1, 0].plot(y_pred_xgb_10_val[sample], label="XGBoost", linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title("10-Min Forecast - Validation", fontsize=12, fontweight="bold")
    axes[1, 0].legend()

    axes[1, 1].plot(y_test_10.values[sample], label="Actual", linewidth=2, alpha=0.8)
    axes[1, 1].plot(y_pred_rf_10_test[sample], label="Random Forest", linewidth=1.5, alpha=0.7)
    axes[1, 1].plot(y_pred_xgb_10_test[sample], label="XGBoost", linewidth=1.5, alpha=0.7)
    axes[1, 1].set_title("10-Min Forecast - Test", fontsize=12, fontweight="bold")
    axes[1, 1].legend()
    save_figure(output_dir / "forecast-vs-actual.png")

    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": rf_5.feature_importances_}).sort_values(
        "importance",
        ascending=False,
    )
    plt.figure(figsize=(10, 6))
    top15 = feat_imp.head(15)
    plt.barh(range(len(top15)), top15["importance"].values, alpha=0.8, edgecolor="black")
    plt.yticks(
        range(len(top15)),
        [feature_display_label(value, public_aliases, args.public_safe) for value in top15["feature"].values],
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance - Random Forest (5-min)", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    save_figure(output_dir / "feature-importance.png")

    train_data["trend_5"] = train_data.apply(lambda row: classify_trend(row["target"], row["target_5min"]), axis=1)
    train_data["trend_10"] = train_data.apply(lambda row: classify_trend(row["target"], row["target_10min"]), axis=1)
    val_data["trend_5"] = val_data.apply(lambda row: classify_trend(row["target"], row["target_5min"]), axis=1)
    val_data["trend_10"] = val_data.apply(lambda row: classify_trend(row["target"], row["target_10min"]), axis=1)
    test_data["trend_5"] = test_data.apply(lambda row: classify_trend(row["target"], row["target_5min"]), axis=1)
    test_data["trend_10"] = test_data.apply(lambda row: classify_trend(row["target"], row["target_10min"]), axis=1)

    y_train_trend_5 = train_data["trend_5"]
    y_val_trend_5 = val_data["trend_5"]
    y_test_trend_5 = test_data["trend_5"]
    y_train_trend_10 = train_data["trend_10"]
    y_val_trend_10 = val_data["trend_10"]
    y_test_trend_10 = test_data["trend_10"]

    results_clf = {"5min": {}, "10min": {}}
    classification_status: dict[str, dict[str, object]] = {}
    predictions: dict[str, dict[str, np.ndarray | None]] = {
        "5min": {"rf_test": None, "xgb_test": None},
        "10min": {"rf_test": None, "xgb_test": None},
    }

    for horizon, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test in [
        ("5min", X_train_5_scaled, X_val_5_scaled, X_test_5_scaled, y_train_trend_5, y_val_trend_5, y_test_trend_5),
        ("10min", X_train_10_scaled, X_val_10_scaled, X_test_10_scaled, y_train_trend_10, y_val_trend_10, y_test_trend_10),
    ]:
        train_classes = sorted(y_train.unique().tolist())
        status = {
            "train_distribution": class_distribution(y_train),
            "validation_distribution": class_distribution(y_val),
            "test_distribution": class_distribution(y_test),
            "unique_train_classes": train_classes,
            "viable": len(train_classes) >= 2,
        }
        classification_status[horizon] = status

        if len(train_classes) < 2:
            constant_class = int(train_classes[0])
            y_pred_val = np.full(len(y_val), constant_class)
            y_pred_test = np.full(len(y_test), constant_class)
            results_clf[horizon]["rf_val"] = evaluate_classification(y_val, y_pred_val)
            results_clf[horizon]["rf_test"] = evaluate_classification(y_test, y_pred_test)
            results_clf[horizon]["xgb_val"] = None
            results_clf[horizon]["xgb_test"] = None
            results_clf[horizon]["note"] = "Trend labels collapse to a single class at the 2% threshold, so supervised classification is not meaningful for this target."
            predictions[horizon]["rf_test"] = y_pred_test
            continue

        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
        rf_clf.fit(X_train_scaled, y_train)
        y_pred_rf_val = rf_clf.predict(X_val_scaled)
        y_pred_rf_test = rf_clf.predict(X_test_scaled)
        results_clf[horizon]["rf_val"] = evaluate_classification(y_val, y_pred_rf_val)
        results_clf[horizon]["rf_test"] = evaluate_classification(y_test, y_pred_rf_test)
        predictions[horizon]["rf_test"] = y_pred_rf_test

        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        xgb_clf.fit(X_train_scaled, y_train)
        y_pred_xgb_val = xgb_clf.predict(X_val_scaled)
        y_pred_xgb_test = xgb_clf.predict(X_test_scaled)
        results_clf[horizon]["xgb_val"] = evaluate_classification(y_val, y_pred_xgb_val)
        results_clf[horizon]["xgb_test"] = evaluate_classification(y_test, y_pred_xgb_test)
        predictions[horizon]["xgb_test"] = y_pred_xgb_test

    if any(status["viable"] for status in classification_status.values()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        chart_specs = [
            (axes[0, 0], y_test_trend_5, predictions["5min"]["rf_test"], "5-Min RF - Confusion Matrix", "Blues"),
            (axes[0, 1], y_test_trend_5, predictions["5min"]["xgb_test"], "5-Min XGB - Confusion Matrix", "Greens"),
            (axes[1, 0], y_test_trend_10, predictions["10min"]["rf_test"], "10-Min RF - Confusion Matrix", "Oranges"),
            (axes[1, 1], y_test_trend_10, predictions["10min"]["xgb_test"], "10-Min XGB - Confusion Matrix", "Purples"),
        ]
        for axis, truth, pred, title, cmap in chart_specs:
            axis.set_title(title, fontsize=12, fontweight="bold")
            if pred is None:
                axis.text(0.5, 0.5, "Not viable\n(single-class labels)", ha="center", va="center", fontsize=12)
                axis.axis("off")
                continue
            cm = confusion_matrix(truth, pred, labels=[0, 1, 2])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=cmap,
                ax=axis,
                xticklabels=["Down", "Up", "Steady"],
                yticklabels=["Down", "Up", "Steady"],
            )
            axis.set_ylabel("True")
            axis.set_xlabel("Predicted")
        save_figure(output_dir / "confusion-matrices.png")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        label_order = ["Down", "Up", "Steady"]
        for axis, horizon in zip(axes, ["5min", "10min"]):
            counts = classification_status[horizon]["train_distribution"]
            values = [counts.get(label, 0) for label in label_order]
            axis.bar(label_order, values, color=["#D96C6C", "#7CC37C", "#6D8BFF"], edgecolor="black")
            axis.set_title(f"{horizon} Trend Label Distribution", fontsize=12, fontweight="bold")
            axis.set_ylabel("Training rows")
        save_figure(output_dir / "trend-label-distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    models = ["Baseline", "Random Forest", "XGBoost"]
    r2_5 = [
        results_reg["5min"]["baseline"]["R2"],
        results_reg["5min"]["rf_test"]["R2"],
        results_reg["5min"]["xgb_test"]["R2"],
    ]
    r2_10 = [
        results_reg["10min"]["baseline"]["R2"],
        results_reg["10min"]["rf_test"]["R2"],
        results_reg["10min"]["xgb_test"]["R2"],
    ]
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width / 2, r2_5, width, label="5-min", alpha=0.8, edgecolor="black")
    axes[0].bar(x + width / 2, r2_10, width, label="10-min", alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Regression Models - R² Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()

    if any(status["viable"] for status in classification_status.values()):
        clf_models = ["Random Forest", "XGBoost"]
        acc_5 = [
            results_clf["5min"]["rf_test"]["Accuracy"],
            results_clf["5min"]["xgb_test"]["Accuracy"] if results_clf["5min"]["xgb_test"] else 0,
        ]
        acc_10 = [
            results_clf["10min"]["rf_test"]["Accuracy"],
            results_clf["10min"]["xgb_test"]["Accuracy"] if results_clf["10min"]["xgb_test"] else 0,
        ]
        x = np.arange(len(clf_models))
        axes[1].bar(x - width / 2, acc_5, width, label="5-min", alpha=0.8, edgecolor="black")
        axes[1].bar(x + width / 2, acc_10, width, label="10-min", alpha=0.8, edgecolor="black")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Classification Models - Accuracy Comparison", fontsize=14, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(clf_models)
        axes[1].legend()
    else:
        label_order = ["Down", "Up", "Steady"]
        counts_5 = classification_status["5min"]["train_distribution"]
        counts_10 = classification_status["10min"]["train_distribution"]
        x = np.arange(len(label_order))
        axes[1].bar(
            x - width / 2,
            [counts_5.get(label, 0) for label in label_order],
            width,
            label="5-min",
            alpha=0.8,
            edgecolor="black",
        )
        axes[1].bar(
            x + width / 2,
            [counts_10.get(label, 0) for label in label_order],
            width,
            label="10-min",
            alpha=0.8,
            edgecolor="black",
        )
        axes[1].set_xlabel("Trend label")
        axes[1].set_ylabel("Training rows")
        axes[1].set_title("Trend Labels Collapse at 2% Threshold", fontsize=14, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(label_order)
        axes[1].legend()
    save_figure(output_dir / "performance-comparison.png")

    best_r2_5 = max(results_reg["5min"]["rf_test"]["R2"], results_reg["5min"]["xgb_test"]["R2"])
    best_r2_10 = max(results_reg["10min"]["rf_test"]["R2"], results_reg["10min"]["xgb_test"]["R2"])
    viable_acc_5 = (
        [
            metric["Accuracy"]
            for metric in [results_clf["5min"]["rf_test"], results_clf["5min"]["xgb_test"]]
            if metric
        ]
        if classification_status["5min"]["viable"]
        else []
    )
    viable_acc_10 = (
        [
            metric["Accuracy"]
            for metric in [results_clf["10min"]["rf_test"], results_clf["10min"]["xgb_test"]]
            if metric
        ]
        if classification_status["10min"]["viable"]
        else []
    )
    best_acc_5 = max(viable_acc_5) if viable_acc_5 else None
    best_acc_10 = max(viable_acc_10) if viable_acc_10 else None

    baseline_rmse_5 = results_reg["5min"]["baseline"]["RMSE"]
    rf_improvement_5 = (
        ((baseline_rmse_5 - results_reg["5min"]["rf_test"]["RMSE"]) / baseline_rmse_5) * 100
        if baseline_rmse_5 != 0
        else None
    )
    xgb_improvement_5 = (
        ((baseline_rmse_5 - results_reg["5min"]["xgb_test"]["RMSE"]) / baseline_rmse_5) * 100
        if baseline_rmse_5 != 0
        else None
    )

    if args.public_safe:
        top_candidates_public = []
        for item in top_candidates.to_dict(orient="records"):
            public_item = dict(item)
            public_item["Variable"] = public_aliases.get(str(item["Variable"]), "Candidate sensor")
            top_candidates_public.append(public_item)
        top_correlations_public = {
            public_aliases.get(key, key): float(value)
            for key, value in target_correlations.head(10).to_dict().items()
        }
        top_importance_public = [
            {
                "feature": feature_display_label(str(item["feature"]), public_aliases, True),
                "importance": float(item["importance"]),
            }
            for item in feat_imp.head(15).to_dict(orient="records")
        ]
    else:
        top_candidates_public = top_candidates.to_dict(orient="records")
        top_correlations_public = {key: float(value) for key, value in target_correlations.head(10).to_dict().items()}
        top_importance_public = feat_imp.head(15).to_dict(orient="records")

    metrics = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "date_start": df.index.min().isoformat(),
            "date_end": df.index.max().isoformat(),
            "days": int((df.index.max() - df.index.min()).days),
            "missing_values": int(df.isnull().sum().sum()),
        },
        "target_selection": {
            "heuristic_target": public_aliases.get(heuristic_target, heuristic_target) if args.public_safe else heuristic_target,
            "target_variable": target_label if args.public_safe else target_variable,
            "target_overridden": bool(args.target),
            "top_candidates": top_candidates_public,
            "target_stats": {
                "mean": float(target_series.mean()),
                "median": float(target_series.median()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "skewness": float(target_series.skew()),
                "kurtosis": float(target_series.kurtosis()),
            },
            "top_correlations": top_correlations_public,
        },
        "features": {
            "feature_count": len(feature_cols),
            "top_15_importance": top_importance_public,
        },
        "splits": {
            "train_rows": int(len(train_data)),
            "validation_rows": int(len(val_data)),
            "test_rows": int(len(test_data)),
        },
        "regression": results_reg,
        "classification": {
            "metrics": results_clf,
            "status": classification_status,
            "reports": {
                "rf_5_test": classification_report(
                    y_test_trend_5,
                    predictions["5min"]["rf_test"],
                    labels=[0, 1, 2],
                    target_names=["Down", "Up", "Steady"],
                    output_dict=True,
                    zero_division=0,
                )
                if predictions["5min"]["rf_test"] is not None
                else None,
                "rf_10_test": classification_report(
                    y_test_trend_10,
                    predictions["10min"]["rf_test"],
                    labels=[0, 1, 2],
                    target_names=["Down", "Up", "Steady"],
                    output_dict=True,
                    zero_division=0,
                )
                if predictions["10min"]["rf_test"] is not None
                else None,
                "xgb_5_test": classification_report(
                    y_test_trend_5,
                    predictions["5min"]["xgb_test"],
                    labels=[0, 1, 2],
                    target_names=["Down", "Up", "Steady"],
                    output_dict=True,
                    zero_division=0,
                )
                if predictions["5min"]["xgb_test"] is not None
                else None,
                "xgb_10_test": classification_report(
                    y_test_trend_10,
                    predictions["10min"]["xgb_test"],
                    labels=[0, 1, 2],
                    target_names=["Down", "Up", "Steady"],
                    output_dict=True,
                    zero_division=0,
                )
                if predictions["10min"]["xgb_test"] is not None
                else None,
            },
        },
        "feasibility": {
            "best_r2_5min": float(best_r2_5),
            "best_r2_10min": float(best_r2_10),
            "best_accuracy_5min": float(best_acc_5) if best_acc_5 is not None else None,
            "best_accuracy_10min": float(best_acc_10) if best_acc_10 is not None else None,
            "best_rmse_improvement_5min_pct": (
                float(max(value for value in [rf_improvement_5, xgb_improvement_5] if value is not None))
                if any(value is not None for value in [rf_improvement_5, xgb_improvement_5])
                else None
            ),
        },
        "visuals": [
            "target-series-overview.png",
            "temporal-patterns.png",
            "sensor-correlations.png",
            "dataset-split.png",
            "forecast-vs-actual.png",
            "feature-importance.png",
            "performance-comparison.png",
        ],
    }

    if (output_dir / "confusion-matrices.png").exists():
        metrics["visuals"].append("confusion-matrices.png")
    if (output_dir / "trend-label-distribution.png").exists():
        metrics["visuals"].append("trend-label-distribution.png")

    metrics_path = output_dir / ("public-summary.json" if args.public_safe else "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
