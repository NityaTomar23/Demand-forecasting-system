"""
Inference helpers for one-step-ahead demand forecasts.

The training pipeline expects engineered lag and rolling features. This module
derives those features from the latest available history so the API and
dashboard can offer an actual forecast flow instead of requiring raw feature
vectors only.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

DEFAULT_LAGS = (7, 14, 28)
DEFAULT_WINDOWS = (7, 14, 30)


def load_sales_history(path: str) -> pd.DataFrame:
    """Load raw sales history used to derive forecast features."""
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    df.sort_values(["store", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _normalize_forecast_date(value: str | date | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def _encode_label(label: str, values: list[str], field_name: str) -> int:
    if label not in values:
        raise ValueError(f"Unknown {field_name} '{label}'. Expected one of: {values}")
    return values.index(label)


def build_next_day_features(
    sales_df: pd.DataFrame,
    metadata: dict,
    store: str,
    promotion: int = 0,
    holiday: int | None = None,
    forecast_date: str | date | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, dict[str, float | int]]:
    """Build a model-ready feature payload for the next available day only."""
    series_df = (
        sales_df[(sales_df["store"] == store)]
        .sort_values("date")
        .copy()
    )
    if series_df.empty:
        raise ValueError(f"No sales history found for store={store}.")

    last_observed = series_df["date"].max().normalize()
    expected_forecast_date = last_observed + pd.Timedelta(days=1)
    requested_forecast_date = _normalize_forecast_date(forecast_date) or expected_forecast_date

    if requested_forecast_date != expected_forecast_date:
        raise ValueError(
            "This project currently supports next-day forecasting only. "
            f"Use forecast_date={expected_forecast_date.date()} for {store}."
        )

    history = series_df.set_index("date")["sales"].groupby(level=0).mean().sort_index().asfreq("D")
    if history.isna().any():
        raise ValueError("Sales history has missing dates; cannot derive lag features safely.")

    required_history = max(max(DEFAULT_LAGS), max(DEFAULT_WINDOWS))
    if len(history) < required_history:
        raise ValueError(
            f"Need at least {required_history} daily observations to build forecast features."
        )

    holiday_value = (
        int(holiday)
        if holiday is not None
        else int(requested_forecast_date.dayofweek >= 5)
    )

    features: dict[str, float | int] = {
        "store": _encode_label(store, metadata["stores"], "store"),
        "promotion": int(promotion),
        "holiday": holiday_value,
    }

    for lag in DEFAULT_LAGS:
        lag_date = requested_forecast_date - pd.Timedelta(days=lag)
        features[f"sales_lag_{lag}"] = float(history.loc[lag_date])

    for window in DEFAULT_WINDOWS:
        start_date = requested_forecast_date - pd.Timedelta(days=window)
        end_date = requested_forecast_date - pd.Timedelta(days=1)
        window_values = history.loc[start_date:end_date]
        features[f"sales_roll_mean_{window}"] = float(window_values.mean())
        features[f"sales_roll_std_{window}"] = float(window_values.std(ddof=1))

    features["day_of_week"] = int(requested_forecast_date.dayofweek)
    features["month"] = int(requested_forecast_date.month)
    features["week_of_year"] = int(requested_forecast_date.isocalendar().week)
    features["is_weekend"] = int(requested_forecast_date.dayofweek >= 5)

    return requested_forecast_date, features
