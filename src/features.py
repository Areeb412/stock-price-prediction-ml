"""Feature engineering for OHLCV stock data."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_DAYS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag, rolling, momentum, and volume features.

    Feature Groups
    --------------
    1. Lag features        : Close & Volume N days ago
    2. Rolling statistics  : Moving averages, std devs, price ratios
    3. Intraday structure  : High-Low range, close position, OC spread
    4. Momentum            : Daily return, rate-of-change over 5/10 days
    5. Volume context      : Volume ratio vs 20-day average
    6. Target variable     : Next day's Close (shifted -1)

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Feature-rich DataFrame including 'Target_Close'
    """
    feat = df.copy()

    for lag in LAG_DAYS:
        feat[f"Close_lag{lag}"] = feat["Close"].shift(lag)
        feat[f"Volume_lag{lag}"] = feat["Volume"].shift(lag)

    for window in ROLLING_WINDOWS:
        feat[f"Close_ma{window}"] = feat["Close"].rolling(window).mean()
        feat[f"Close_std{window}"] = feat["Close"].rolling(window).std()
        feat[f"Price_to_ma{window}"] = feat["Close"] / feat[f"Close_ma{window}"]

    feat["HL_range"] = feat["High"] - feat["Low"]
    feat["Close_pct_of_range"] = (
        (feat["Close"] - feat["Low"]) / (feat["HL_range"] + 1e-8)
    )
    feat["OC_spread"] = feat["Close"] - feat["Open"]
    feat["Open_gap"] = feat["Open"] - feat["Close"].shift(1)

    feat["Daily_return"] = feat["Close"].pct_change()
    feat["ROC_5"] = feat["Close"].pct_change(5)
    feat["ROC_10"] = feat["Close"].pct_change(10)

    feat["Volume_ratio_20d"] = feat["Volume"] / feat["Volume"].rolling(20).mean()
    feat["Volume_change"] = feat["Volume"].pct_change()

    # Today's features are paired with the next trading day's close.
    feat["Target_Close"] = feat["Close"].shift(-1)

    feat.dropna(inplace=True)

    logger.info(
        f"Feature engineering complete: {feat.shape[0]:,} rows, {feat.shape[1]} cols"
    )
    return feat


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excludes the target)."""
    return [c for c in df.columns if c != "Target_Close"]
