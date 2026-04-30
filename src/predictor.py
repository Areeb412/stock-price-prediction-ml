"""Create a next-day stock price prediction from trained models."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def predict_next_day(results: dict, df_features: pd.DataFrame) -> dict:
    """
    Generate a next-day closing price prediction.

    Parameters
    ----------
    results : dict
        Output of train_and_evaluate(), including models and scaler.
    df_features : pd.DataFrame
        Full feature-engineered DataFrame

    Returns
    -------
    dict
        Prediction details for all three model variants
    """
    feature_cols = results["feature_cols"]
    scaler = results["scaler"]
    lr_model = results["models"]["lr"]
    rf_model = results["models"]["rf"]

    last_row = df_features[feature_cols].iloc[[-1]]
    last_date = df_features.index[-1]
    current_close = float(df_features["Close"].iloc[-1])

    next_day = last_date + pd.offsets.BDay(1)

    # Use the training scaler so inference matches the fitted feature scale.
    last_scaled = scaler.transform(last_row)

    lr_price = float(lr_model.predict(last_scaled)[0])
    rf_price = float(rf_model.predict(last_scaled)[0])
    ens_price = round((lr_price + rf_price) / 2, 4)

    def build_pred(price: float) -> dict:
        change = round(price - current_close, 4)
        change_pct = round((change / current_close) * 100, 4)
        return {
            "predicted_close": round(price, 4),
            "change_dollars":  change,
            "change_percent":  change_pct,
            "direction":       "UP" if change > 0 else "DOWN",
        }

    return {
        "ticker":          df_features.attrs.get("ticker", "N/A"),
        "prediction_date": str(next_day.date()),
        "based_on_date":   str(last_date.date()),
        "current_close":   round(current_close, 4),
        "predictions": {
            "linear_regression": build_pred(lr_price),
            "random_forest":     build_pred(rf_price),
            "ensemble":          build_pred(ens_price),
        },
    }
