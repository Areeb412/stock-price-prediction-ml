"""Train and evaluate the stock prediction models."""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def train_and_evaluate(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    train_ratio: float = 0.80,
    rf_n_estimators: int = 200,
    rf_max_depth: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Split, scale, train, and evaluate the models.

    Returns a single dictionary containing everything:
    - Trained model objects
    - Fitted scaler
    - Train/test DataFrames with predictions attached
    - Evaluation metrics for both models
    - Feature importances

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of engineer_features()
    feature_cols : list[str]
        Names of feature columns (not including Target_Close)
    train_ratio : float
        Fraction of data for training (default 0.80)
    rf_n_estimators : int
        Number of trees in Random Forest
    rf_max_depth : int
        Maximum depth of each tree
    random_state : int
        Seed for reproducibility

    Returns
    -------
    dict
        Results dictionary with models, metrics, predictions, and feature importances.
    """
    np.random.seed(random_state)

    # Time series split: train on older rows and test on newer rows.
    split_idx = int(len(df_features) * train_ratio)
    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["Target_Close"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Target_Close"].values

    logger.info(f"Train: {len(train_df):,} days | Test: {len(test_df):,} days")

    # Fit the scaler on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training Linear Regression...")
    lr = LinearRegression(fit_intercept=True, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)

    logger.info(f"Training Random Forest ({rf_n_estimators} trees)...")
    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train_scaled, y_train)

    lr_preds = lr.predict(X_test_scaled)
    rf_preds = rf.predict(X_test_scaled)
    ens_preds = (lr_preds + rf_preds) / 2

    def metrics(y_true, y_pred):
        """Compute MAE, RMSE, R2, MAPE, and directional accuracy."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        dir_acc = float(
            np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        )
        return {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "mape": round(mape, 4),
            "directional_accuracy": round(dir_acc, 2),
        }

    importance_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    test_df = test_df.copy()
    test_df["LR_Predicted"] = lr_preds
    test_df["RF_Predicted"] = rf_preds
    test_df["Ens_Predicted"] = ens_preds

    return {
        "models": {"lr": lr, "rf": rf},
        "scaler": scaler,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "split_date": str(test_df.index[0].date()),
        "metrics": {
            "linear_regression": metrics(y_test, lr_preds),
            "random_forest": metrics(y_test, rf_preds),
            "ensemble": metrics(y_test, ens_preds),
        },
        "feature_importance": importance_df.to_dict(orient="records"),
        "oob_score": round(float(rf.oob_score_), 4),
    }
