# Stock Price Prediction ML

This project predicts the next trading day's closing price for a stock using historical OHLCV data from Yahoo Finance. It includes a Jupyter notebook for exploration and a Gradio app for running the same workflow from a small web UI.

The model output is for learning and experimentation only. It should not be used for trading or investment decisions.

## What It Does

- Downloads adjusted stock price data with `yfinance`
- Builds lag, rolling average, volatility, momentum, and volume features
- Trains Linear Regression, Random Forest, and a simple ensemble
- Evaluates the models with MAE, RMSE, R2, MAPE, and directional accuracy
- Saves charts for EDA, predictions, feature importance, and model comparison
- Provides an optional Gradio interface for trying different tickers and settings

## Project Structure

```text
stock-price-prediction-ml/
|-- stock_prediction.ipynb      # Notebook workflow
|-- gradio_app.py               # Interactive Gradio app
|-- src/
|   |-- data_loader.py          # Yahoo Finance download and validation
|   |-- features.py             # Feature engineering
|   |-- model_trainer.py        # Model training and evaluation
|   `-- predictor.py            # Next-day prediction helper
|-- data/                       # Cached/downloaded data
|-- outputs/                    # Saved charts
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Setup

This repository uses `uv`.

```bash
uv sync
```

If you want to use the notebook, open `stock_prediction.ipynb` and select the project virtual environment as the kernel.

## Run the Notebook

Open `stock_prediction.ipynb`, change the ticker or date range in the `CONFIG` dictionary if needed, then run the cells from top to bottom.

Example:

```python
CONFIG = {
    "TICKER": "AAPL",
    "START_DATE": "2019-01-01",
    "END_DATE": datetime.today().strftime("%Y-%m-%d"),
    "TRAIN_RATIO": 0.80,
}
```

The notebook saves generated figures under `outputs/`.

## Run the Gradio App

```bash
uv run python gradio_app.py
```

Then open:

```text
http://localhost:7860
```

From the app, you can change the ticker, date range, train/test split, and Random Forest settings.

## Model Notes

The train/test split is chronological because stock data is time series data. Older rows are used for training and newer rows are held out for testing.

The Random Forest is usually the stronger model here because it can pick up non-linear relationships in the engineered features. Linear Regression is kept as a baseline, and the ensemble is a simple average of both model predictions.

## Outputs

The project can generate:

- Historical price and volume plots
- Actual vs predicted closing price plots
- Model comparison dashboard
- Random Forest feature importance chart
- Next-trading-day prediction table

## Disclaimer

This is an educational machine learning project. Stock prices are noisy and affected by events that are not present in the historical OHLCV data. Do not use this project as financial advice.
