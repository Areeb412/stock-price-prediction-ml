# 📈 Stock Price Prediction — ML-Powered Short-Term Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Notebook-.ipynb-orange?logo=jupyter)
![uv](https://img.shields.io/badge/env-uv-purple)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade machine learning system for short-term stock price prediction using **Linear Regression** and **Random Forest** models trained on real market data fetched live from Yahoo Finance.

---

## 🎯 Project Objective

Predict the **next day's closing price** of a given stock using historical OHLCV (Open, High, Low, Close, Volume) data combined with a rich set of engineered time series features.

---

## 🏗️ Architecture

```
Yahoo Finance (yfinance API)
      │
      ▼
Raw OHLCV Data → EDA → Feature Engineering → Preprocessing
                                                   │
                               ┌───────────────────┤
                               ▼                   ▼
                      Linear Regression     Random Forest
                               │                   │
                               └─────────┬─────────┘
                                         ▼
                              Evaluation + Visualization
                                         │
                                         ▼
                               Next-Day Price Prediction
```

---

## ✨ Key Features

- **Live Data Fetching**: Pulls real, adjusted stock data via `yfinance`
- **25+ Engineered Features**: Lag features, rolling statistics, momentum indicators, volume ratios
- **Two ML Models**: Linear Regression (baseline) + Random Forest Regressor (production)
- **Comprehensive Evaluation**: MAE, RMSE, R², MAPE, Directional Accuracy
- **5 Publication-Quality Plots**: EDA dashboard, train/test split, actual vs predicted, model comparison dashboard, feature importance
- **Next-Day Prediction**: Generates an actual forecast for tomorrow's price
- **Production Practices**: Logging, modular functions, configuration dict, reproducible seeds, data validation

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install all dependencies
uv sync

# 4. Open the notebook in VSCode
code .
# Then open stock_prediction.ipynb and select the .venv kernel
```

### Changing the Stock

In `stock_prediction.ipynb`, Cell 1, change the `CONFIG["TICKER"]`:

```python
CONFIG = {
    "TICKER": "TSLA",  # Change to any valid ticker: MSFT, GOOGL, AMZN, etc.
    ...
}
```

Then run all cells with `Ctrl+Shift+P` → "Run All Cells".

---

## 📊 Results (Apple — AAPL)

| Model             | MAE    | RMSE   | R²     | Directional Acc |
|-------------------|--------|--------|--------|-----------------|
| Linear Regression | ~$2.40 | ~$3.10 | ~0.980 | ~52%            |
| Random Forest     | ~$1.80 | ~$2.50 | ~0.988 | ~54%            |

> Results vary based on the date range and market conditions at time of running.

---

## 🔬 Feature Engineering

| Feature Group       | Features Created                                              |
|---------------------|---------------------------------------------------------------|
| Lag Features        | Close and Volume for lags 1, 2, 3, 5, 10 days               |
| Rolling Statistics  | 5, 10, 20-day moving averages, std deviations, price ratios  |
| Intraday Structure  | High-Low range, Close % of range, Open-Close spread          |
| Momentum            | Daily return, 5-day ROC, 10-day ROC, open gap               |
| Volume Context      | Volume ratio vs 20-day average, daily volume change          |

---

## 📁 Project Structure

```
stock-price-predictor/
├── stock_prediction.ipynb  # Main notebook (12 cells)
├── pyproject.toml          # uv project configuration
├── uv.lock                 # Locked dependency versions
├── data/                   # Auto-generated cached CSV
├── outputs/                # Auto-generated plots and charts
└── README.md               # This file
```

---

## 🛠️ Tech Stack

| Tool           | Version | Purpose                          |
|----------------|---------|----------------------------------|
| Python         | 3.11    | Core language                    |
| uv             | latest  | Package & environment management |
| yfinance       | 0.2+    | Yahoo Finance data API           |
| pandas         | 2.0+    | Data manipulation                |
| numpy          | 1.24+   | Numerical computing              |
| scikit-learn   | 1.3+    | ML models and preprocessing      |
| matplotlib     | 3.7+    | Plotting                         |
| seaborn        | 0.12+   | Statistical visualization        |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price prediction is inherently uncertain and influenced by countless unpredictable factors. **Do not use the outputs of this model for real investment or trading decisions.**

---
