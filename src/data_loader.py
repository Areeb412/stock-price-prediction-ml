"""Download and validate OHLCV stock data."""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_and_validate(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance and validate it.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. 'AAPL', 'TSLA')
    start : str
        Start date 'YYYY-MM-DD'
    end : str
        End date 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        Clean OHLCV DataFrame indexed by date

    Raises
    ------
    ValueError
        If ticker is invalid or no data is returned
    """
    logger.info(f"Fetching {ticker} from {start} to {end}...")

    # Adjust prices so splits and dividends do not create artificial jumps.
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Please verify the symbol is correct (e.g. AAPL, TSLA, MSFT)."
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    # Forward-fill occasional gaps before dropping any remaining incomplete rows.
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Fetched {len(df):,} trading days for {ticker}.")
    return df
