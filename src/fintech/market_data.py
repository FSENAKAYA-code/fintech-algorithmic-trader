"""Explicit optional Yahoo Finance adapter; never called by training defaults."""

from datetime import date

import pandas as pd


def download_adjusted_prices(ticker: str, start: date, end: date) -> pd.Series:
    if not ticker.strip() or start >= end:
        raise ValueError("provide a ticker and an increasing date range")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("install yfinance in the project uv environment") from exc
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if frame.empty or "Close" not in frame:
        raise ValueError("the requested ticker/date range returned no adjusted prices")
    close = frame["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)
