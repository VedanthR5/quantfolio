"""
Data fetching utilities for VR Quantfolio Intro
================================================
Shared functions for downloading and processing stock data.
"""

import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


# Available tickers for selection
AVAILABLE_TICKERS = {
    "Tech Giants": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"],
    "Finance": ["JPM", "BAC", "GS", "V", "MA", "BRK-B"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Consumer": ["KO", "PEP", "PG", "WMT", "COST", "NKE"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"],
}


def get_available_tickers() -> dict:
    """Return dictionary of available tickers by category."""
    return AVAILABLE_TICKERS


def get_all_tickers() -> List[str]:
    """Return flat list of all available tickers."""
    all_tickers = []
    for tickers in AVAILABLE_TICKERS.values():
        all_tickers.extend(tickers)
    return sorted(list(set(all_tickers)))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    auto_adjust: bool = False
) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with caching.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        auto_adjust: Whether to auto-adjust for splits/dividends
    
    Returns:
        DataFrame with OHLCV data, or None if fetch fails
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            multi_level_index=False,
            progress=False
        )
        
        if df is None or df.empty:
            return None
        
        # Clean the data
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_stocks(
    tickers: List[str],
    start_date: str,
    end_date: str,
    column: str = "Close"
) -> Optional[pd.DataFrame]:
    """
    Fetch data for multiple stocks and return combined DataFrame.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        column: Which column to extract (default: "Close")
    
    Returns:
        DataFrame with tickers as columns, dates as index
    """
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        
        if data is None or data.empty:
            return None
        
        # Handle multi-level columns for multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            result_df = data[column].copy()
        else:
            result_df = data[[column]].copy()
            result_df.columns = tickers
        
        result_df = result_df.dropna()
        return result_df
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def get_default_date_range(years: int = 5) -> Tuple[datetime, datetime]:
    """
    Get default date range for data selection.
    
    Args:
        years: Number of years to look back
    
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    return start_date, end_date


def calculate_returns(prices: pd.DataFrame, period: str = "daily") -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame of prices
        period: One of "daily", "weekly", "monthly"
    
    Returns:
        DataFrame of returns
    """
    if period == "daily":
        return prices.pct_change().dropna()
    elif period == "weekly":
        return prices.resample('W').last().pct_change().dropna()
    elif period == "monthly":
        return prices.resample('M').last().pct_change().dropna()
    else:
        raise ValueError(f"Unknown period: {period}")


def get_stock_info(ticker: str) -> dict:
    """
    Get basic info about a stock.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
        }
    except Exception:
        return {
            "name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": 0,
            "currency": "USD",
        }
