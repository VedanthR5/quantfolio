"""
Shared utilities package for VR Quantfolio Intro
"""

from .styles import apply_custom_css
from .data_fetcher import fetch_stock_data, get_available_tickers

__all__ = ['apply_custom_css', 'fetch_stock_data', 'get_available_tickers']
