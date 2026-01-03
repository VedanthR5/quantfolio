"""
Data Selection Page - AutoML Stock Analysis
============================================
Select stock ticker and date range for analysis.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import sys
sys.path.append('..')

from utils.styles import apply_custom_css, add_footer
from utils.data_fetcher import (
    fetch_stock_data, 
    get_available_tickers, 
    get_default_date_range,
    get_stock_info
)

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("Data Selection")
st.markdown("Choose a stock and date range to begin your analysis.")

st.markdown("---")

# =============================================================================
# STOCK SELECTION
# =============================================================================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Stock")
    
    # Category selection
    tickers_by_category = get_available_tickers()
    category = st.selectbox(
        "Category",
        options=list(tickers_by_category.keys()),
        index=0
    )
    
    # Ticker selection
    ticker = st.selectbox(
        "Ticker Symbol",
        options=tickers_by_category[category],
        index=0
    )
    
    # Or enter custom ticker
    custom_ticker = st.text_input(
        "Or enter custom ticker",
        placeholder="e.g., AMD, NFLX",
        help="Enter any valid Yahoo Finance ticker symbol"
    )
    
    if custom_ticker:
        ticker = custom_ticker.upper()

with col2:
    st.subheader("Date Range")
    
    # Preset date ranges
    preset = st.radio(
        "Quick Select",
        options=["1 Year", "3 Years", "5 Years", "10 Years", "Custom"],
        horizontal=True,
        index=2
    )
    
    # Calculate dates based on preset
    end_date = datetime.now()
    if preset == "1 Year":
        start_date = end_date - timedelta(days=365)
    elif preset == "3 Years":
        start_date = end_date - timedelta(days=3*365)
    elif preset == "5 Years":
        start_date = end_date - timedelta(days=5*365)
    elif preset == "10 Years":
        start_date = end_date - timedelta(days=10*365)
    else:
        start_date = end_date - timedelta(days=5*365)
    
    # Date inputs
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start Date",
            value=start_date,
            max_value=end_date
        )
    with col_end:
        end_date_val = st.date_input(
            "End Date",
            value=end_date
        )
        end_date = end_date_val if not isinstance(end_date_val, tuple) else end_date

st.markdown("---")

# =============================================================================
# FETCH DATA
# =============================================================================

if st.button("Fetch Stock Data", type="primary", width='stretch'):
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        if df is not None and not df.empty:
            # Store in session state
            st.session_state['automl_data'] = df
            st.session_state['automl_ticker'] = ticker
            st.session_state['automl_start_date'] = str(start_date)
            st.session_state['automl_end_date'] = str(end_date)
            
            st.success(f"Loaded {len(df)} trading days of data for {ticker}")
            
            # Display stock info
            info = get_stock_info(ticker)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", info['name'][:20] + "..." if len(info['name']) > 20 else info['name'])
            with col2:
                st.metric("Sector", info['sector'])
            with col3:
                st.metric("Trading Days", f"{len(df):,}")
            with col4:
                if info['market_cap'] > 0:
                    market_cap_b = info['market_cap'] / 1e9
                    st.metric("Market Cap", f"${market_cap_b:.1f}B")
                else:
                    st.metric("Market Cap", "N/A")
            
            # Preview data
            st.subheader("Data Preview")
            
            tab1, tab2 = st.tabs(["Latest Data", "Statistics"])
            
            with tab1:
                st.dataframe(df.tail(10), width='stretch')
            
            with tab2:
                st.dataframe(df.describe(), width='stretch')
            
            # Quick chart
            st.subheader("Price Overview")
            st.line_chart(df['Close'], width='stretch')
            
        else:
            st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")

# =============================================================================
# DISPLAY CURRENT DATA (if exists)
# =============================================================================

if 'automl_data' in st.session_state and st.session_state['automl_data'] is not None:
    st.markdown("---")
    st.info(f"Current dataset: **{st.session_state.get('automl_ticker', 'Unknown')}** | "
            f"{len(st.session_state['automl_data'])} trading days | "
            f"{st.session_state.get('automl_start_date', '')} to {st.session_state.get('automl_end_date', '')}")
    
    st.markdown("**Next step:** Head to Visualization to explore the data.")

add_footer()
