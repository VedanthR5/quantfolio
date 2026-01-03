"""
Visualization Page - AutoML Stock Analysis
===========================================
Interactive visualizations for stock data exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')

from utils.styles import apply_custom_css, add_footer

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("Data Visualization")
st.markdown("Explore stock price patterns with interactive charts.")

st.markdown("---")

# =============================================================================
# CHECK FOR DATA
# =============================================================================

if 'automl_data' not in st.session_state or st.session_state['automl_data'] is None:
    st.warning("No data loaded. Please go to Data Selection first.")
    st.page_link("pages/1_Data_Selection.py", label="Go to Data Selection")
    st.stop()

df = st.session_state['automl_data']
ticker = st.session_state.get('automl_ticker', 'Stock')

st.info(f"Analyzing: **{ticker}** | {len(df)} trading days")

# =============================================================================
# VISUALIZATION OPTIONS
# =============================================================================

viz_type = st.selectbox(
    "Select Visualization",
    options=[
        "Price Chart with Moving Averages",
        "Candlestick Chart",
        "Volume Analysis",
        "Returns Distribution",
        "Lag Plot Analysis",
        "Rolling Statistics"
    ]
)

st.markdown("---")

# =============================================================================
# PRICE CHART WITH MOVING AVERAGES
# =============================================================================

if viz_type == "Price Chart with Moving Averages":
    st.subheader("Price Chart with Moving Averages")
    
    col1, col2 = st.columns(2)
    with col1:
        fast_ma = st.slider("Fast MA Period", 5, 50, 10)
    with col2:
        slow_ma = st.slider("Slow MA Period", 20, 200, 50)
    
    # Calculate moving averages
    df_viz = df.copy()
    df_viz['Fast_MA'] = df_viz['Close'].rolling(window=fast_ma).mean()
    df_viz['Slow_MA'] = df_viz['Close'].rolling(window=slow_ma).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_viz.index, y=df_viz['Close'],
        mode='lines', name='Close Price',
        line=dict(color='#1E88E5', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_viz.index, y=df_viz['Fast_MA'],
        mode='lines', name=f'{fast_ma}-Day MA',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_viz.index, y=df_viz['Slow_MA'],
        mode='lines', name=f'{slow_ma}-Day MA',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker} Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate signals
    df_viz['Signal'] = np.where(df_viz['Fast_MA'] > df_viz['Slow_MA'], 1, -1)
    df_viz['Position'] = df_viz['Signal'].diff()
    
    buy_signals = df_viz[df_viz['Position'] == 2]
    sell_signals = df_viz[df_viz['Position'] == -2]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Buy Signals", len(buy_signals))
    with col2:
        st.metric("Sell Signals", len(sell_signals))

# =============================================================================
# CANDLESTICK CHART
# =============================================================================

elif viz_type == "Candlestick Chart":
    st.subheader("Candlestick Chart")
    
    # Date range slider
    days = st.slider("Number of recent days to display", 30, min(500, len(df)), 90)
    df_candle = df.tail(days)
    
    fig = go.Figure(data=[go.Candlestick(
        x=df_candle.index,
        open=df_candle['Open'],
        high=df_candle['High'],
        low=df_candle['Low'],
        close=df_candle['Close'],
        name=ticker
    )])
    
    fig.update_layout(
        title=f'{ticker} Candlestick Chart (Last {days} Days)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

elif viz_type == "Volume Analysis":
    st.subheader("Volume Analysis")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#1E88E5')),
        row=1, col=1
    )
    
    colors = ['#4ECDC4' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#FF6B6B' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} Price and Volume',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Daily Volume", f"{df['Volume'].mean():,.0f}")
    with col2:
        st.metric("Max Volume", f"{df['Volume'].max():,.0f}")
    with col3:
        st.metric("Min Volume", f"{df['Volume'].min():,.0f}")

# =============================================================================
# RETURNS DISTRIBUTION
# =============================================================================

elif viz_type == "Returns Distribution":
    st.subheader("Returns Distribution Analysis")
    
    returns = df['Close'].pct_change().dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker_color='#1E88E5',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'{ticker} Daily Returns Distribution',
        xaxis_title='Daily Return',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Return statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Return", f"{returns.mean()*100:.3f}%")
    with col2:
        st.metric("Std Dev", f"{returns.std()*100:.3f}%")
    with col3:
        st.metric("Skewness", f"{returns.skew():.3f}")
    with col4:
        st.metric("Kurtosis", f"{returns.kurtosis():.3f}")

# =============================================================================
# LAG PLOT ANALYSIS
# =============================================================================

elif viz_type == "Lag Plot Analysis":
    st.subheader("Lag Plot Analysis")
    
    lag = st.slider("Lag Period (days)", 1, 30, 1)
    
    df_lag = df.copy()
    df_lag['Lagged_Close'] = df_lag['Close'].shift(lag)
    df_lag = df_lag.dropna()
    
    # Calculate correlation
    correlation = df_lag['Close'].corr(df_lag['Lagged_Close'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_lag['Close'],
        y=df_lag['Lagged_Close'],
        mode='markers',
        marker=dict(color='#1E88E5', size=5, opacity=0.5),
        name='Data Points'
    ))
    
    fig.update_layout(
        title=f'{ticker} Lag Plot ({lag}-Day Lag) | Correlation: {correlation:.4f}',
        xaxis_title="Today's Close",
        yaxis_title=f"{lag} Day(s) Ago Close",
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"**Correlation Coefficient: {correlation:.4f}** - "
            f"{'Strong' if abs(correlation) > 0.8 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'} "
            f"{'positive' if correlation > 0 else 'negative'} correlation")

# =============================================================================
# ROLLING STATISTICS
# =============================================================================

elif viz_type == "Rolling Statistics":
    st.subheader("Rolling Statistics")
    
    window = st.slider("Rolling Window (days)", 5, 60, 14)
    
    df_roll = df.copy()
    df_roll['Rolling_Mean'] = df_roll['Close'].rolling(window=window).mean()
    df_roll['Rolling_Std'] = df_roll['Close'].rolling(window=window).std()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price and Rolling Mean', 'Rolling Standard Deviation')
    )
    
    fig.add_trace(
        go.Scatter(x=df_roll.index, y=df_roll['Close'], name='Close', line=dict(color='#1E88E5')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_roll.index, y=df_roll['Rolling_Mean'], name=f'{window}-Day Mean', line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_roll.index, y=df_roll['Rolling_Std'], name=f'{window}-Day Std', line=dict(color='#4ECDC4')),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} Rolling Statistics ({window}-Day Window)',
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# NAVIGATION
# =============================================================================

st.markdown("---")
st.markdown("**Next step:** Head to AutoML Training to train prediction models.")

add_footer()
