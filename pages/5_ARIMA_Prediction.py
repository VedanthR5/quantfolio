"""
ARIMA Prediction Page - Portfolio Analysis
===========================================
Time series forecasting with ARIMA and NeuralProphet.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from utils.styles import apply_custom_css, add_footer
from utils.data_fetcher import fetch_stock_data, get_available_tickers

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("ARIMA Price Prediction")
st.markdown("Forecast stock prices using time series models (ARIMA and NeuralProphet).")

st.markdown("---")

# =============================================================================
# DATA SELECTION
# =============================================================================

st.subheader("Step 1: Select Stock Data")

col1, col2, col3 = st.columns(3)

with col1:
    tickers = get_available_tickers()
    all_tickers = []
    for t in tickers.values():
        all_tickers.extend(t)
    
    ticker = st.selectbox("Stock Ticker", options=sorted(set(all_tickers)), index=0)

with col2:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=5*365)
    )

with col3:
    end_date = st.date_input(
        "End Date",
        value=datetime.now()
    )

# Fetch data
if st.button("Load Data", type="secondary"):
    with st.spinner(f"Loading {ticker} data..."):
        df = fetch_stock_data(ticker, str(start_date), str(end_date))
        if df is not None:
            st.session_state['arima_data'] = df
            st.session_state['arima_ticker'] = ticker
            st.success(f"Loaded {len(df)} trading days")
        else:
            st.error("Failed to load data")

# Use existing data if available
if 'arima_data' not in st.session_state:
    # Try to use automl data if available
    if 'automl_data' in st.session_state:
        st.session_state['arima_data'] = st.session_state['automl_data']
        st.session_state['arima_ticker'] = st.session_state.get('automl_ticker', 'AAPL')
        st.info(f"Using data from AutoML: {st.session_state['arima_ticker']}")
    else:
        st.info("Click 'Load Data' to fetch stock data.")
        st.stop()

df = st.session_state['arima_data']
ticker = st.session_state.get('arima_ticker', 'Stock')

st.markdown("---")

# =============================================================================
# STATIONARITY ANALYSIS
# =============================================================================

st.subheader("Step 2: Stationarity Analysis")

with st.expander("Understanding Stationarity", expanded=False):
    st.markdown("""
    **Why is stationarity important?**
    
    ARIMA models assume the data is *stationary* — meaning the statistical properties 
    (mean, variance) don't change over time. Stock prices are typically non-stationary 
    because they trend upward over time.
    
    **Solution:** Apply *differencing* to transform prices into price changes, 
    which are typically stationary.
    """)

# Train/Test Split
train_size = st.slider("Training Set Size (%)", 60, 90, 80)
split_idx = int(len(df) * train_size / 100)

train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Samples", len(train_data))
with col2:
    st.metric("Test Samples", len(test_data))

# Plot train/test split
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], name='Training', line=dict(color='#1E88E5')))
fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], name='Test', line=dict(color='#4ECDC4')))
fig.update_layout(title=f'{ticker} Train/Test Split', xaxis_title='Date', yaxis_title='Price ($)', template='plotly_white')
st.plotly_chart(fig, width='stretch')

# ADF Test
if st.button("Run Stationarity Test (ADF)"):
    from statsmodels.tsa.stattools import adfuller
    
    train_series = train_data['Close']
    
    # Test on original data
    adf_result = adfuller(train_series, autolag='AIC')
    
    st.markdown("**Augmented Dickey-Fuller Test Results (Original Data):**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Statistic", f"{adf_result[0]:.4f}")
    with col2:
        st.metric("P-Value", f"{adf_result[1]:.6f}")
    with col3:
        is_stationary = adf_result[1] < 0.05
        st.metric("Stationary?", "Yes" if is_stationary else "No")
    
    if not is_stationary:
        st.warning("Data is non-stationary. Applying differencing...")
        
        # Test on differenced data
        train_diff = train_series.diff().dropna()
        adf_diff = adfuller(train_diff, autolag='AIC')
        
        st.markdown("**After Differencing:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Statistic", f"{adf_diff[0]:.4f}")
        with col2:
            st.metric("P-Value", f"{adf_diff[1]:.10f}")
        with col3:
            st.metric("Stationary?", "Yes" if adf_diff[1] < 0.05 else "No")

st.markdown("---")

# =============================================================================
# MODEL TRAINING
# =============================================================================

st.subheader("Step 3: Train Forecasting Model")

model_type = st.radio(
    "Select Model",
    options=["ARIMA (Classical)", "NeuralProphet (Deep Learning)"],
    horizontal=True
)

if model_type == "ARIMA (Classical)":
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("p (AR order)", 0, 10, 5)
    with col2:
        d = st.number_input("d (Differencing)", 0, 3, 1)
    with col3:
        q = st.number_input("q (MA order)", 0, 10, 0)
    
    if st.button("Train ARIMA Model", type="primary"):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from sklearn.metrics import mean_squared_error
            
            with st.spinner("Training ARIMA model... This may take a few minutes."):
                train_series = train_data['Close']
                test_series = test_data['Close']
                
                # Differencing
                train_diff = train_series.diff().dropna()
                test_diff = test_series.diff().dropna()
                
                # Walk-forward validation
                history = list(train_diff)
                predictions = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for t in range(len(test_diff)):
                    model = ARIMA(history, order=(int(p), int(d), int(q)))
                    model_fit = model.fit()
                    
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    
                    obs = test_diff.iloc[t]
                    history.append(obs)
                    
                    # Update progress
                    progress = (t + 1) / len(test_diff)
                    progress_bar.progress(progress)
                    if t % 50 == 0:
                        status_text.text(f"Processing: {t}/{len(test_diff)} ({progress*100:.1f}%)")
                
                progress_bar.empty()
                status_text.empty()
                
                # Reverse differencing
                reverse_test = np.r_[test_series.iloc[0], test_diff].cumsum()
                reverse_pred = np.r_[test_series.iloc[0], predictions].cumsum()
                
                # Calculate errors
                mse = mean_squared_error(reverse_test, reverse_pred)
                
                # SMAPE
                denominator = (np.abs(reverse_pred) + np.abs(reverse_test)) + 1e-8
                smape = np.mean(np.abs(reverse_pred - reverse_test) * 200 / denominator)
                
                st.success("ARIMA model trained!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.2f}")
                with col2:
                    st.metric("SMAPE", f"{smape:.2f}%")
                
                # Plot results
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_series.index, y=train_series, name='Training', line=dict(color='#1E88E5')))
                fig.add_trace(go.Scatter(x=test_series.index, y=reverse_test, name='Actual', line=dict(color='#4ECDC4')))
                fig.add_trace(go.Scatter(x=test_series.index, y=reverse_pred, name='Predicted', line=dict(color='#FF6B6B', dash='dash')))
                
                fig.update_layout(
                    title=f'{ticker} ARIMA({int(p)},{int(d)},{int(q)}) Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Store results
                st.session_state['arima_predictions'] = reverse_pred
                st.session_state['arima_actual'] = reverse_test
        
        except Exception as e:
            st.error(f"Error training ARIMA model: {str(e)}")

else:  # NeuralProphet
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    with col2:
        ar_order = st.slider("Autoregression Order", 1, 30, 7)
    
    if st.button("Train NeuralProphet Model", type="primary"):
        try:
            from neuralprophet import NeuralProphet
            
            with st.spinner("Training NeuralProphet model... This may take a few minutes."):
                # Prepare data for NeuralProphet
                df_prophet = df[['Close']].reset_index()
                df_prophet.columns = ['ds', 'y']
                
                # Train model
                model = NeuralProphet(
                    n_lags=ar_order,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                )
                
                model.fit(df_prophet, freq='D')
                
                # Make future predictions
                future = model.make_future_dataframe(df_prophet, periods=forecast_days)
                forecast = model.predict(future)
                
                st.success("NeuralProphet model trained!")
                
                # Plot forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical', line=dict(color='#1E88E5')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name='Forecast', line=dict(color='#FF6B6B', dash='dash')))
                
                fig.update_layout(
                    title=f'{ticker} NeuralProphet {forecast_days}-Day Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Show forecast table
                st.subheader("Forecast Values")
                forecast_display = forecast[['ds', 'yhat1']].tail(forecast_days)
                forecast_display.columns = ['Date', 'Predicted Price']
                st.dataframe(forecast_display, width='stretch')
        
        except ImportError:
            st.error("NeuralProphet is not installed. Please install it with: `pip install neuralprophet`")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

# =============================================================================
# NAVIGATION
# =============================================================================

st.markdown("---")
st.markdown("**Next step:** Head to Portfolio Optimization to build optimal portfolios.")

add_footer()
