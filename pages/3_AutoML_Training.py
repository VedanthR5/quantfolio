"""
AutoML Training Page - AutoML Stock Analysis
=============================================
Automated machine learning model training with PyCaret.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from utils.styles import apply_custom_css, add_footer

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("AutoML Model Training")
st.markdown("Use PyCaret to automatically find the best model for stock prediction.")

st.markdown("---")

# =============================================================================
# CHECK FOR DATA
# =============================================================================

if 'automl_data' not in st.session_state or st.session_state['automl_data'] is None:
    st.warning("No data loaded. Please go to Data Selection first.")
    st.page_link("pages/1_Data_Selection.py", label="Go to Data Selection")
    st.stop()

df = st.session_state['automl_data'].copy()
ticker = st.session_state.get('automl_ticker', 'Stock')

st.info(f"Training on: **{ticker}** | {len(df)} trading days")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

st.subheader("Step 1: Feature Engineering")

with st.expander("Configure Features", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        include_ma = st.checkbox("Include Moving Averages", value=True)
        ma_periods = st.multiselect(
            "MA Periods",
            options=[5, 10, 20, 50, 100],
            default=[10, 20, 50],
            disabled=not include_ma
        )
    
    with col2:
        include_momentum = st.checkbox("Include Momentum Indicators", value=True)
        include_volatility = st.checkbox("Include Volatility Features", value=True)
    
    target_col = st.selectbox(
        "Target Variable",
        options=["Close", "Open", "High", "Low"],
        index=0
    )
    
    forecast_horizon = st.slider(
        "Forecast Horizon (days ahead)",
        min_value=1,
        max_value=30,
        value=1,
        help="How many days ahead to predict"
    )

# =============================================================================
# PREPARE DATA
# =============================================================================

if st.button("Prepare Features", type="secondary"):
    with st.spinner("Engineering features..."):
        df_ml = df.copy()
        
        # Moving averages
        if include_ma:
            for period in ma_periods:
                df_ml[f'MA_{period}'] = df_ml['Close'].rolling(window=period).mean()
        
        # Momentum indicators
        if include_momentum:
            df_ml['Returns'] = df_ml['Close'].pct_change()
            df_ml['Momentum_5'] = df_ml['Close'].pct_change(periods=5)
            df_ml['Momentum_10'] = df_ml['Close'].pct_change(periods=10)
            # Calculate RSI
            delta = df_ml['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_ml['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility features
        if include_volatility:
            df_ml['Volatility_5'] = df_ml['Close'].rolling(window=5).std()
            df_ml['Volatility_20'] = df_ml['Close'].rolling(window=20).std()
            df_ml['Daily_Range'] = (df_ml['High'] - df_ml['Low']) / df_ml['Close']
        
        # Target: Future price
        df_ml['Target'] = df_ml[target_col].shift(-forecast_horizon)
        
        # Drop NaN values
        df_ml = df_ml.dropna()
        
        # Store prepared data
        st.session_state['automl_prepared_data'] = df_ml
        st.session_state['automl_target'] = 'Target'
        st.session_state['automl_forecast_horizon'] = forecast_horizon
        
        st.success(f"Prepared {len(df_ml)} samples with {len(df_ml.columns)} features")
        
        # Show feature preview
        st.dataframe(df_ml.head(), width='stretch')


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =============================================================================
# MODEL TRAINING
# =============================================================================

st.markdown("---")
st.subheader("Step 2: Model Training")

if 'automl_prepared_data' not in st.session_state:
    st.info("Please prepare features first before training.")
else:
    df_ml = st.session_state['automl_prepared_data']
    
    with st.expander("Training Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            train_size = st.slider("Training Set Size (%)", 60, 90, 80)
            fold = st.slider("Cross-Validation Folds", 2, 10, 5)
        
        with col2:
            n_select = st.slider("Top Models to Compare", 3, 15, 5)
            session_id = st.number_input("Random Seed", value=123, step=1)
    
    if st.button("Start AutoML Training", type="primary", width='stretch'):
        try:
            from pycaret.regression import setup, compare_models, pull, plot_model
            
            with st.spinner("Setting up PyCaret environment..."):
                # Prepare features (exclude target from features)
                feature_cols = [col for col in df_ml.columns if col not in ['Target', 'Date']]
                
                # Setup PyCaret
                exp = setup(
                    data=df_ml,
                    target='Target',
                    train_size=train_size/100,
                    fold=fold,
                    session_id=int(session_id),
                    verbose=False,
                    html=False
                )
                
                st.success("PyCaret environment configured")
            
            with st.spinner("Comparing models... This may take a few minutes."):
                # Compare models
                best_models = compare_models(n_select=n_select)
                
                # Get comparison results
                results = pull()
                
                st.success("Model comparison complete!")
                
                # Display results
                st.subheader("Model Comparison Results")
                st.dataframe(results, width='stretch')
                
                # Store best model
                if isinstance(best_models, list):
                    best_model = best_models[0]
                else:
                    best_model = best_models
                
                st.session_state['automl_best_model'] = best_model
                st.session_state['automl_results'] = results
                
                st.success(f"Best Model: **{type(best_model).__name__}**")
            
            # Model plots
            st.subheader("Model Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Residuals", "Prediction Error", "Feature Importance"])
            
            with tab1:
                try:
                    plot_model(best_model, plot='residuals', save=True)
                    st.image('Residuals.png')
                except Exception as e:
                    st.info("Residual plot not available for this model type.")
            
            with tab2:
                try:
                    plot_model(best_model, plot='error', save=True)
                    st.image('Prediction Error.png')
                except Exception as e:
                    st.info("Prediction error plot not available for this model type.")
            
            with tab3:
                try:
                    plot_model(best_model, plot='feature', save=True)
                    st.image('Feature Importance.png')
                except Exception as e:
                    st.info("Feature importance plot not available for this model type.")
        
        except ImportError:
            st.error("PyCaret is not installed. Please install it with: `pip install pycaret`")
        except Exception as e:
            st.error(f"Error during training: {str(e)}")

# =============================================================================
# NAVIGATION
# =============================================================================

st.markdown("---")

if 'automl_best_model' in st.session_state:
    st.markdown("**Next step:** Head to Export Model to save your trained model.")
else:
    st.markdown("Train a model to proceed to the export step.")

add_footer()
