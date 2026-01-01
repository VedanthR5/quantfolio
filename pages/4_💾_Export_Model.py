"""
Export Model Page - AutoML Stock Analysis
==========================================
Export trained models and make predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')

from utils.styles import apply_custom_css

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("💾 Export Model")
st.markdown("Save your trained model and make predictions on new data.")

st.markdown("---")

# =============================================================================
# CHECK FOR MODEL
# =============================================================================

if 'automl_best_model' not in st.session_state:
    st.warning("⚠️ No trained model found. Please go to **AutoML Training** first.")
    st.page_link("pages/3_🤖_AutoML_Training.py", label="Go to AutoML Training", icon="🤖")
    st.stop()

model = st.session_state['automl_best_model']
ticker = st.session_state.get('automl_ticker', 'Stock')
forecast_horizon = st.session_state.get('automl_forecast_horizon', 1)

st.success(f"✅ Model loaded: **{type(model).__name__}** for {ticker}")

# =============================================================================
# MODEL SUMMARY
# =============================================================================

st.subheader("📋 Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", type(model).__name__)
with col2:
    st.metric("Target Stock", ticker)
with col3:
    st.metric("Forecast Horizon", f"{forecast_horizon} day(s)")

# Display training results if available
if 'automl_results' in st.session_state:
    with st.expander("View Training Results", expanded=False):
        st.dataframe(st.session_state['automl_results'], use_container_width=True)

st.markdown("---")

# =============================================================================
# EXPORT MODEL
# =============================================================================

st.subheader("💾 Save Model")

col1, col2 = st.columns([2, 1])

with col1:
    model_name = st.text_input(
        "Model Name",
        value=f"{ticker.lower()}_model_{datetime.now().strftime('%Y%m%d')}",
        help="Name for the saved model file (without extension)"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    export_button = st.button("📥 Export Model", type="primary", use_container_width=True)

if export_button:
    try:
        from pycaret.regression import save_model
        
        with st.spinner("Saving model..."):
            save_model(model, model_name)
            st.success(f"✅ Model saved as `{model_name}.pkl`")
            
            # Provide download info
            st.info(f"""
            **Model saved successfully!**
            
            To load this model later:
            ```python
            from pycaret.regression import load_model
            model = load_model('{model_name}')
            ```
            """)
    
    except ImportError:
        st.error("❌ PyCaret is not installed. Please install it with: `pip install pycaret`")
    except Exception as e:
        st.error(f"❌ Error saving model: {str(e)}")

st.markdown("---")

# =============================================================================
# MAKE PREDICTIONS
# =============================================================================

st.subheader("🔮 Make Predictions")

prediction_method = st.radio(
    "Prediction Method",
    options=["Use Current Data", "Upload New Data"],
    horizontal=True
)

if prediction_method == "Use Current Data":
    if 'automl_prepared_data' in st.session_state:
        df_predict = st.session_state['automl_prepared_data'].copy()
        
        # Show last few rows for prediction
        st.markdown("**Latest data for prediction:**")
        st.dataframe(df_predict.tail(5), use_container_width=True)
        
        if st.button("🎯 Generate Predictions", type="secondary"):
            try:
                from pycaret.regression import predict_model
                
                with st.spinner("Generating predictions..."):
                    predictions = predict_model(model, data=df_predict)
                    
                    st.success("✅ Predictions generated!")
                    
                    # Display predictions
                    pred_col = 'prediction_label' if 'prediction_label' in predictions.columns else predictions.columns[-1]
                    
                    st.subheader("Prediction Results")
                    
                    # Show last 10 predictions
                    results_df = predictions[['Close', 'Target', pred_col]].tail(10).copy()
                    results_df.columns = ['Current Close', 'Actual Future', 'Predicted Future']
                    results_df['Error (%)'] = abs(results_df['Actual Future'] - results_df['Predicted Future']) / results_df['Actual Future'] * 100
                    
                    st.dataframe(results_df.round(2), use_container_width=True)
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        mape = results_df['Error (%)'].mean()
                        st.metric("Mean Absolute % Error", f"{mape:.2f}%")
                    with col2:
                        latest_pred = results_df['Predicted Future'].iloc[-1]
                        st.metric("Latest Prediction", f"${latest_pred:.2f}")
            
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
    else:
        st.warning("No prepared data available. Please go back to AutoML Training.")

else:
    uploaded_file = st.file_uploader(
        "Upload CSV file with features",
        type=['csv'],
        help="Upload a CSV file with the same features used in training"
    )
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.markdown("**Uploaded data preview:**")
            st.dataframe(new_data.head(), use_container_width=True)
            
            if st.button("🎯 Generate Predictions", type="secondary"):
                from pycaret.regression import predict_model
                
                with st.spinner("Generating predictions..."):
                    predictions = predict_model(model, data=new_data)
                    st.success("✅ Predictions generated!")
                    st.dataframe(predictions, use_container_width=True)
                    
                    # Download button
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"{ticker}_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# =============================================================================
# NAVIGATION
# =============================================================================

st.markdown("---")
st.markdown("""
**Explore More:**
- [ARIMA Prediction](5_🔮_ARIMA_Prediction) - Time series forecasting
- [Portfolio Optimization](6_💼_Portfolio_Optimization) - Build optimal portfolios
""")
