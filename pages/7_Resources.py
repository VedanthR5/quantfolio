"""
Resources Page
==============
Learning resources and documentation links.
"""

import streamlit as st
import sys
sys.path.append('..')

from utils.styles import apply_custom_css, add_footer

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("Learning Resources")
st.markdown("Tutorials, documentation, and next steps for quantitative finance.")

st.markdown("---")

# =============================================================================
# JUPYTER NOTEBOOKS
# =============================================================================

st.subheader("Interactive Notebooks")

st.markdown("""
Explore the methodology behind this application with our Jupyter notebooks:
""")


st.markdown("""
#### ARIMA Fundamentals
A step-by-step walkthrough of ARIMA time series forecasting:
- Stationarity testing (ADF test)
- Differencing transformation
- ACF/PACF analysis
- Walk-forward validation
- Error metrics (MSE, SMAPE)

`tutorials/arima_fundamentals.ipynb`
""")


st.markdown("---")

# =============================================================================
# DOCUMENTATION LINKS
# =============================================================================

st.subheader("Documentation")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Data and Analysis
    - [yfinance](https://pypi.org/project/yfinance/) — Yahoo Finance API
    - [pandas](https://pandas.pydata.org/docs/) — Data manipulation
    - [NumPy](https://numpy.org/doc/) — Numerical computing
    - [Plotly](https://plotly.com/python/) — Interactive charts
    """)

with col2:
    st.markdown("""
    #### Machine Learning
    - [PyCaret](https://pycaret.gitbook.io/) — AutoML library
    - [scikit-learn](https://scikit-learn.org/) — ML algorithms
    - [statsmodels](https://www.statsmodels.org/) — Statistical models
    - [NeuralProphet](https://neuralprophet.com/) — Time series DL
    """)

with col3:
    st.markdown("""
    #### Portfolio Analysis
    - [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) — Portfolio optimization
    - [QuantStats](https://github.com/ranaroussi/quantstats) — Performance analytics
    - [Backtesting.py](https://kernc.github.io/backtesting.py/) — Strategy testing
    """)

st.markdown("---")

# =============================================================================
# CONCEPTS EXPLAINED
# =============================================================================

st.subheader("Key Concepts")

with st.expander("ARIMA Model Explained"):
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** is a widely used time series forecasting model.
    
    **Components:**
    - **AR (AutoRegressive)** — Uses past values to predict future values
    - **I (Integrated)** — Differencing to make data stationary
    - **MA (Moving Average)** — Uses past forecast errors
    
    **Parameters (p, d, q):**
    - `p` — Number of lag observations (AR terms)
    - `d` — Degree of differencing
    - `q` — Size of moving average window
    
    **When to use:**
    - Time series data with trends
    - Data that can be made stationary through differencing
    - Short to medium-term forecasting
    """)

with st.expander("Modern Portfolio Theory (MPT)"):
    st.markdown("""
    **Modern Portfolio Theory** (Markowitz, 1952) provides a framework for constructing 
    portfolios that maximize expected return for a given level of risk.
    
    **Key Concepts:**
    - **Efficient Frontier** — Set of optimal portfolios offering highest return for given risk
    - **Diversification** — Combining uncorrelated assets reduces overall risk
    - **Sharpe Ratio** — Risk-adjusted return measure (Return / Volatility)
    
    **Optimization Objectives:**
    - Maximum Sharpe Ratio (best risk-adjusted return)
    - Minimum Volatility (lowest risk)
    - Risk Parity (equal risk contribution)
    
    **Limitations:**
    - Assumes normal distribution of returns
    - Based on historical data (past performance doesn't guarantee future results)
    - Sensitive to input estimates
    """)

with st.expander("Stationarity and the ADF Test"):
    st.markdown("""
    **Stationarity** means statistical properties (mean, variance) don't change over time.
    
    **Why it matters:**
    - Most time series models assume stationarity
    - Non-stationary data can lead to spurious correlations
    - Stock prices are typically non-stationary (they trend)
    
    **Augmented Dickey-Fuller (ADF) Test:**
    - Tests the null hypothesis that data is non-stationary
    - p-value < 0.05 means you can reject the null — data IS stationary
    - p-value > 0.05 means you fail to reject — data is NOT stationary
    
    **Making data stationary:**
    - **Differencing** — Subtract previous value (removes trend)
    - **Log transform** — Stabilizes variance
    - **Detrending** — Remove trend component
    """)

with st.expander("Performance Metrics"):
    st.markdown("""
    **Prediction Metrics:**
    - **MSE (Mean Squared Error)** — Average of squared prediction errors
    - **RMSE** — Square root of MSE (in original units)
    - **MAPE** — Mean Absolute Percentage Error
    - **SMAPE** — Symmetric MAPE (handles zero values better)
    
    **Portfolio Metrics:**
    - **Sharpe Ratio** — (Return - Risk-free rate) / Volatility
    - **Sortino Ratio** — Like Sharpe but only penalizes downside volatility
    - **Max Drawdown** — Largest peak-to-trough decline
    - **Calmar Ratio** — Annual return / Max drawdown
    - **Beta** — Sensitivity to market movements
    - **Alpha** — Excess return over benchmark
    """)

st.markdown("---")

# =============================================================================
# NEXT STEPS
# =============================================================================

st.subheader("Next Steps")

st.markdown("""
Ready to dive deeper? Here are some ideas:

1. **Explore the Notebooks** — Clone the repository and run the Jupyter notebooks locally
2. **Customize Models** — Modify ARIMA parameters or add new features
3. **Add More Assets** — Extend the portfolio with international stocks or crypto
4. **Implement Strategies** — Use backtesting to test trading strategies
5. **Deploy Your Own** — Fork the repo and deploy your customized version

**Links:**
- [GitHub Repository](https://github.com/vedanthr5/vr-quantfolio-intro)
- [Live Documentation](https://vedanthr5.github.io/vr-quantfolio-intro)
""")

st.markdown("---")

# =============================================================================
# ABOUT
# =============================================================================

st.subheader("About This Project")

st.markdown("""
**VR Quantfolio Intro** is an open-source educational platform for learning 
quantitative finance with Python.

**Features:**
- AutoML stock prediction with PyCaret
- Time series forecasting (ARIMA, NeuralProphet)
- Portfolio optimization (Mean-Variance, Riskfolio)
- Interactive visualizations
- Educational Jupyter notebooks

**Tech Stack:**
- Streamlit (Web Framework)
- PyCaret (AutoML)
- Riskfolio-Lib (Portfolio Optimization)
- NeuralProphet (Time Series)
- Plotly (Visualization)

**License:** MIT
""")

add_footer()
