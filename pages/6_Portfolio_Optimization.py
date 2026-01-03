"""
Portfolio Optimization Page
============================
Mean-variance optimization with efficient frontier visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from utils.styles import apply_custom_css, add_footer
from utils.data_fetcher import fetch_multiple_stocks, get_available_tickers, calculate_returns

apply_custom_css()

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("Portfolio Optimization")
st.markdown("Build optimal portfolios using mean-variance optimization (Markowitz).")

st.markdown("---")

# =============================================================================
# PORTFOLIO SELECTION
# =============================================================================

st.subheader("Step 1: Select Portfolio Assets")

tickers_dict = get_available_tickers()

# Flatten tickers for multiselect
all_tickers = []
for category, ticks in tickers_dict.items():
    all_tickers.extend(ticks)
all_tickers = sorted(set(all_tickers))

col1, col2 = st.columns([2, 1])

with col1:
    selected_tickers = st.multiselect(
        "Select Stocks (minimum 2)",
        options=all_tickers,
        default=["AAPL", "GOOGL", "MSFT", "AMZN"],
        help="Select at least 2 stocks for portfolio optimization"
    )
    
    # Custom tickers input
    custom_tickers = st.text_input(
        "Or add custom tickers (comma-separated)",
        placeholder="e.g., AMD, NFLX, TSLA",
        help="Enter any valid Yahoo Finance ticker symbols"
    )
    if custom_tickers:
        custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        # Add custom tickers to selection (avoid duplicates)
        for t in custom_list:
            if t not in selected_tickers:
                selected_tickers.append(t)

with col2:
    preset = st.selectbox(
        "Or use preset",
        options=["Custom", "Tech Giants", "Diversified", "ETF Mix"],
    )
    
    if preset == "Tech Giants":
        selected_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA"]
    elif preset == "Diversified":
        selected_tickers = ["AAPL", "JPM", "JNJ", "XOM", "KO", "SPY"]
    elif preset == "ETF Mix":
        selected_tickers = ["SPY", "QQQ", "IWM", "DIA", "VTI"]

# Date range
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=3*365),
        key="port_start"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        key="port_end"
    )

if len(selected_tickers) < 2:
    st.warning("Please select at least 2 stocks for portfolio optimization.")
    st.stop()

# Fetch data
if st.button("Load Portfolio Data", type="secondary"):
    with st.spinner(f"Loading data for {len(selected_tickers)} stocks..."):
        prices = fetch_multiple_stocks(selected_tickers, str(start_date), str(end_date))
        if prices is not None:
            st.session_state['portfolio_prices'] = prices
            st.session_state['portfolio_tickers'] = selected_tickers
            st.success(f"Loaded {len(prices)} trading days for {len(selected_tickers)} stocks")
        else:
            st.error("Failed to load data")

if 'portfolio_prices' not in st.session_state:
    st.info("Click 'Load Portfolio Data' to fetch stock data.")
    st.stop()

prices = st.session_state['portfolio_prices']
tickers = st.session_state.get('portfolio_tickers', selected_tickers)

# Display price chart
fig = go.Figure()
for ticker in prices.columns:
    # Normalize to 100 for comparison
    normalized = prices[ticker] / prices[ticker].iloc[0] * 100
    fig.add_trace(go.Scatter(x=prices.index, y=normalized, name=ticker, mode='lines'))

fig.update_layout(
    title='Normalized Price Performance (Base = 100)',
    xaxis_title='Date',
    yaxis_title='Normalized Price',
    template='plotly_white',
    height=400
)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# =============================================================================
# PORTFOLIO ANALYSIS
# =============================================================================

st.subheader("Step 2: Portfolio Analysis")

# Calculate returns
returns = calculate_returns(prices, "daily")

# Annualized metrics
ann_returns = returns.mean() * 252
ann_volatility = returns.std() * np.sqrt(252)
sharpe_ratio = ann_returns / ann_volatility

# Display metrics
st.markdown("**Individual Asset Metrics (Annualized)**")

metrics_df = pd.DataFrame({
    'Expected Return': ann_returns,
    'Volatility': ann_volatility,
    'Sharpe Ratio': sharpe_ratio
})
metrics_df = metrics_df.round(4)

st.dataframe(metrics_df.style.format({
    'Expected Return': '{:.2%}',
    'Volatility': '{:.2%}',
    'Sharpe Ratio': '{:.2f}'
}), width='stretch')

# Correlation matrix
st.markdown("**Correlation Matrix**")
corr_matrix = returns.corr()

fig = px.imshow(
    corr_matrix,
    labels=dict(color="Correlation"),
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1
)
fig.update_layout(height=400)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

st.subheader("Step 3: Portfolio Optimization")

optimization_method = st.radio(
    "Optimization Method",
    options=["Manual Weights", "Mean-Variance Optimization", "Riskfolio-Lib (Advanced)"],
    horizontal=True
)

if optimization_method == "Manual Weights":
    st.markdown("**Set Portfolio Weights:**")
    
    weights = {}
    cols = st.columns(len(tickers))
    
    for i, ticker in enumerate(tickers):
        with cols[i % len(cols)]:
            weights[ticker] = st.slider(
                ticker,
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(tickers),
                step=0.05,
                key=f"weight_{ticker}"
            )
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    weights_array = np.array([weights[t] for t in tickers])
    
    # Calculate portfolio metrics
    port_return = np.dot(weights_array, ann_returns)
    port_vol = np.sqrt(np.dot(weights_array.T, np.dot(returns.cov() * 252, weights_array)))
    port_sharpe = port_return / port_vol
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Return", f"{port_return:.2%}")
    with col2:
        st.metric("Portfolio Volatility", f"{port_vol:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
    
    # Pie chart of weights
    fig = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        title="Portfolio Allocation"
    )
    st.plotly_chart(fig, width='stretch')

elif optimization_method == "Mean-Variance Optimization":
    st.markdown("**Efficient Frontier Simulation**")
    
    num_portfolios = st.slider("Number of Random Portfolios", 1000, 10000, 5000)
    
    if st.button("Generate Efficient Frontier", type="primary"):
        with st.spinner("Simulating portfolios..."):
            n_assets = len(tickers)
            
            # Generate random portfolios
            port_returns = []
            port_vols = []
            port_weights = []
            
            cov_matrix = returns.cov() * 252
            
            for _ in range(num_portfolios):
                # Random weights
                w = np.random.random(n_assets)
                w = w / w.sum()
                
                # Portfolio metrics
                ret = np.dot(w, ann_returns)
                vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                
                port_returns.append(ret)
                port_vols.append(vol)
                port_weights.append(w)
            
            port_returns = np.array(port_returns)
            port_vols = np.array(port_vols)
            sharpe_ratios = port_returns / port_vols
            
            # Find optimal portfolios
            max_sharpe_idx = sharpe_ratios.argmax()
            min_vol_idx = port_vols.argmin()
            
            st.success("Efficient frontier generated!")
            
            # Plot
            fig = go.Figure()
            
            # All portfolios
            fig.add_trace(go.Scatter(
                x=port_vols,
                y=port_returns,
                mode='markers',
                marker=dict(
                    size=5,
                    color=sharpe_ratios,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe')
                ),
                name='Portfolios',
                text=[f'Sharpe: {s:.2f}' for s in sharpe_ratios],
                hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}'
            ))
            
            # Max Sharpe portfolio
            fig.add_trace(go.Scatter(
                x=[port_vols[max_sharpe_idx]],
                y=[port_returns[max_sharpe_idx]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name=f'Max Sharpe ({sharpe_ratios[max_sharpe_idx]:.2f})'
            ))
            
            # Min Volatility portfolio
            fig.add_trace(go.Scatter(
                x=[port_vols[min_vol_idx]],
                y=[port_returns[min_vol_idx]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='diamond'),
                name=f'Min Volatility ({port_vols[min_vol_idx]:.2%})'
            ))
            
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Show optimal weights
            st.markdown("**Optimal Portfolio Weights (Max Sharpe):**")
            optimal_weights = port_weights[max_sharpe_idx]
            weights_df = pd.DataFrame({
                'Asset': tickers,
                'Weight': optimal_weights
            })
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
            st.dataframe(weights_df, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Return", f"{port_returns[max_sharpe_idx]:.2%}")
            with col2:
                st.metric("Optimal Volatility", f"{port_vols[max_sharpe_idx]:.2%}")
            with col3:
                st.metric("Optimal Sharpe", f"{sharpe_ratios[max_sharpe_idx]:.2f}")

else:  # Riskfolio-Lib
    st.markdown("**Advanced Optimization with Riskfolio-Lib**")
    
    objective = st.selectbox(
        "Optimization Objective",
        options=["Max Sharpe Ratio", "Min Volatility", "Max Return", "Risk Parity"]
    )
    
    if st.button("Optimize with Riskfolio", type="primary"):
        try:
            import riskfolio as rp
            
            with st.spinner("Optimizing portfolio..."):
                # Create portfolio object
                port = rp.Portfolio(returns=returns)
                
                # Calculate optimal portfolio
                port.assets_stats(method_mu='hist', method_cov='hist')
                
                if objective == "Max Sharpe Ratio":
                    w = port.optimization(model='Classic', rm='MV', obj='Sharpe')
                elif objective == "Min Volatility":
                    w = port.optimization(model='Classic', rm='MV', obj='MinRisk')
                elif objective == "Max Return":
                    w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
                else:  # Risk Parity
                    w = port.rp_optimization(model='Classic', rm='MV')
                
                st.success("Portfolio optimized!")
                
                # Display weights
                st.markdown("**Optimal Weights:**")
                weights_df = w.round(4)
                weights_df.columns = ['Weight']
                st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}), width='stretch')
                
                # Pie chart
                fig = px.pie(
                    values=w.values.flatten(),
                    names=w.index,
                    title=f"Optimal Portfolio ({objective})"
                )
                st.plotly_chart(fig, width='stretch')
                
                # Plot efficient frontier
                st.markdown("**Efficient Frontier:**")
                frontier = port.efficient_frontier(model='Classic', rm='MV', points=50)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=frontier['Risk'],
                    y=frontier['Return'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#1E88E5', width=3)
                ))
                
                fig2.update_layout(
                    title='Efficient Frontier (Riskfolio)',
                    xaxis_title='Volatility',
                    yaxis_title='Return',
                    template='plotly_white'
                )
                st.plotly_chart(fig2, width='stretch')
        
        except ImportError:
            st.error("Riskfolio-Lib is not installed. Please install it with: `pip install riskfolio-lib`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# =============================================================================
# NAVIGATION
# =============================================================================

st.markdown("---")
st.markdown("**Next step:** Check out Resources to learn more about quantitative finance.")

add_footer()
