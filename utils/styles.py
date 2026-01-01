"""
Shared CSS styling for VR Quantfolio Intro
==========================================
Unified styling across all pages of the multipage Streamlit app.
"""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header styling */
        h1 {
            color: #1E88E5;
            font-weight: 700;
        }
        
        h2, h3 {
            color: #424242;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Metric cards with gradient */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card h3 {
            color: white;
            margin: 0;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        /* Success/Error/Info boxes */
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .info-box {
            background-color: #cce5ff;
            border: 1px solid #b8daff;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* DataFrame styling */
        .dataframe {
            font-size: 0.85rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #1E88E5;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #1E88E5;
        }
        
        /* Code blocks */
        code {
            background-color: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        
        /* Links */
        a {
            color: #1E88E5;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #888;
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)


def metric_card(title: str, value: str, delta: Optional[str] = None) -> str:
    """
    Generate HTML for a gradient metric card.
    
    Args:
        title: The metric title
        value: The metric value
        delta: Optional delta/change indicator
    
    Returns:
        HTML string for the metric card
    """
    delta_html = f'<p style="margin:0; opacity:0.8;">{delta}</p>' if delta else ''
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <p class="value">{value}</p>
        {delta_html}
    </div>
    """


def info_box(message: str, box_type: str = "info") -> None:
    """
    Display a styled info/success/error box.
    
    Args:
        message: The message to display
        box_type: One of "info", "success", "error"
    """
    st.markdown(f'<div class="{box_type}-box">{message}</div>', unsafe_allow_html=True)
