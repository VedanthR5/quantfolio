# 🚀 VR Quantfolio Intro

> Interactive Quantitative Finance Toolkit with Python

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vr-quantfolio.streamlit.app)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://vedanthr5.github.io/vr-quantfolio-intro)

## 📊 Overview

This project provides an interactive web application for exploring quantitative finance concepts, including:

- **📈 Stock Data Analysis**: Fetch and visualize historical stock data
- **🤖 AutoML Stock Prediction**: Train ML models using PyCaret
- **🔮 Time Series Forecasting**: ARIMA and NeuralProphet implementations
- **💼 Portfolio Optimization**: Efficient frontier and risk-adjusted returns

## 🎯 Features

### Streamlit App

- **Data Selection**: Choose stocks and date ranges
- **Visualization**: Interactive Plotly charts
- **AutoML Training**: One-click model comparison
- **ARIMA Forecasting**: Walk-forward validation
- **Portfolio Optimization**: Mean-variance optimization

### Jupyter Tutorials

- **ARIMA Fundamentals**: Step-by-step time series forecasting guide
- Detailed explanations of stationarity, differencing, and cumsum reversal

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/vedanthr5/vr-quantfolio-intro.git
cd vr-quantfolio-intro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running the App

### Streamlit App

```bash
streamlit run 🏠_Home.py
```

Then open http://localhost:8501 in your browser.

### Jupyter Tutorials

```bash
jupyter notebook tutorials/
```

## 📁 Project Structure

```
vr-quantfolio-intro/
├── home.py                 # Main Streamlit entry point
├── requirements.txt           # Python dependencies
├── _quarto.yml               # Quarto config for GitHub Pages
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── pages/                     # Streamlit multipage app
│   ├── 1_📊_Data_Selection.py
│   ├── 2_📈_Visualization.py
│   ├── 3_🤖_AutoML_Training.py
│   ├── 4_💾_Export_Model.py
│   ├── 5_🔮_ARIMA_Prediction.py
│   ├── 6_💼_Portfolio_Optimization.py
│   └── 7_📚_Resources.py
│
├── utils/                     # Shared utilities
│   ├── __init__.py
│   ├── data_fetcher.py       # yfinance data utilities
│   └── styles.py             # CSS and styling
│
├── tutorials/                 # Jupyter notebooks
│   └── arima_fundamentals.ipynb
│
└── docs/                      # GitHub Pages output
    └── index.html
```

## 📚 Tutorials

### ARIMA Time Series Forecasting

The [arima_fundamentals.ipynb](tutorials/arima_fundamentals.ipynb) notebook covers:

1. Data loading and exploration
2. Stationarity testing (ADF test)
3. Differencing transformation
4. Walk-forward ARIMA training
5. Cumulative sum reversal
6. Error metrics (MSE, SMAPE)

## 🔧 Technologies

| Category          | Tools                      |
| ----------------- | -------------------------- |
| **Web App**       | Streamlit                  |
| **Data**          | pandas, yfinance           |
| **ML**            | PyCaret, scikit-learn      |
| **Time Series**   | statsmodels, NeuralProphet |
| **Portfolio**     | Riskfolio-Lib              |
| **Visualization** | Plotly, matplotlib         |
| **Documentation** | Quarto, GitHub Pages       |

## 📈 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set main file to `home.py`

### GitHub Pages (Quarto)

```bash
# Install Quarto
# https://quarto.org/docs/get-started/

# Render notebooks to HTML
quarto render

# Push docs/ folder to GitHub
git add docs/
git commit -m "Update GitHub Pages"
git push
```

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 👤 Author

**Vedanth R**

- GitHub: [@vedanthr5](https://github.com/vedanthr5)

---

⭐ Star this repo if you find it helpful!
