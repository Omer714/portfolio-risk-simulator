# Portfolio Risk Simulator

An interactive web app for analyzing multi-asset portfolio risk and return, built with Python and Streamlit.

**Live app:** https://portfolio-risk-simulator-zz5vm3tgefxw87poesenmk.streamlit.app/

## What it does

This tool lets you build a 5-asset portfolio (SPY, QQQ, TLT, GLD, VNQ), adjust the weights, and instantly see a full risk/return analysis — including metrics most free portfolio tools don't offer.

## Features

- **Core metrics** — annualised return, volatility, Sharpe ratio, and historical max drawdown
- **Value at Risk (VaR) & Conditional VaR (CVaR)** — adjustable confidence level (90/95/99%) and time horizon (1 month to 3 years), with results shown in both % and dollar terms
- **Risk contribution analysis** — shows how much risk each asset actually contributes to the portfolio, not just its capital weight (e.g., an asset can be 10% of your capital but reduce your overall risk)
- **Auto-generated narrative** — plain-English summary of the portfolio's risk profile, based on the calculated metrics
- **Monte Carlo simulation** — 1,000 simulated future paths with interactive percentile bands (hover to see exact values at any point in time)
- **Efficient Frontier** — 10,000 randomly simulated portfolios plotted against your own, with the historically optimal Minimum Volatility and Maximum Sharpe portfolios identified
- **Optimizer comparison** — see your portfolio's return/risk/Sharpe side-by-side with the two optimal portfolios found in the simulation

## Data

Historical daily closing prices for SPY, QQQ, TLT, GLD, and VNQ, covering **January 1, 2016 to September 5, 2026**.

## Exploratory analysis

`analysis.ipynb` contains the original step-by-step Jupyter notebook used to develop and test the underlying calculations (returns, correlation, VaR/CVaR, efficient frontier simulation) before they were built into the Streamlit app.

## Tech stack

Python · Streamlit · pandas · NumPy · Matplotlib · Plotly

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This is a personal analytics/portfolio project built to explore quantitative risk concepts (VaR, CVaR, Markowitz optimization, Monte Carlo simulation) and translate them into an accessible, interactive tool. It is not financial advice — past performance and historical correlations don't guarantee future results.
