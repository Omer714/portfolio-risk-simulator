import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Portfolio Risk Simulator")
st.write("Adjust the weights in the side bar — they should add up to 100%.")

st.sidebar.header("Portfolio Weights")

spy_weight = st.sidebar.slider("SPY", 0, 100, 60)
qqq_weight = st.sidebar.slider("QQQ", 0, 100, 20)
tlt_weight = st.sidebar.slider("TLT", 0, 100, 10)
gld_weight = st.sidebar.slider("GLD", 0, 100, 5)
vnq_weight = st.sidebar.slider("VNQ", 0, 100, 5)

total_weight = spy_weight + qqq_weight + tlt_weight + gld_weight + vnq_weight
st.sidebar.write(f"Total: {total_weight}%")

if total_weight != 100:
    st.sidebar.warning("Weights should add up to 100%")
    st.stop()

st.sidebar.header("Risk Settings")

horizon_label = st.sidebar.selectbox(
    "Time Horizon",
    ["1 Month", "3 Months", "1 Year", "3 Years"],
    index=2
)
horizon_days_map = {"1 Month": 21, "3 Months": 63, "1 Year": 252, "3 Years": 756}
num_days = horizon_days_map[horizon_label]

confidence_label = st.sidebar.selectbox(
    "Confidence Level",
    ["90%", "95%", "99%"],
    index=1
)
confidence_map = {"90%": 10, "95%": 5, "99%": 1}
percentile_cutoff = confidence_map[confidence_label]

st.sidebar.header("Investment Amount")
investment_amount = st.sidebar.number_input(
    "Starting Amount ($)",
    min_value=100,
    max_value=10_000_000,
    value=10000,
    step=500
)

asset_names = ["SPY", "QQQ", "TLT", "GLD", "VNQ"]
weights = {
    "SPY": spy_weight / 100,
    "QQQ": qqq_weight / 100,
    "TLT": tlt_weight / 100,
    "GLD": gld_weight / 100,
    "VNQ": vnq_weight / 100
}

@st.cache_data
def load_data():
    files = {
        "SPY": "SPY_daily_close_Jan2016_Sept2026.xlsx",
        "QQQ": "QQQ_daily_close_Jan2016_Sept2026.xlsx",
        "TLT": "TLT_daily_close_Jan2016_Sept2026.xlsx",
        "GLD": "GLD_daily_close_Jan2016_Sept2026.xlsx",
        "VNQ": "VNQ_daily_close_Jan2016_Sept2026.xlsx"
    }
    prices = pd.DataFrame()
    for ticker, filename in files.items():
        df = pd.read_excel(filename)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        prices[ticker] = df["Close"]
    prices = prices.sort_index()
    returns = prices.pct_change().dropna()
    return returns

returns = load_data()

weights_array = np.array([weights[a] for a in asset_names])
portfolio_returns = (returns[asset_names] * weights_array).sum(axis=1)

portfolio_annual_return = portfolio_returns.mean() * 252
portfolio_annual_volatility = portfolio_returns.std() * (252 ** 0.5)
risk_free_rate = 0.04
sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_annual_volatility

cumulative_returns = (1 + portfolio_returns).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_dd = drawdown.min()

# Monte Carlo simulation — using the selected horizon
np.random.seed(42)
num_simulations = 1000
mean_daily_return = portfolio_returns.mean()
daily_volatility = portfolio_returns.std()

simulated_paths = np.zeros((num_days, num_simulations))
for sim in range(num_simulations):
    daily_returns_sim = np.random.normal(mean_daily_return, daily_volatility, num_days)
    simulated_paths[:, sim] = (1 + daily_returns_sim).cumprod()

final_values = simulated_paths[-1, :]
VaR_pct = 1 - np.percentile(final_values, percentile_cutoff)
worst_pct_values = final_values[final_values <= np.percentile(final_values, percentile_cutoff)]
CVaR_pct = 1 - worst_pct_values.mean()

# Risk contribution per asset
cov_matrix = returns[asset_names].cov() * 252
portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
marginal_contribution = np.dot(cov_matrix, weights_array)
risk_contribution = weights_array * marginal_contribution / portfolio_variance
risk_contribution_pct = pd.Series(risk_contribution * 100, index=asset_names)
weights_pct = pd.Series(weights_array * 100, index=asset_names)

# Efficient Frontier simulation
@st.cache_data
def run_efficient_frontier(_returns, asset_names, risk_free_rate, num_portfolios=10000):
    num_assets = len(asset_names)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    np.random.seed(42)
    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        weights_record.append(w)

        port_return = np.sum(_returns[asset_names].mean() * w) * 252
        port_vol = np.sqrt(np.dot(w.T, np.dot(_returns[asset_names].cov() * 252, w)))
        sharpe = (port_return - risk_free_rate) / port_vol

        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = sharpe

    return results, weights_record

ef_results, ef_weights_record = run_efficient_frontier(returns, asset_names, risk_free_rate)

min_vol_idx = ef_results[1].argmin()
max_sharpe_idx = ef_results[2].argmax()


st.header("Portfolio Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Annual Return", f"{portfolio_annual_return*100:.2f}%")
col2.metric("Annual Volatility", f"{portfolio_annual_volatility*100:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric(
    "Max Drawdown",
    f"{max_dd*100:.2f}%",
    f"-${abs(max_dd)*investment_amount:,.0f}"
)
col5.metric(
    f"VaR ({confidence_label}, {horizon_label})",
    f"{VaR_pct*100:.2f}%",
    f"-${VaR_pct*investment_amount:,.0f}"
)
col6.metric(
    f"CVaR ({confidence_label}, {horizon_label})",
    f"{CVaR_pct*100:.2f}%",
    f"-${CVaR_pct*investment_amount:,.0f}"
)

st.header("What This Means")

top_risk_asset = risk_contribution_pct.idxmax()
top_risk_value = risk_contribution_pct.max()
top_weight_value = weights_pct[top_risk_asset]
hedges = risk_contribution_pct[risk_contribution_pct < 0]

narrative = f"**{top_risk_asset}** makes up {top_weight_value:.0f}% of your portfolio's capital, but contributes {top_risk_value:.0f}% of its total risk.\n\n"

if len(hedges) > 0:
    hedge_names = ", ".join(hedges.index)
    verb = "reduces" if len(hedges) == 1 else "reduce"
    narrative += f"**{hedge_names}** currently {verb} your overall portfolio risk, acting as a hedge rather than added exposure.\n\n"

narrative += f"Your portfolio has a Sharpe ratio of **{sharpe_ratio:.2f}**, meaning you earn {sharpe_ratio:.2f} units of return for every unit of risk taken.\n\n"
narrative += f"Historically, this portfolio's worst peak-to-trough decline was **{max_dd*100:.1f}%**.\n\n"
narrative += (
    f"There's a {100-int(confidence_label.strip('%'))}% chance of losing more than "
    f"**{VaR_pct*100:.1f}%** (**\${VaR_pct*investment_amount:,.0f}**) over a {horizon_label.lower()} horizon — "
    f"and if that happens, the average loss in that worst-case group is closer to "
    f"**{CVaR_pct*100:.1f}%** (**\${CVaR_pct*investment_amount:,.0f}**)."
)

st.markdown(narrative)

# --- Charts ---
st.header("Charts")

st.subheader("Historical Drawdown")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.fill_between(returns.index, drawdown * 100, 0, color="indianred", alpha=0.6)
ax1.set_ylabel("Drawdown (%)")
ax1.set_title("Portfolio Drawdown Over Time")
st.pyplot(fig1)

st.subheader(f"Monte Carlo Simulation — {horizon_label} Forward")

# Calculate percentile bands across all simulations, for each day
p5 = np.percentile(simulated_paths, 5, axis=1) * investment_amount
p25 = np.percentile(simulated_paths, 25, axis=1) * investment_amount
p50 = np.percentile(simulated_paths, 50, axis=1) * investment_amount
p75 = np.percentile(simulated_paths, 75, axis=1) * investment_amount
p95 = np.percentile(simulated_paths, 95, axis=1) * investment_amount

fig2, ax2 = plt.subplots(figsize=(10, 5))

days = np.arange(num_days)

ax2.fill_between(days, p5, p95, color="steelblue", alpha=0.15, label="5th–95th percentile")
ax2.fill_between(days, p25, p75, color="steelblue", alpha=0.3, label="25th–75th percentile")
ax2.plot(days, p50, color="red", linewidth=2, label="Median outcome")

ax2.set_xlabel("Trading Days")
ax2.set_ylabel(f"Portfolio Value (starting at ${investment_amount:,.0f})")
ax2.legend()
st.pyplot(fig2)

st.subheader("Capital Weight vs. Risk Contribution")
fig3, ax3 = plt.subplots(figsize=(10, 5))
x = np.arange(len(asset_names))
width = 0.35
ax3.bar(x - width/2, weights_pct, width, label="Weight (%)", color="steelblue")
ax3.bar(x + width/2, risk_contribution_pct, width, label="Risk Contribution (%)", color="indianred")
ax3.axhline(y=0, color="black", linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(asset_names)
ax3.legend()
st.pyplot(fig3)

st.subheader("Efficient Frontier")
fig4, ax4 = plt.subplots(figsize=(10, 6))

scatter = ax4.scatter(ef_results[1], ef_results[0], c=ef_results[2], cmap="viridis", s=8, alpha=0.5)
plt.colorbar(scatter, ax=ax4, label="Sharpe Ratio")

ax4.scatter(portfolio_annual_volatility, portfolio_annual_return,
            color="red", marker="o", s=120,edgecolor="black", linewidth=1.5, label="Your Portfolio")
ax4.scatter(ef_results[1, min_vol_idx], ef_results[0, min_vol_idx],
            color="blue", marker="o", s=120, edgecolor="black", linewidth=1.5, label="Min Volatility")
ax4.scatter(ef_results[1, max_sharpe_idx], ef_results[0, max_sharpe_idx],
            color="orange", marker="o", s=120, edgecolor="black", linewidth=1.5, label="Max Sharpe")

ax4.set_xlabel("Annualised Volatility (Risk)")
ax4.set_ylabel("Annualised Return")
ax4.legend()
st.pyplot(fig4)