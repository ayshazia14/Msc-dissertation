import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="FinRL Agent Benchmarking Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 FinRL Agent Benchmarking Dashboard")
st.markdown("**DJIA 30 · Test Period: July 2020 – May 2022 · Starting Capital: $1,000,000**")

# Load data
@st.cache_data
def load_data():
    ppo = pd.read_csv("results/ppo_test_equity_curve.csv")
    a2c = pd.read_csv("results/a2c_test_equity_curve.csv")
    llm = pd.read_csv("results/llm_test_equity_curve.csv")
    bah = pd.read_csv("results/bah_equity_curve.csv")
    metrics = pd.read_csv("results/metrics_summary.csv")

    for df in [ppo, a2c, llm, bah]:
        df["date"] = pd.to_datetime(df["date"])

    try:
        actions = pd.read_csv("results/llm_test_actions.csv")
        actions["date"] = pd.to_datetime(actions["date"])
    except:
        actions = None

    try:
        reasoning = pd.read_csv("results/llm_reasoning_log.csv")
        reasoning["date"] = pd.to_datetime(reasoning["date"])
    except:
        reasoning = None

    return ppo, a2c, llm, bah, metrics, actions, reasoning

ppo, a2c, llm, bah, metrics, actions, reasoning = load_data()

# Sidebar filters
st.sidebar.header("Filters")
agents_to_show = st.sidebar.multiselect(
    "Select Agents",
    ["PPO", "A2C", "LLM (Llama 3.3 70B)", "Buy & Hold"],
    default=["PPO", "A2C", "LLM (Llama 3.3 70B)", "Buy & Hold"]
)

market_regime = st.sidebar.selectbox(
    "Market Regime",
    ["Full Period", "Post-COVID Rebound (Jul–Dec 2020)",
     "Bull Market (Jan–Dec 2021)", "Bear Market (Jan–May 2022)"]
)

regime_dates = {
    "Full Period": ("2020-07-01", "2022-05-31"),
    "Post-COVID Rebound (Jul–Dec 2020)": ("2020-07-01", "2020-12-31"),
    "Bull Market (Jan–Dec 2021)": ("2021-01-01", "2021-12-31"),
    "Bear Market (Jan–May 2022)": ("2022-01-01", "2022-05-31"),
}
start_date, end_date = regime_dates[market_regime]

def filter_df(df):
    return df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

ppo_f = filter_df(ppo)
a2c_f = filter_df(a2c)
llm_f = filter_df(llm)
bah_f = filter_df(bah)

# Section 1: Summary metrics
st.markdown("---")
st.subheader("Section 1 — Summary Metrics")

col1, col2, col3, col4 = st.columns(4)
ppo_row = metrics[metrics["Agent"] == "PPO"].iloc[0]
a2c_row = metrics[metrics["Agent"] == "A2C"].iloc[0]
llm_row = metrics[metrics["Agent"] == "LLM (Llama 3.3 70B)"].iloc[0]
bah_row = metrics[metrics["Agent"] == "Buy & Hold"].iloc[0]

col1.metric("PPO Cumulative Return", f"{ppo_row['Cumulative Return (%)']:.2f}%")
col2.metric("A2C Cumulative Return", f"{a2c_row['Cumulative Return (%)']:.2f}%")
col3.metric("LLM Cumulative Return", f"{llm_row['Cumulative Return (%)']:.2f}%")
col4.metric("Buy & Hold Return", f"{bah_row['Cumulative Return (%)']:.2f}%")

col1b, col2b, col3b, col4b = st.columns(4)
col1b.metric("PPO Sharpe Ratio", f"{ppo_row['Sharpe Ratio']:.3f}")
col2b.metric("A2C Sharpe Ratio", f"{a2c_row['Sharpe Ratio']:.3f}")
col3b.metric("LLM Sharpe Ratio", f"{llm_row['Sharpe Ratio']:.3f}")
col4b.metric("LLM Total API Cost", "$0.33")

# Section 2: Portfolio value over time
st.markdown("---")
st.subheader("Section 2 — Portfolio Value Over Time")

fig_equity = go.Figure()
color_map = {
    "PPO": "#2196F3",
    "A2C": "#4CAF50",
    "LLM (Llama 3.3 70B)": "#FF9800",
    "Buy & Hold": "#9E9E9E"
}

if "PPO" in agents_to_show:
    fig_equity.add_trace(go.Scatter(x=ppo_f["date"], y=ppo_f["account_value"],
        name="PPO", line=dict(color=color_map["PPO"], width=2)))
if "A2C" in agents_to_show:
    fig_equity.add_trace(go.Scatter(x=a2c_f["date"], y=a2c_f["account_value"],
        name="A2C", line=dict(color=color_map["A2C"], width=2)))
if "LLM (Llama 3.3 70B)" in agents_to_show:
    fig_equity.add_trace(go.Scatter(x=llm_f["date"], y=llm_f["account_value"],
        name="LLM (Llama 3.3 70B)", line=dict(color=color_map["LLM (Llama 3.3 70B)"], width=2)))
if "Buy & Hold" in agents_to_show:
    fig_equity.add_trace(go.Scatter(x=bah_f["date"], y=bah_f["account_value"],
        name="Buy & Hold", line=dict(color=color_map["Buy & Hold"], width=2, dash="dash")))

fig_equity.update_layout(
    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    hovermode="x unified", height=450,
    yaxis_tickformat="$,.0f"
)
st.plotly_chart(fig_equity, use_container_width=True)

# Section 3: Drawdown chart
st.markdown("---")
st.subheader("Section 3 — Drawdown Over Time")

def compute_drawdown(df):
    values = df["account_value"].values
    rolling_max = pd.Series(values).cummax()
    return ((pd.Series(values) - rolling_max) / rolling_max * 100).values

fig_dd = go.Figure()
if "PPO" in agents_to_show:
    fig_dd.add_trace(go.Scatter(x=ppo_f["date"], y=compute_drawdown(ppo_f),
        name="PPO", line=dict(color=color_map["PPO"], width=2), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"))
if "A2C" in agents_to_show:
    fig_dd.add_trace(go.Scatter(x=a2c_f["date"], y=compute_drawdown(a2c_f),
        name="A2C", line=dict(color=color_map["A2C"], width=2), fill="tozeroy", fillcolor="rgba(76,175,80,0.1)"))
if "LLM (Llama 3.3 70B)" in agents_to_show:
    fig_dd.add_trace(go.Scatter(x=llm_f["date"], y=compute_drawdown(llm_f),
        name="LLM", line=dict(color=color_map["LLM (Llama 3.3 70B)"], width=2), fill="tozeroy", fillcolor="rgba(255,152,0,0.1)"))
if "Buy & Hold" in agents_to_show:
    fig_dd.add_trace(go.Scatter(x=bah_f["date"], y=compute_drawdown(bah_f),
        name="Buy & Hold", line=dict(color=color_map["Buy & Hold"], width=2, dash="dash")))

fig_dd.update_layout(
    xaxis_title="Date", yaxis_title="Drawdown (%)",
    hovermode="x unified", height=350
)
st.plotly_chart(fig_dd, use_container_width=True)

# Section 4: Metrics comparison table
st.markdown("---")
st.subheader("Section 4 — Metrics Comparison Table")

display_cols = ["Agent", "Cumulative Return (%)", "Annualised Return (%)",
                "Sharpe Ratio", "Max Drawdown (%)", "Annualised Volatility (%)",
                "Sortino Ratio", "Calmar Ratio", "Win Rate (%)"]
st.dataframe(metrics[display_cols].set_index("Agent"), use_container_width=True)

# Section 5: LLM cost efficiency
st.markdown("---")
st.subheader("Section 5 — LLM Cost Efficiency")

if actions is not None and "cumulative_cost" in actions.columns:
    actions_f = actions[(actions["date"] >= start_date) & (actions["date"] <= end_date)]
    llm_merged = llm_f.merge(actions_f[["date", "cumulative_cost"]], on="date", how="left")
    llm_merged["return_pct"] = (llm_merged["account_value"] - 1_000_000) / 1_000_000 * 100

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=llm_merged["cumulative_cost"],
        y=llm_merged["return_pct"],
        mode="lines+markers",
        line=dict(color="#FF9800", width=2),
        marker=dict(size=4),
        name="LLM Return vs Cost"
    ))
    fig_cost.update_layout(
        xaxis_title="Cumulative API Cost ($)",
        yaxis_title="Cumulative Return (%)",
        height=350
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption(f"Total inference cost: $0.33 across 482 trading days (~$0.00068 per day)")
else:
    st.info("Cost data not available for selected period.")

# Section 6: LLM reasoning trace viewer
st.markdown("---")
st.subheader("Section 6 — LLM Reasoning Trace Viewer")

if reasoning is not None:
    reasoning_f = reasoning[(reasoning["date"] >= start_date) & (reasoning["date"] <= end_date)]
    available_dates = reasoning_f["date"].dt.strftime("%Y-%m-%d").tolist()
    available_dates = [d for d in available_dates if reasoning_f[reasoning_f["date"].dt.strftime("%Y-%m-%d") == d]["reasoning"].values[0] != ""]

    if available_dates:
        selected_date = st.selectbox("Select a trading day to view LLM reasoning:", available_dates[:50])
        row = reasoning_f[reasoning_f["date"].dt.strftime("%Y-%m-%d") == selected_date]
        if not row.empty:
            st.code(row["reasoning"].values[0], language="json")
    else:
        st.info("No reasoning traces available for selected period.")
else:
    st.info("Reasoning log not found.")

st.markdown("---")
st.caption("MSc Data Science & AI Dissertation · University of Liverpool · 2025/26")