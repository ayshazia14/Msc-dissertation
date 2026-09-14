import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="FinRL Agent Benchmarking Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
HERE = Path(__file__).resolve().parent
BASE_CANDIDATES = [HERE, HERE.parent]
BASE_DIR = next(
    (candidate for candidate in BASE_CANDIDATES if (candidate / "results").exists()),
    HERE,
)
RESULTS_DIR = BASE_DIR / "results"

# Styling 
st.markdown(
    """
    <style>
        .stApp { background-color: #0F1729; }

        section[data-testid="stSidebar"] {
            width: 300px !important;
            min-width: 300px !important;
            transform: none !important;
            visibility: visible !important;
        }

        button[data-testid="baseButton-header"] {
            display: none !important;
        }

        [data-testid="metric-container"] {
            background: #162035;
            border: 1px solid #1E2A3A;
            border-radius: 10px;
            padding: 16px 20px;
        }

        [data-testid="metric-container"] label {
            color: #B0BEC5 !important;
            font-size: 12px !important;
        }

        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-size: 28px !important;
            font-weight: 700 !important;
        }

        .section-header {
            font-size: 18px;
            font-weight: 700;
            color: #FFFFFF;
            margin: 28px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #1E2A3A;
        }

        .insight-box {
            background: #0D2748;
            border-left: 4px solid #2196F3;
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin: 12px 0;
            color: #B0BEC5;
            font-size: 14px;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #0A1020;
            border-bottom: 1px solid #1E2A3A;
        }

        .stTabs [data-baseweb="tab"] { color: #B0BEC5; }

        .stTabs [aria-selected="true"] {
            color: #2196F3 !important;
            border-bottom: 2px solid #2196F3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants / helpers
LLM_NAME = "LLM (GPT-OSS-120B)"
ALL_AGENTS = ["PPO", "A2C", LLM_NAME, "Buy & Hold", "Equal Weight", "DJIA Index"]

COLORS = {
    "PPO": "#2196F3",
    "A2C": "#4CAF50",
    LLM_NAME: "#FF9800",
    "Buy & Hold": "#9E9E9E",
    "Equal Weight": "#CE93D8",
    "DJIA Index": "#00BCD4",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0F1729",
    font=dict(color="#B0BEC5", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E2A3A", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#1E2A3A", showline=False, zeroline=False),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#1E2A3A",
        borderwidth=1,
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)

REGIMES = {
    "Full Period": ("2020-07-01", "2022-05-27"),
    "Post-COVID Rebound (Jul–Dec 2020)": ("2020-07-01", "2020-12-31"),
    "Bull Market (Jan–Dec 2021)": ("2021-01-01", "2021-12-31"),
    "Bear Market (Jan–May 2022)": ("2022-01-01", "2022-05-27"),
}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"

def merged_layout(**overrides):
    base = {k: v for k, v in PLOT_LAYOUT.items() if k not in overrides}
    base.update(overrides)
    return base

def read_csv(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

def standardise_agent_names(series: pd.Series) -> pd.Series:
    return series.replace(
        {
            "openai/gpt-oss-120b": LLM_NAME,
            "gpt-oss-120b": LLM_NAME,
            "LLM (openai/gpt-oss-120b)": LLM_NAME,
            "LLM (Llama 3.3 70B)": LLM_NAME,
            "Llama 3.3 70B": LLM_NAME,
        }
    )

@st.cache_data
def load_data():
    ppo = read_csv("ppo_test_equity_curve.csv")
    a2c = read_csv("a2c_test_equity_curve.csv")
    llm = read_csv("llm_test_equity_curve.csv")
    bah = read_csv("bah_equity_curve.csv")
    ew = read_csv("ew_equity_curve.csv")
    djia = read_csv("djia_equity_curve.csv")
    bootstrap_ci = read_csv("bootstrap_sharpe_ci.csv")

    # Buy & Hold / Equal Weight
    metrics_file = read_csv("metrics_summary.csv")
    if metrics_file is not None and "Agent" in metrics_file.columns:
        metrics_file = metrics_file.copy()
        metrics_file["Agent"] = standardise_agent_names(metrics_file["Agent"])
        metrics_file = metrics_file.drop_duplicates(subset=["Agent"], keep="first")

    actions = read_csv("llm_test_actions.csv")
    reasoning = read_csv("llm_reasoning_log.csv")

    if actions is not None:
        actions = actions.copy()

        if "api_cost" not in actions.columns and "cumulative_cost" not in actions.columns:
            actions["api_cost"] = 0.0007
            actions["cumulative_cost"] = actions["api_cost"].cumsum()

        elif "api_cost" not in actions.columns and "cumulative_cost" in actions.columns:
            actions["api_cost"] = actions["cumulative_cost"].diff().fillna(
                actions["cumulative_cost"].iloc[0]
            )

        elif "cumulative_cost" not in actions.columns and "api_cost" in actions.columns:
            actions["cumulative_cost"] = actions["api_cost"].fillna(0).cumsum()

    if bootstrap_ci is not None and "Agent" in bootstrap_ci.columns:
        bootstrap_ci = bootstrap_ci.copy()
        bootstrap_ci["Agent"] = standardise_agent_names(bootstrap_ci["Agent"])
        bootstrap_ci = bootstrap_ci.drop_duplicates(subset=["Agent"], keep="first")

    return ppo, a2c, llm, bah, ew, djia, metrics_file, bootstrap_ci, actions, reasoning

def filter_dates(df, start_date, end_date):
    if df is None or "date" not in df.columns:
        return None

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()

def compute_drawdown(df):
    if df is None or len(df) == 0:
        return np.array([])

    values = pd.to_numeric(df["account_value"], errors="coerce").dropna()
    if values.empty:
        return np.array([])

    running_max = values.cummax()
    return ((values - running_max) / running_max * 100).to_numpy()

def compute_metrics_from_curve(df, agent_name):
    """Recalculate metrics from the currently displayed equity curve."""
    if df is None or len(df) < 2:
        return None

    required = {"date", "account_value"}
    if not required.issubset(df.columns):
        return None

    clean = df[["date", "account_value"]].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["account_value"] = pd.to_numeric(
        clean["account_value"],
        errors="coerce"
    )
    clean = (
        clean.dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(clean) < 2:
        return None

    values = clean["account_value"]
    returns = values.pct_change().dropna()

    if returns.empty:
        return None

    start_value = float(values.iloc[0])
    end_value = float(values.iloc[-1])
    cumulative_return = (
        end_value / start_value - 1.0
    ) * 100.0

    elapsed_days = (
        clean["date"].iloc[-1] - clean["date"].iloc[0]
    ).days

    annualised_return = (
        (
            (end_value / start_value)
            ** (365.0 / elapsed_days)
            - 1.0
        ) * 100.0
        if elapsed_days > 0 and start_value > 0
        else np.nan
    )

    return_std = returns.std(ddof=1)

    annualised_volatility = (
        return_std * np.sqrt(252) * 100.0
    )

    sharpe = (
        returns.mean() / return_std * np.sqrt(252)
        if pd.notna(return_std) and return_std > 0
        else np.nan
    )

    running_max = values.cummax()
    drawdowns = (values - running_max) / running_max
    max_drawdown = drawdowns.min() * 100.0

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)

    sortino = (
        returns.mean() / downside_std * np.sqrt(252)
        if (
            len(downside) > 1
            and pd.notna(downside_std)
            and downside_std > 0
        )
        else np.nan
    )

    calmar = (
        annualised_return / abs(max_drawdown)
        if pd.notna(annualised_return) and max_drawdown != 0
        else np.nan
    )

    win_rate = (returns > 0).mean() * 100.0

    return {
        "Agent": agent_name,
        "Cumulative Return (%)": cumulative_return,
        "Annualised Return (%)": annualised_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown,
        "Annualised Volatility (%)": annualised_volatility,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Win Rate (%)": win_rate,
    }

def metrics_dataframe(data_map):
    rows = []
    for agent_name, df in data_map.items():
        result = compute_metrics_from_curve(df, agent_name)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)

# Load data
ppo, a2c, llm, bah, ew, djia, metrics_file, bootstrap_ci, actions, reasoning = load_data()

FULL_DATA_MAP = {
    "PPO": ppo,
    "A2C": a2c,
    LLM_NAME: llm,
    "Buy & Hold": bah,
    "Equal Weight": ew,
    "DJIA Index": djia,
}

if actions is not None and "api_cost" in actions.columns:
    llm_total_cost = float(
        pd.to_numeric(actions["api_cost"], errors="coerce")
        .fillna(0)
        .sum()
    )
elif actions is not None and "cumulative_cost" in actions.columns:
    llm_total_cost = float(
        pd.to_numeric(
            actions["cumulative_cost"],
            errors="coerce"
        ).max()
    )
else:
    llm_total_cost = 0.0


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    agents_to_show = st.multiselect(
        "Select Strategies / Benchmarks",
        ALL_AGENTS,
        default=ALL_AGENTS,
    )

    st.markdown("---")

    market_regime = st.selectbox(
        "Market Regime",
        list(REGIMES.keys()),
    )

    st.markdown("---")

    show_annotations = st.toggle("Show regime annotations", value=True)
    show_volatility = st.toggle("Show rolling volatility", value=False)

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.markdown("**Daily valuations:** 482 trading dates")
    st.markdown("**Active decision cadence:** Every 5 trading days")
    st.markdown("**Active-agent decisions:** 97 each")
    st.markdown("**Starting capital:** $1,000,000")
    st.markdown("**Transaction cost:** 0.1%")
    st.markdown("**Universe:** Fixed 28-stock DJIA subset")
    st.markdown(f"**Estimated LLM inference cost:** ${llm_total_cost:.4f}")

    st.markdown("---")
    st.markdown("*MSc Data Science & AI*")
    st.markdown("*University of Liverpool*")

# Apply selected date range
start_date, end_date = REGIMES[market_regime]

DATA_MAP = {
    agent: filter_dates(df, start_date, end_date)
    for agent, df in FULL_DATA_MAP.items()
}

metrics = metrics_dataframe(DATA_MAP)
m = metrics.set_index("Agent") if not metrics.empty else pd.DataFrame()

# Header
col_h1, col_h2 = st.columns([3, 1])

with col_h1:
    st.markdown("# 📈 FinRL Agent Benchmarking Dashboard")
    st.markdown(
        "**Fixed 28-Stock DJIA Subset · Test Period: July 2020 – May 2022 · Starting Capital: $1,000,000**"
    )

with col_h2:
    regime_colors = {
        "Full Period": "#2196F3",
        "Post-COVID Rebound (Jul–Dec 2020)": "#4CAF50",
        "Bull Market (Jan–Dec 2021)": "#FF9800",
        "Bear Market (Jan–May 2022)": "#EF5350",
    }
    color = regime_colors[market_regime]

    st.markdown(
        f"""
        <div style='background:{color}22;border:1px solid {color};border-radius:8px;
                    padding:12px 16px;margin-top:16px;text-align:center'>
            <div style='color:{color};font-weight:700;font-size:13px'>{market_regime}</div>
            <div style='color:#B0BEC5;font-size:11px'>{start_date} → {end_date}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Section 1: Summary Metrics
st.markdown(
    '<div class="section-header">Section 1 — Summary Metrics</div>',
    unsafe_allow_html=True,
)

selected_available = [a for a in agents_to_show if not metrics.empty and a in m.index]

if selected_available:
    metric_cols = st.columns(len(selected_available))
    for col, agent in zip(metric_cols, selected_available):
        row = m.loc[agent]
        label = "LLM" if agent == LLM_NAME else agent
        col.metric(
            label=label,
            value=f"{row['Cumulative Return (%)']:.2f}%",
            delta=(
                f"Sharpe {row['Sharpe Ratio']:.3f} · "
                f"DD {row['Max Drawdown (%)']:.2f}%"
            ),
            delta_color="off",
        )
else:
    st.info("Select at least one available agent from the sidebar.")

# Secondary summary cards remain useful even when agents are filtered.
summary_cols = st.columns(4)

if not metrics.empty and "PPO" in m.index:
    summary_cols[0].metric(
        "PPO Sharpe Ratio",
        f"{m.loc['PPO', 'Sharpe Ratio']:.3f}",
        delta="Risk-adjusted performance",
        delta_color="off",
    )
    summary_cols[1].metric(
        "PPO Max Drawdown",
        f"{m.loc['PPO', 'Max Drawdown (%)']:.2f}%",
        delta="Peak-to-trough loss",
        delta_color="off",
    )

summary_cols[2].metric(
    "Estimated LLM Inference Cost",
    f"${llm_total_cost:.4f}",
    delta="97 genuine model decisions",
    delta_color="off",
)

if not metrics.empty and LLM_NAME in m.index:
    llm_win_rate = float(m.loc[LLM_NAME, "Win Rate (%)"])
    ppo_win_rate = float(m.loc["PPO", "Win Rate (%)"]) if "PPO" in m.index else np.nan
    diff = llm_win_rate - ppo_win_rate if pd.notna(ppo_win_rate) else np.nan
    delta_text = f"{diff:+.2f}pp vs PPO" if pd.notna(diff) else "Current period"
    summary_cols[3].metric(
        "LLM Win Rate",
        f"{llm_win_rate:.2f}%",
        delta=delta_text,
    )

# Section 2: Portfolio Value
st.markdown(
    '<div class="section-header">Section 2 — Portfolio Value Over Time</div>',
    unsafe_allow_html=True,
)

fig_eq = go.Figure()

for agent in agents_to_show:
    df = DATA_MAP.get(agent)
    if df is None or len(df) == 0:
        continue

    # Return relative to the first point in the selected period.
    base_value = df["account_value"].iloc[0]
    pct_return = (df["account_value"] / base_value - 1.0) * 100.0

    fig_eq.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["account_value"],
            name=agent,
            customdata=pct_return,
            line=dict(
                color=COLORS[agent],
                width=2.5,
                dash="dash" if agent in {"Buy & Hold", "DJIA Index"} else "solid",
            ),
            hovertemplate=(
                "<b>%{data.name}</b><br>"
                "%{x|%b %d, %Y}<br>"
                "Value: $%{y:,.0f}<br>"
                "Return: %{customdata:.2f}%<extra></extra>"
            ),
        )
    )

if show_annotations and market_regime == "Full Period":
    for date, label, annotation_color in [
        ("2020-10-01", "Post-COVID<br>Rebound", "#4CAF50"),
        ("2021-06-01", "Bull Market<br>2021", "#FF9800"),
        ("2022-03-01", "Bear Market<br>2022", "#EF5350"),
    ]:
        fig_eq.add_vline(
            x=date,
            line_dash="dot",
            line_color=annotation_color,
            line_width=1,
            opacity=0.5,
        )
        fig_eq.add_annotation(
            x=date,
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(color=annotation_color, size=10),
            bgcolor="rgba(15,23,41,0.8)",
            borderpad=4,
        )

fig_eq.update_layout(
    **merged_layout(
        height=420,
        yaxis_tickformat="$,.0f",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
    )
)
st.plotly_chart(fig_eq, width="stretch")

# rolling volatility
if show_volatility:
    st.markdown(
        '<div class="section-header">Rolling 30-Day Annualised Volatility</div>',
        unsafe_allow_html=True,
    )

    fig_vol = go.Figure()
    rolling_window = 30

    for agent in agents_to_show:
        df = DATA_MAP.get(agent)
        if df is None or len(df) <= rolling_window:
            continue

        daily_returns = df["account_value"].pct_change()
        rolling_vol = (
            daily_returns.rolling(rolling_window).std() * np.sqrt(252) * 100
        )

        fig_vol.add_trace(
            go.Scatter(
                x=df["date"],
                y=rolling_vol,
                name=agent,
                line=dict(
                    color=COLORS[agent],
                    width=2,
                    dash="dash" if agent in {"Buy & Hold", "DJIA Index"} else "solid",
                ),
                hovertemplate=(
                    "<b>%{data.name}</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "Annualised Vol: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig_vol.update_layout(
        **merged_layout(
            height=280,
            xaxis_title="Date",
            yaxis_title=f"{rolling_window}-Day Rolling Annualised Volatility (%)",
        )
    )
    st.plotly_chart(fig_vol, width="stretch")

# Section 3: Drawdown
st.markdown(
    '<div class="section-header">Section 3 — Drawdown Over Time</div>',
    unsafe_allow_html=True,
)

fig_dd = go.Figure()

for agent in agents_to_show:
    df = DATA_MAP.get(agent)
    if df is None or len(df) == 0:
        continue

    dd = compute_drawdown(df)
    x_values = df["date"].iloc[-len(dd):]

    fig_dd.add_trace(
        go.Scatter(
            x=x_values,
            y=dd,
            name=agent,
            line=dict(color=COLORS[agent], width=2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(COLORS[agent], 0.08),
            hovertemplate=(
                "<b>%{data.name}</b><br>"
                "%{x|%b %d, %Y}<br>"
                "Drawdown: %{y:.2f}%<extra></extra>"
            ),
        )
    )

fig_dd.update_layout(
    **merged_layout(
        height=280,
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
    )
)
st.plotly_chart(fig_dd, width="stretch")

# Section 4: Full Metrics Comparison
st.markdown(
    '<div class="section-header">Section 4 — Full Metrics Comparison</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📊 Comparison Table", "📈 Radar Chart"])

with tab1:
    display_cols = [
        "Agent",
        "Cumulative Return (%)",
        "Annualised Return (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Annualised Volatility (%)",
        "Sortino Ratio",
        "Calmar Ratio",
        "Win Rate (%)",
    ]

    if not metrics.empty:
        display_df = metrics[
            metrics["Agent"].isin(agents_to_show)
        ][display_cols].copy()

        numeric_cols = [c for c in display_cols if c != "Agent"]
        display_df[numeric_cols] = display_df[numeric_cols].round(3)
        display_df = display_df.rename(
            columns={"Agent": "Strategy / Benchmark"}
        )
        st.dataframe(display_df, width="stretch", hide_index=True)
    else:
        st.info("Metrics are unavailable for the selected period.")

    # Bootstrap intervals apply only to the complete test period.
    if market_regime == "Full Period":
        if bootstrap_ci is not None and not bootstrap_ci.empty:
            st.markdown(
                "#### Bootstrap 95% Confidence Intervals — Sharpe Ratio"
            )

            ci = bootstrap_ci.copy()
            ci["Agent"] = standardise_agent_names(ci["Agent"])
            ci = ci[ci["Agent"].isin(agents_to_show)].copy()

            ci_display = ci[
                ["Agent", "Sharpe", "CI Lower", "CI Upper"]
            ].copy()

            for column in ["Sharpe", "CI Lower", "CI Upper"]:
                ci_display[column] = pd.to_numeric(
                    ci_display[column],
                    errors="coerce"
                ).round(3)

            ci_display["95% CI"] = ci_display.apply(
                lambda row: (
                    f"[{row['CI Lower']:.3f}, "
                    f"{row['CI Upper']:.3f}]"
                ),
                axis=1,
            )

            st.dataframe(
                ci_display[["Agent", "Sharpe", "95% CI"]].rename(
                    columns={"Agent": "Strategy / Benchmark"}
                ),
                width="stretch",
                hide_index=True,
            )

            if "PPO" in ci["Agent"].values:
                ci_indexed = ci.set_index("Agent")
                ppo_low = float(
                    ci_indexed.loc["PPO", "CI Lower"]
                )
                ppo_high = float(
                    ci_indexed.loc["PPO", "CI Upper"]
                )

                overlaps = []

                for agent in ci_indexed.index:
                    if agent == "PPO":
                        continue

                    low = float(
                        ci_indexed.loc[agent, "CI Lower"]
                    )
                    high = float(
                        ci_indexed.loc[agent, "CI Upper"]
                    )

                    overlaps.append(
                        not (
                            high < ppo_low
                            or low > ppo_high
                        )
                    )

                if overlaps and all(overlaps):
                    st.info(
                        "All displayed 95% Sharpe confidence "
                        "intervals overlap with PPO. This provides "
                        "no clear evidence of different full-period "
                        "Sharpe ratios; interval overlap is not a "
                        "formal pairwise hypothesis test."
                    )
        else:
            st.info(
                "Bootstrap confidence intervals are unavailable."
            )
    else:
        st.caption(
            "Bootstrap Sharpe confidence intervals are reported "
            "for the complete 482-date test period only."
        )

with tab2:

    radar_agents = [
        a for a in agents_to_show
        if not metrics.empty and a in m.index
    ]

    categories = [
        "Sharpe",
        "Sortino",
        "Calmar",
        "Win Rate (/10)"
    ]

    if radar_agents:

        raw_vals = {
            agent: [
                float(m.loc[agent, "Sharpe Ratio"]),
                float(m.loc[agent, "Sortino Ratio"]),
                float(m.loc[agent, "Calmar Ratio"]),
                float(m.loc[agent, "Win Rate (%)"]) / 10.0,
            ]
            for agent in radar_agents
        }

        # Normalise each metric independently for visual comparison.
        metric_max = []
        for i in range(len(categories)):
            vals = [raw_vals[a][i] for a in radar_agents if pd.notna(raw_vals[a][i])]
            metric_max.append(max(vals) if vals and max(vals) != 0 else 1.0)

        fig_radar = go.Figure()

        for agent in radar_agents:
            vals = raw_vals[agent]
            norm = [
                (0 if pd.isna(v) else v / metric_max[i])
                for i, v in enumerate(vals)
            ]

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=norm + [norm[0]],
                    theta=categories + [categories[0]],
                    name=agent,
                    line=dict(
                        color=COLORS[agent],
                        width=3 if agent == "Buy & Hold" else 2,
                        dash="dash" if agent in {"Buy & Hold", "DJIA Index"} else "solid",
                    ),
                    fill="toself",
                    fillcolor=hex_to_rgba(COLORS[agent], 0.13),
                )
            )

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0F1729",
                radialaxis=dict(
                    visible=True,
                    gridcolor="#1E2A3A",
                    color="#B0BEC5",
                ),
                angularaxis=dict(
                    gridcolor="#1E2A3A",
                    color="#B0BEC5",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B0BEC5"),
            showlegend=True,
            height=400,
        )
        st.plotly_chart(fig_radar, width="stretch")
    else:
        st.info("Select at least one agent to display the radar chart.")


# Section 5: Daily Return Distribution
st.markdown(
    '<div class="section-header">Section 5 — Daily Return Distribution</div>',
    unsafe_allow_html=True,
)

fig_dist = go.Figure()

for agent in agents_to_show:
    df = DATA_MAP.get(agent)
    if df is None or len(df) < 2:
        continue

    returns = df["account_value"].pct_change().dropna() * 100

    fig_dist.add_trace(
        go.Violin(
            x=returns,
            name=agent,
            line_color=COLORS[agent],
            fillcolor=hex_to_rgba(COLORS[agent], 0.3),
            opacity=0.8,
            meanline_visible=True,
            orientation="h",
            side="positive",
            width=1.8,
            points=False,
        )
    )

fig_dist.update_layout(
    **merged_layout(
        height=340,
        xaxis_title="Daily Return (%)",
        yaxis_title="Strategy / Benchmark",
        violingap=0.1,
        violinmode="overlay",
    )
)
st.plotly_chart(fig_dist, width="stretch")


# Section 6: LLM Cost Efficiency
st.markdown(
    '<div class="section-header">Section 6 — LLM Cost Efficiency</div>',
    unsafe_allow_html=True,
)

if actions is not None and llm is not None and "api_cost" in actions.columns:
    actions_f = filter_dates(actions, start_date, end_date)
    llm_f = DATA_MAP.get(LLM_NAME)

    if actions_f is not None and llm_f is not None and len(actions_f) > 0:
        actions_f = actions_f.copy()
        actions_f["period_cumulative_cost"] = (
            pd.to_numeric(actions_f["api_cost"], errors="coerce")
            .fillna(0)
            .cumsum()
        )

        llm_merged = llm_f.merge(
            actions_f[["date", "period_cumulative_cost", "api_cost"]],
            on="date",
            how="left",
        )

        llm_merged["period_cumulative_cost"] = (
            llm_merged["period_cumulative_cost"]
            .ffill()
            .fillna(0)
        )
        llm_merged["api_cost"] = (
            llm_merged["api_cost"].fillna(0)
        )

        base_value = llm_merged["account_value"].iloc[0]
        llm_merged["return_pct"] = (
            llm_merged["account_value"] / base_value - 1.0
        ) * 100.0

        if "decision_day" in actions_f.columns:
            period_decisions = int(
                actions_f["decision_day"]
                .astype(str)
                .str.lower()
                .eq("true")
                .sum()
            )
        else:
            period_decisions = int(
                pd.to_numeric(
                    actions_f["api_cost"],
                    errors="coerce"
                ).fillna(0).gt(0).sum()
            )

        ce_c1, ce_c2 = st.columns([2, 1])

        with ce_c1:
            fig_cost = go.Figure()
            fig_cost.add_trace(
                go.Scatter(
                    x=llm_merged["period_cumulative_cost"],
                    y=llm_merged["return_pct"],
                    mode="lines",
                    line=dict(color="#FF9800", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(255,152,0,0.1)",
                    hovertemplate=(
                        "<b>LLM Agent</b><br>"
                        "Estimated Cost: $%{x:.4f}<br>"
                        "Return: %{y:.2f}%<extra></extra>"
                    ),
                )
            )
            fig_cost.update_layout(
                **merged_layout(
                    height=320,
                    xaxis_title="Estimated Cumulative API Cost ($)",
                    yaxis_title="Cumulative Return (%)",
                )
            )
            st.plotly_chart(fig_cost, width="stretch")

        with ce_c2:
            period_total_cost = float(actions_f["period_cumulative_cost"].max())
            st.metric(
                "Estimated Inference Cost",
                f"${period_total_cost:.4f}"
            )
            st.metric(
                "Daily Valuations Covered",
                str(len(llm_merged))
            )
            st.metric(
                "Genuine LLM Decisions",
                str(period_decisions)
            )
    else:
        st.info("Cost data not available for the selected period.")
else:
    st.info("Cost data not available.")

# Section 7: Market Regime Analysis
st.markdown(
    '<div class="section-header">Section 7 — Market Regime Analysis</div>',
    unsafe_allow_html=True,
)
regime_ranges = [
    ("Post-COVID Rebound", "2020-07-01", "2020-12-31"),
    ("Bull Market 2021", "2021-01-01", "2021-12-31"),
    ("Bear Market 2022", "2022-01-01", "2022-05-27"),
]

regime_data = []
for regime_name, regime_start, regime_end in regime_ranges:
    for agent_name, df in FULL_DATA_MAP.items():
        sub = filter_dates(df, regime_start, regime_end)
        if sub is None or len(sub) < 2:
            continue

        start_value = sub["account_value"].iloc[0]
        end_value = sub["account_value"].iloc[-1]
        regime_return = (end_value / start_value - 1.0) * 100.0

        regime_data.append(
            {
                "Regime": regime_name,
                "Agent": agent_name,
                "Return (%)": round(regime_return, 2),
            }
        )
if regime_data:
    regime_df = pd.DataFrame(regime_data)
    regime_df["Agent Short"] = regime_df["Agent"].replace(
        {LLM_NAME: "LLM", "Buy & Hold": "B&H", "Equal Weight": "EW"}
    )

    fig_regime = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Post-COVID Rebound (2020)",
            "Bull Market (2021)",
            "Bear Market (2022)",
        ],
        horizontal_spacing=0.08,
    )

    regime_order = ["Post-COVID Rebound", "Bull Market 2021", "Bear Market 2022"]

    for col_idx, regime_name in enumerate(regime_order, start=1):
        sub = regime_df[regime_df["Regime"] == regime_name]

        for agent in agents_to_show:
            row = sub[sub["Agent"] == agent]
            if row.empty:
                continue

            val = float(row["Return (%)"].iloc[0])
            short = row["Agent Short"].iloc[0]

            fig_regime.add_trace(
                go.Bar(
                    x=[short],
                    y=[val],
                    name=agent,
                    marker_color=COLORS[agent],
                    opacity=0.9,
                    text=[f"{val:+.1f}%"],
                    textposition="outside",
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"<b>{agent}</b><br>"
                        f"Regime: {regime_name}<br>"
                        "Return: %{y:+.2f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

    fig_regime.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F1729",
        font=dict(color="#B0BEC5", family="Inter, sans-serif"),
        height=380,
        legend=dict(
            bgcolor="rgba(15,23,41,0.8)",
            bordercolor="#1E2A3A",
            borderwidth=1,
            orientation="h",
            y=1.18,
            x=0.28,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    fig_regime.update_xaxes(
        gridcolor="#1E2A3A",
        showline=False,
        zeroline=True,
        zerolinecolor="#1E2A3A",
    )
    fig_regime.update_yaxes(
        gridcolor="#1E2A3A",
        showline=False,
        zeroline=True,
        zerolinecolor="#1E2A3A",
    )

    st.plotly_chart(fig_regime, width="stretch")
else:
    st.info("Regime analysis data is unavailable.")

# Section 8: LLM Decision Trace Viewer
st.markdown('<div class="section-header">Section 8 — LLM Reasoning Trace Viewer</div>', unsafe_allow_html=True)
if reasoning is not None:
    reasoning_f = reasoning[(reasoning["date"] >= start_date) & (reasoning["date"] <= end_date)].copy()
    valid = reasoning_f[reasoning_f["reasoning"].astype(str).str.strip().str.len() > 5]
    available_dates = valid["date"].dt.strftime("%Y-%m-%d").tolist()
    if available_dates:
        rv_c1, rv_c2, rv_c3 = st.columns([1, 2, 1])
        with rv_c1:
            selected_date = st.selectbox("Select LLM decision date:", available_dates)
 
        row = valid[valid["date"].dt.strftime("%Y-%m-%d") == selected_date]
        if not row.empty:
            raw = row["reasoning"].values[0]
 
            # Parse the reasoning JSON
            rationale_text = None
            actions_dict   = None
            try:
                cleaned = raw.strip()
                if "```" in cleaned:
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                parsed = json.loads(cleaned)
 
                # Handle both {rationale, actions} and flat action-only dicts
                if "rationale" in parsed:
                    rationale_text = parsed["rationale"]
                    actions_dict   = parsed.get("actions", {})
                else:
                    # Flat dict — all values are action scores
                    actions_dict = {k: v for k, v in parsed.items() if isinstance(v, (int, float))}
 
            except Exception:
                # Not valid JSON — display as plain text
                rationale_text = raw
 
            # Rationale box
            with rv_c2:
                st.markdown("**Model Rationale**")
                if rationale_text:
                    safe_rationale = (
                        str(rationale_text)
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    st.markdown(
                        "<div style='background:#0D2748;border-left:4px solid #2196F3;"
                        "border-radius:0 8px 8px 0;padding:14px 18px;color:#B0BEC5;"
                        "font-size:14px;line-height:1.6;margin-bottom:12px'>"
                        f"{safe_rationale}"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No rationale text found for this date.")
 
            # Top buys / sells summary
            with rv_c3:
                if actions_dict:
                    st.markdown("**Top Buys**")
                    buys = sorted(
                        [(t, v) for t, v in actions_dict.items() if v > 0],
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    for tic, val in buys:
                        st.markdown(
                            f"<span style='color:#4CAF50;font-weight:bold'>{tic}</span>"
                            f" <span style='color:#B0BEC5'>{val:+.2f}</span>",
                            unsafe_allow_html=True,
                        )
 
                    st.markdown("**Top Sells**")
                    sells = sorted(
                        [(t, v) for t, v in actions_dict.items() if v < 0],
                        key=lambda x: x[1]
                    )[:5]
                    for tic, val in sells:
                        st.markdown(
                            f"<span style='color:#EF5350;font-weight:bold'>{tic}</span>"
                            f" <span style='color:#B0BEC5'>{val:+.2f}</span>",
                            unsafe_allow_html=True,
                        )
                    # Raw JSON expander for transparency
                    with st.expander("Raw JSON"):
                        st.code(raw, language="json")
    else:
        st.info("No reasoning traces available for the selected period.")
else:
    st.info("Reasoning log not found — check if llm_reasoning_log.csv exists in results/.")
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4A5568;font-size:12px;padding:8px'>"
    "MSc Data Science & AI Dissertation · University of Liverpool · Aysha Ziaulhaque"
    "</div>",
    unsafe_allow_html=True,
)