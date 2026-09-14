"""
Adds a real DJIA Index baseline — fetches actual ^DJI historical prices via
yfinance for your exact test window, normalizes to the same $1M starting
capital and 0.1% transaction cost as your other baselines, and appends it
to metrics_summary.csv.
 
Requires internet access (yfinance hits Yahoo Finance), so run this on your
own machine, not in a sandboxed environment.
 
Install if needed:
    pip install yfinance --break-system-packages
 
Run from your project root:
    python evaluation/add_djia_index.py
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os
 
INITIAL_AMOUNT = 1_000_000
TRANSACTION_COST_PCT = 0.001  # matches every other agent/baseline
TEST_START = "2020-07-01"
TEST_END = "2022-05-31"
 
 
def fetch_djia_curve():
    print(f"Fetching ^DJI from {TEST_START} to {TEST_END}...")

    raw = yf.download(
        "^DJI",
        start=TEST_START,
        end=TEST_END,
        progress=False
    )

    if raw.empty:
        raise RuntimeError(
            "No DJIA data returned; check the internet connection."
        )

    close = raw["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close.index = pd.to_datetime(close.index)

    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close.index = close.index.normalize()
    close = pd.to_numeric(close, errors="coerce")

    test_data = pd.read_csv("data/test_data.csv")
    test_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(test_data["date"].unique()))
    )

    close = close.reindex(test_dates)

    if close.isna().any():
        missing = close[close.isna()].index.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(
            f"DJIA prices are missing for test dates: {missing}"
        )

    first_price = float(close.iloc[0])
    investable_amount = INITIAL_AMOUNT / (
        1 + TRANSACTION_COST_PCT
    )
    shares = investable_amount / first_price

    account_values = close.astype(float) * shares
    account_values.iloc[0] = INITIAL_AMOUNT

    return pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "account_value": account_values.to_numpy()
    })


def compute_metrics(equity_curve, name="DJIA Index"):
    df = equity_curve.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
 
    values = df["account_value"].values
    returns = pd.Series(values).pct_change().dropna()
 
    cumulative_return = (values[-1] - values[0]) / values[0]
    n_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    annualised_return = (1 + cumulative_return) ** (365 / n_days) - 1
 
    trading_days_per_year = 252
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_days_per_year)
 
    rolling_max = pd.Series(values).cummax()
    drawdowns = (pd.Series(values) - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
 
    annualised_volatility = returns.std() * np.sqrt(trading_days_per_year)
 
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 else 0
    calmar_ratio = annualised_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = (returns > 0).sum() / len(returns)
 
    return {
        "Agent": name,
        "Final Value ($)": round(values[-1], 2),
        "Cumulative Return (%)": round(cumulative_return * 100, 2),
        "Annualised Return (%)": round(annualised_return * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Annualised Volatility (%)": round(annualised_volatility * 100, 2),
        "Sortino Ratio": round(sortino_ratio, 3),
        "Calmar Ratio": round(calmar_ratio, 3),
        "Win Rate (%)": round(win_rate * 100, 2),
    }
 
 
if __name__ == "__main__":
    djia_curve = fetch_djia_curve()
    os.makedirs("results", exist_ok=True)
    djia_curve.to_csv("results/djia_equity_curve.csv", index=False)
    print(f"Saved {len(djia_curve)} rows to results/djia_equity_curve.csv")
 
    djia_metrics = compute_metrics(djia_curve, "DJIA Index")
 
    metrics_path = "results/metrics_summary.csv"
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df[metrics_df["Agent"] != "DJIA Index"]  # avoid duplicate if rerun
    metrics_df = pd.concat([metrics_df, pd.DataFrame([djia_metrics])], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
 
    print("\n" + "=" * 70)
    print("DJIA INDEX METRICS")
    print("=" * 70)
    for k, v in djia_metrics.items():
        print(f"{k:<28}{v}")
    print(f"\nAppended to {metrics_path}")