import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_metrics(equity_curve, name="Agent"):
    """Compute all performance metrics from an equity curve."""
    df = equity_curve.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    values = df["account_value"].values
    returns = pd.Series(values).pct_change().dropna()
    
    # Primary metrics
    cumulative_return = (values[-1] - values[0]) / values[0]
    
    n_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    annualised_return = (1 + cumulative_return) ** (365 / n_days) - 1
    
    trading_days_per_year = 252
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_days_per_year)
    
    rolling_max = pd.Series(values).cummax()
    drawdowns = (pd.Series(values) - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    annualised_volatility = returns.std() * np.sqrt(trading_days_per_year)
    
    # Secondary metrics
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

def compute_baselines(test_df):
    """Compute buy-and-hold and equal weight baselines."""
    INITIAL_AMOUNT = 1_000_000
    dates = sorted(test_df["date"].unique())
    
    # Buy and hold — invest equally in all stocks on day 1
    first_day = test_df[test_df["date"] == dates[0]]
    tickers = first_day["tic"].unique()
    allocation_per_stock = INITIAL_AMOUNT / len(tickers)
    
    bah_holdings = {}
    for _, row in first_day.iterrows():
        shares = allocation_per_stock / row["close"]
        bah_holdings[row["tic"]] = shares
    
    bah_curve = []
    ew_curve = []
    
    for date in dates:
        day = test_df[test_df["date"] == date]
        prices = {row["tic"]: row["close"] for _, row in day.iterrows()}
        
        bah_value = sum(bah_holdings.get(tic, 0) * prices.get(tic, 0) for tic in tickers)
        bah_curve.append({"date": date, "account_value": bah_value})
        
        # Equal weight rebalanced daily
        ew_value = sum(allocation_per_stock * (prices.get(tic, 0) / first_day[first_day["tic"] == tic]["close"].values[0]) for tic in tickers)
        ew_curve.append({"date": date, "account_value": ew_value})
    
    return pd.DataFrame(bah_curve), pd.DataFrame(ew_curve)

if __name__ == "__main__":
    print("Loading results...")
    
    ppo = pd.read_csv("results/ppo_test_equity_curve.csv")
    a2c = pd.read_csv("results/a2c_test_equity_curve.csv")
    llm = pd.read_csv("results/llm_test_equity_curve.csv")
    
    test_df = pd.read_csv("data/test_data.csv")
    test_df = test_df.sort_values(["date", "tic"]).reset_index(drop=True)
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
    
    print("Computing baselines...")
    bah_curve, ew_curve = compute_baselines(test_df)
    
    print("Computing metrics...")
    results = []
    results.append(compute_metrics(ppo, "PPO"))
    results.append(compute_metrics(a2c, "A2C"))
    results.append(compute_metrics(llm, "LLM (Llama 3.3 70B)"))
    results.append(compute_metrics(bah_curve, "Buy & Hold"))
    results.append(compute_metrics(ew_curve, "Equal Weight"))
    
    metrics_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*80)
    print(metrics_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/metrics_summary.csv", index=False)
    bah_curve.to_csv("results/bah_equity_curve.csv", index=False)
    ew_curve.to_csv("results/ew_equity_curve.csv", index=False)
    
    print("\nResults saved to results/metrics_summary.csv")