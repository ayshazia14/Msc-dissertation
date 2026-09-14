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

def compute_baselines(
    test_df,
    transaction_cost_pct=0.001,
    rebalance_every=5
):
    """Compute Buy & Hold and five-trading-day Equal Weight baselines."""

    initial_amount = 1_000_000

    prices = (
        test_df.pivot(index="date", columns="tic", values="close")
        .sort_index()
    )

    if prices.isna().any().any():
        raise ValueError("Missing prices detected in baseline data.")

    dates = prices.index.tolist()
    price_array = prices.to_numpy(dtype=float)
    stock_count = price_array.shape[1]

    # Initial purchase including the 0.1% transaction cost.
    investable_amount = initial_amount / (1 + transaction_cost_pct)

    # Buy & Hold: purchase once on the first date.
    bah_shares = (
        investable_amount / stock_count
    ) / price_array[0]

    bah_values = [initial_amount]

    for day_index in range(1, len(dates)):
        bah_values.append(
            float((bah_shares * price_array[day_index]).sum())
        )

    # Equal Weight: initial allocation, then rebalance every five
    # trading days. Portfolio value is still recorded every day.
    ew_shares = (
        investable_amount / stock_count
    ) / price_array[0]

    ew_values = [initial_amount]

    for day_index in range(1, len(dates)):
        current_prices = price_array[day_index]
        current_asset_values = ew_shares * current_prices
        pre_trade_value = float(current_asset_values.sum())

        # Record the value before today's action, matching the FinRL
        # environment's daily valuation/transition convention.
        ew_values.append(pre_trade_value)

        if day_index % rebalance_every == 0 and day_index < len(dates) - 1:
            # Solve transaction cost and post-trade target iteratively.
            post_trade_value = pre_trade_value

            for _ in range(20):
                target_value = post_trade_value / stock_count
                turnover = float(
                    abs(target_value - current_asset_values).sum()
                )
                updated_value = (
                    pre_trade_value
                    - transaction_cost_pct * turnover
                )

                if abs(updated_value - post_trade_value) < 1e-8:
                    break

                post_trade_value = updated_value

            ew_shares = (
                post_trade_value / stock_count
            ) / current_prices

    bah_curve = pd.DataFrame({
        "date": dates,
        "account_value": bah_values
    })

    ew_curve = pd.DataFrame({
        "date": dates,
        "account_value": ew_values
    })

    return bah_curve, ew_curve

if __name__ == "__main__":
    print("Loading results...")

    ppo = pd.read_csv("results/ppo_test_equity_curve.csv")
    a2c = pd.read_csv("results/a2c_test_equity_curve.csv")
    llm = pd.read_csv("results/llm_test_equity_curve.csv")

    test_df = pd.read_csv("data/test_data.csv")
    test_df = test_df.sort_values(["date", "tic"]).reset_index(drop=True)
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")

    print("Computing baselines...")
    bah_curve, ew_curve = compute_baselines(test_df, transaction_cost_pct=0.001)  # matches 0.1% agent cost

    print("Computing metrics...")
    results = []
    results.append(compute_metrics(ppo, "PPO"))
    results.append(compute_metrics(a2c, "A2C"))
    results.append(compute_metrics(llm, "LLM (openai/gpt-oss-120b)"))
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