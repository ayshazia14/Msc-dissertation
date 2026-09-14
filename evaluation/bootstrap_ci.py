"""
Bootstrap confidence intervals for Sharpe ratio.

Answers: is PPO's Sharpe (1.123) meaningfully different from Equal Weight's
Sharpe (1.123), or is this apparent tie / any other gap between agents just
noise from a single 482-day sample?

Method: resample the daily returns WITH REPLACEMENT 1,000 times, recompute
Sharpe each time, and report the 2.5th/97.5th percentile as a 95% CI.
This does not require retraining anything — it only needs the equity curve
CSVs you already have.

Run from your project root:
    python evaluation/bootstrap_ci.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
N_BOOTSTRAP = 1000
TRADING_DAYS_PER_YEAR = 252
CONFIDENCE = 0.95


def load_returns(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    returns = df["account_value"].pct_change().dropna().values
    return returns


def sharpe(returns):
    standard_deviation = returns.std(ddof=1)

    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return 0.0

    return (
        returns.mean() / standard_deviation
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)


def bootstrap_sharpe_ci(returns, n_bootstrap=N_BOOTSTRAP, confidence=CONFIDENCE):
    n = len(returns)
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        boot_sharpes[i] = sharpe(sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_sharpes, 100 * alpha / 2)
    upper = np.percentile(boot_sharpes, 100 * (1 - alpha / 2))
    point_estimate = sharpe(returns)
    return point_estimate, lower, upper, boot_sharpes


def cis_overlap(lo1, hi1, lo2, hi2):
    """True if the two confidence intervals overlap at all."""
    return not (hi1 < lo2 or hi2 < lo1)


if __name__ == "__main__":
    agents = {
        "PPO":                       "results/ppo_test_equity_curve.csv",
        "A2C":                       "results/a2c_test_equity_curve.csv",
        "LLM (openai/gpt-oss-120b)": "results/llm_test_equity_curve.csv",
        "Buy & Hold":                "results/bah_equity_curve.csv",
        "Equal Weight":              "results/ew_equity_curve.csv",
        "DJIA Index": "results/djia_equity_curve.csv"
    }

    print("Loading equity curves and computing bootstrap Sharpe CIs...")
    print(f"({N_BOOTSTRAP} resamples per agent, {int(CONFIDENCE*100)}% CI)\n")

    results = {}
    for name, path in agents.items():
        returns = load_returns(path)
        point, lo, hi, _ = bootstrap_sharpe_ci(returns)
        results[name] = (point, lo, hi)

    print(f"{'Agent':<28}{'Sharpe':>10}{'95% CI':>22}")
    print("-" * 60)
    for name, (point, lo, hi) in results.items():
        print(f"{name:<28}{point:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}")

    print("\n" + "=" * 60)
    print("PAIRWISE OVERLAP CHECK (PPO vs each other agent)")
    print("=" * 60)
    ppo_point, ppo_lo, ppo_hi = results["PPO"]
    for name, (point, lo, hi) in results.items():
        if name == "PPO":
            continue
        overlap = cis_overlap(ppo_lo, ppo_hi, lo, hi)
        verdict = "OVERLAP — not statistically distinguishable" if overlap else "NO OVERLAP — distinguishable"
        print(f"PPO vs {name:<28} {verdict}")

    metrics_df = pd.DataFrame([
        {"Agent": name, "Sharpe": point, "CI Lower": lo, "CI Upper": hi}
        for name, (point, lo, hi) in results.items()
    ])
    metrics_df.to_csv("results/bootstrap_sharpe_ci.csv", index=False)
    print("\nSaved to results/bootstrap_sharpe_ci.csv")
