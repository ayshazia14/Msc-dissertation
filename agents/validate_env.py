import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from config import INITIAL_AMOUNT, TRANSACTION_COST, INDICATORS

def validate_environment():
    train_df = pd.read_csv("data/train_data.csv")
    train_df = train_df.sort_values(["date", "tic"]).reset_index(drop=True)
    train_df.index = train_df["date"].factorize()[0]
    train_df["date"] = pd.to_datetime(train_df["date"]).dt.strftime("%Y-%m-%d")
    print(f"Train data shape: {train_df.shape}")

    # Count unique stocks
    stock_dimension = len(train_df.tic.unique())
    print(f"\nNumber of stocks: {stock_dimension}")

    # State space size: balance + stock prices + holdings + indicators
    state_space = 1 + (stock_dimension * 2) + (stock_dimension * len(INDICATORS))
    print(f"State space size: {state_space}")

    # Environment configuration
    env_kwargs = {
        "hmax": 100,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [TRANSACTION_COST] * stock_dimension,
        "sell_cost_pct": [TRANSACTION_COST] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "turbulence_threshold": 380,
        "risk_indicator_col": "turbulence",
    }

    print("\nInitialising StockTradingEnv...")
    env = StockTradingEnv(df=train_df, **env_kwargs)
    obs, _ = env.reset()

    print(f"Observation length: {len(obs)}")
    print(f"Action space: {env.action_space}")
    print(f"Initial portfolio value: ${env.asset_memory[0]:,.2f}")

    print("\nRunning 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1} | Reward: {reward:.4f} | Portfolio: ${env.asset_memory[-1]:,.2f}")

    print("\nEnvironment validation complete!")
    return env_kwargs, stock_dimension

if __name__ == "__main__":
    validate_environment()