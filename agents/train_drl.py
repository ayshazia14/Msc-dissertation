import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C

from config import INITIAL_AMOUNT, TRANSACTION_COST, INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def train_models():
    print('Loading training dataset...')
    train_df = pd.read_csv('data/train_data.csv').sort_values(["date","tic"]).reset_index(drop=True)
    test_df = pd.read_csv('data/test_data.csv').sort_values(["date","tic"]).reset_index(drop=True)

    train_df.index = train_df["date"].factorize()[0]
    test_df.index = test_df["date"].factorize()[0]

    stock_dimension = len(train_df.tic.unique())
    state_space = 1 + (stock_dimension * 2) + (stock_dimension * len(INDICATORS))

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
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    '''
    Phase 1: Training the model
    following the DRL approach using PPO and A2C algorithms.
    week 2 of proposal: we will train the model using the training dataset and save the trained model for backtesting.
    '''
    print("\nPhase 1: Training the model")
    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    # Train PPO model
    print("\nTraining PPO model")
    model_ppo = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/ppo/"
    )

    '''
    50k timesteps is a good starting point to see convergence, but you may want to increase this for better performance.
    The model will be saved to the 'models' directory for later use in backtesting.
    '''
    model_ppo.learn(total_timesteps=50000, tb_log_name="ppo_run_1")
    model_ppo.save("models/ppo_finrl_djia")
    print("PPO training finished. Weights saved to models/ppo_finrl_djia.zip")

    # Train A2C model
    print("\nTraining A2C model")
    model_a2c = A2C(
        "MlpPolicy",
        env_train,
        learning_rate=0.0007,
        n_steps=5,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/a2c/"
    )

    model_a2c.learn(total_timesteps=50000, tb_log_name="a2c_run_1")
    model_a2c.save("models/a2c_finrl_djia")
    print("A2C training finished. Weights saved to models/a2c_finrl_djia.zip")

    print("\nTraining completed. Run "
          "python agents/backtest_drl_weekly.py "
          "for the final five-day-cadence evaluation.")

if __name__ == "__main__":
    print("Initializing DRL Training Pipeline...")
    train_models()