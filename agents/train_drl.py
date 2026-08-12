import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C
from agents.llm_agent import LLMTradingAgent

from config import INITIAL_AMOUNT, TRANSACTION_COST, INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def train_and_backtest():
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

    '''
    Phase 2: Backtesting the trained models
    week 3 of proposal: we will backtest the trained models using the test dataset and evaluate their performance.
    '''
    print("\nPhase 2: Backtesting the trained models")

    def backtest_agent(model_class, model_path, agent_name):
        print(f"Running out-of-sample backtest for {agent_name}")
        
        # Initialize environment structures natively
        raw_test_env = StockTradingEnv(df=test_df, **env_kwargs)
        sb_test_env, _ = raw_test_env.get_sb_env()
        model = model_class.load(model_path, env=sb_test_env)

        # Reset the raw environment and safely extract the initial observation array
        reset_output = raw_test_env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            # Step directly on the raw environment to prevent auto-reset wiping out data
            step_output = raw_test_env.step(action)
            
            # Accommodate both 4-value (legacy Gym) and 5-value (modern Gym/Gymnasium) returns
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                obs, reward, done, info = step_output

        # Extract history logs from the raw environment before it gets touched
        df_account_value = raw_test_env.save_asset_memory()
        df_actions = raw_test_env.save_action_memory()
        
        return df_account_value, df_actions
    
    def backtest_llm_agent(agent_name="LLM_GPT4o"):
        print(f"\nRunning out-of-sample backtest for {agent_name}")
        
        raw_test_env = StockTradingEnv(df=test_df, **env_kwargs)
        llm_agent = LLMTradingAgent(model_name="gpt-4o-mini")
        
        reset_output = raw_test_env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        done = False
        
        unique_dates = sorted(test_df['date'].unique())
        step_idx = 0
        
        while not done and step_idx < len(unique_dates):
            current_date = unique_dates[step_idx]
            day_market_data = test_df[test_df['date'] == current_date]
            
            action = llm_agent.generate_portfolio_actions(current_date, day_market_data)

            action_np = np.array(action, dtype=np.float32)
            step_output = raw_test_env.step(action_np)
            
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                obs, reward, done, info = step_output
                
            step_idx += 1
            if step_idx % 10 == 0:
                print(f"Processed {step_idx}/{len(unique_dates)} days for LLM...")

        df_account_value = raw_test_env.save_asset_memory()
        df_actions = raw_test_env.save_action_memory()
        return df_account_value, df_actions
    
    ppo_equity, ppo_actions = backtest_agent(PPO, "models/ppo_finrl_djia", "PPO")
    a2c_equity, a2c_actions = backtest_agent(A2C, "models/a2c_finrl_djia", "A2C")
    
    llm_equity, llm_actions = backtest_llm_agent("GPT-4o-Mini")

    ppo_equity.to_csv("results/ppo_test_equity_curve.csv", index=False)
    a2c_equity.to_csv("results/a2c_test_equity_curve.csv", index=False)
    ppo_actions.to_csv("results/ppo_test_actions.csv", index=False)
    a2c_actions.to_csv("results/a2c_test_actions.csv", index=False)
    
    llm_equity.to_csv("results/llm_test_equity_curve.csv", index=False)
    llm_actions.to_csv("results/llm_test_actions.csv", index=False)

    print("\nBacktesting completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    print("Initializing DRL Training and Backtesting Pipeline...")
    train_and_backtest()