import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import json
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("config", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
INITIAL_AMOUNT = config.INITIAL_AMOUNT
TRANSACTION_COST = config.TRANSACTION_COST
INDICATORS = config.INDICATORS
DJIA_TICKERS = config.DJIA_TICKERS

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

def build_market_prompt(date, stock_data):
    lines = []
    lines.append(f"Date: {date}")
    lines.append(f"You are managing a diversified portfolio of DJIA stocks.")
    lines.append(f"\nCurrent market conditions for each stock:")
    
    for _, row in stock_data.iterrows():
        lines.append(
            f"- {row['tic']}: Close=${row['close']:.2f}, "
            f"MACD={row['macd']:.3f}, "
            f"RSI={row['rsi_30']:.1f}, "
            f"CCI={row['cci_30']:.1f}, "
            f"ADX={row['dx_30']:.1f}"
        )
    
    lines.append(f"\nTurbulence index: {stock_data['turbulence'].iloc[0]:.2f}")
    lines.append(
        f"\nTask: For each stock, decide an action between -1.0 (full sell) "
        f"and +1.0 (full buy), where 0 means hold. "
        f"Consider momentum, trend strength, and market turbulence."
    )
    lines.append(
        f"\nRespond ONLY with a JSON object mapping ticker symbols to action values. "
        f"Example: {{\"AAPL\": 0.5, \"MSFT\": -0.3, \"JPM\": 0.0}}"
        f"\nInclude all {len(stock_data)} tickers in your response."
    )
    
    return "\n".join(lines)

def get_llm_action(date, stock_data, tickers):
    prompt = build_market_prompt(date, stock_data)
    
    try:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trader. You analyse technical indicators and make precise trading decisions. Always respond with valid JSON only, no explanation outside the JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limit hit on {date}, switching to 8B model...")
                response = client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert quantitative trader. You analyse technical indicators and make precise trading decisions. Always respond with valid JSON only, no explanation outside the JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
            else:
                raise e

        raw = response.choices[0].message.content.strip()
        
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        actions_dict = json.loads(raw)
        actions = np.array([actions_dict.get(tic, 0.0) for tic in tickers])
        actions = np.clip(actions, -1.0, 1.0)
        
        reasoning = raw
        cost = (response.usage.prompt_tokens + response.usage.completion_tokens) / 1000 * 0.00059
        
        return actions, reasoning, cost

    except Exception as e:
        print(f"LLM error on {date}: {e}")
        return np.zeros(len(tickers)), "", 0.0

def run_llm_backtest(test_df, tickers, completed_dates=None):
    if completed_dates is None:
        completed_dates = set()

    dates = sorted(test_df["date"].unique())
    
    portfolio_value = INITIAL_AMOUNT
    cash = INITIAL_AMOUNT
    holdings = {tic: 0 for tic in tickers}
    
    equity_curve = []
    actions_log = []
    reasoning_log = []
    total_cost = 0.0
    
    remaining = [d for d in dates if d not in completed_dates]
    print(f"Running LLM backtest across {len(remaining)} remaining trading days...")
    
    for i, date in enumerate(dates):
        if date in completed_dates:
            continue

        day_data = test_df[test_df["date"] == date]
        prices = {row["tic"]: row["close"] for _, row in day_data.iterrows()}
        
        actions, reasoning, cost = get_llm_action(date, day_data, tickers)
        total_cost += cost
        
        for j, tic in enumerate(tickers):
            if tic not in prices:
                continue
            price = prices[tic]
            action = actions[j]
            shares_to_trade = int(abs(action) * 100)
            
            if action > 0.1 and cash >= shares_to_trade * price:
                cost_trade = shares_to_trade * price * (1 + TRANSACTION_COST)
                if cash >= cost_trade:
                    cash -= cost_trade
                    holdings[tic] += shares_to_trade
            elif action < -0.1 and holdings[tic] > 0:
                shares_to_sell = min(shares_to_trade, holdings[tic])
                cash += shares_to_sell * price * (1 - TRANSACTION_COST)
                holdings[tic] -= shares_to_sell
        
        stock_value = sum(holdings[tic] * prices.get(tic, 0) for tic in tickers)
        portfolio_value = cash + stock_value
        
        equity_curve.append({"date": date, "account_value": portfolio_value})
        actions_log.append({"date": date, "actions": actions.tolist(), "api_cost": cost, "cumulative_cost": total_cost})
        reasoning_log.append({"date": date, "reasoning": reasoning})
        
        processed = i + 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(dates)} days | Portfolio: ${portfolio_value:,.2f} | API cost so far: ${total_cost:.4f}")
    
    print(f"\nLLM backtest complete!")
    print(f"Final portfolio value: ${portfolio_value:,.2f}")
    print(f"Total API cost: ${total_cost:.4f}")
    
    return pd.DataFrame(equity_curve), pd.DataFrame(actions_log), reasoning_log


if __name__ == "__main__":
    print("Loading test data...")
    test_df = pd.read_csv("data/test_data.csv")
    test_df = test_df.sort_values(["date", "tic"]).reset_index(drop=True)
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
    
    tickers = sorted(test_df["tic"].unique().tolist())
    print(f"Tickers: {len(tickers)} stocks")
    
    checkpoint_path = "results/llm_checkpoint.csv"
    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path)
        completed_dates = set(checkpoint["date"].tolist())
        print(f"Resuming from checkpoint — {len(completed_dates)} days already completed")
    else:
        completed_dates = set()
        print("Starting fresh run")
    
    equity_curve, actions_log, reasoning_log = run_llm_backtest(
        test_df, tickers, completed_dates=completed_dates
    )
    
    os.makedirs("results", exist_ok=True)

    if os.path.exists(checkpoint_path) and len(completed_dates) > 0:
        old = pd.read_csv(checkpoint_path)
        equity_curve = pd.concat([old[["date","account_value"]], equity_curve]).drop_duplicates("date").sort_values("date")
    
    equity_curve.to_csv("results/llm_test_equity_curve.csv", index=False)
    equity_curve.to_csv(checkpoint_path, index=False)
    actions_log.to_csv("results/llm_test_actions.csv", index=False)
    
    reasoning_df = pd.DataFrame(reasoning_log)
    reasoning_df.to_csv("results/llm_reasoning_log.csv", index=False)
    
    print("Results saved to results/")