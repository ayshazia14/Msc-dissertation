from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "config.py").exists() else SCRIPT_DIR.parent

CONFIG_PATH = ROOT / "config.py"
spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load configuration from {CONFIG_PATH}")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

INITIAL_AMOUNT = float(config.INITIAL_AMOUNT)
TRANSACTION_COST = float(config.TRANSACTION_COST)
INDICATORS = list(config.INDICATORS)

CADENCE = 5
HMAX = 100
DATA_PATH = ROOT / "data" / "test_data.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def environment_kwargs(stock_dimension: int) -> dict:
    state_space = 1 + (stock_dimension * 2) + (
        stock_dimension * len(INDICATORS)
    )
    return {
        "hmax": HMAX,
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
        "print_verbosity": 10_000,
    }


def step_environment(environment: StockTradingEnv, action: np.ndarray):
    output = environment.step(np.asarray(action, dtype=np.float32).copy())
    if len(output) == 5:
        observation, reward, terminated, truncated, info = output
        done = bool(terminated or truncated)
    else:
        observation, reward, done, info = output
        done = bool(done)
    return observation, reward, done, info


def load_test_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {DATA_PATH}")

    test_df = pd.read_csv(DATA_PATH)
    required = {"date", "tic", "close", "turbulence", *INDICATORS}
    missing = sorted(required - set(test_df.columns))
    if missing:
        raise RuntimeError(
            "test_data.csv is missing required columns: " + ", ".join(missing)
        )

    test_df = test_df.sort_values(["date", "tic"]).reset_index(drop=True)
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
    test_df["tic"] = test_df["tic"].astype(str)
    dates = test_df["date"].drop_duplicates().tolist()
    tickers = sorted(test_df["tic"].unique().tolist())

    counts = test_df.groupby("date")["tic"].nunique()
    bad_dates = counts[counts != len(tickers)]
    if not bad_dates.empty:
        raise RuntimeError(
            "Every date must contain all tickers. Bad dates: "
            + ", ".join(bad_dates.index[:5].tolist())
        )

    test_df.index = test_df["date"].factorize()[0]
    return test_df, tickers, dates


def action_memory_frame(environment: StockTradingEnv) -> pd.DataFrame:
    frame = environment.save_action_memory().copy()
    if "date" not in frame.columns:
        frame = frame.reset_index()
    return frame


def equity_frame(environment: StockTradingEnv) -> pd.DataFrame:
    frame = environment.save_asset_memory().copy()
    if "date" not in frame.columns:
        frame = frame.reset_index()
    return frame[["date", "account_value"]]


def run_weekly_backtest(
    model_class,
    model_path: Path,
    agent_name: str,
    test_df: pd.DataFrame,
    tickers: list[str],
    dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not model_path.exists() and not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"Missing model {model_path}. Run agents/train_drl.py once to train it."
        )

    print(f"Running 5-trading-day out-of-sample backtest for {agent_name}...")
    environment = StockTradingEnv(
        df=test_df,
        **environment_kwargs(len(tickers)),
    )
    reset_output = environment.reset()
    observation = reset_output[0] if isinstance(reset_output, tuple) else reset_output
    model = model_class.load(str(model_path))

    total_steps = len(dates) - 1
    total_decisions = (total_steps + CADENCE - 1) // CADENCE
    audit_rows: list[dict] = []
    decision_count = 0

    for step_index in range(total_steps):
        date = dates[step_index]
        decision_day = step_index % CADENCE == 0
        if decision_day:
            predicted_action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(predicted_action, dtype=np.float32).reshape(-1)
            if action.shape != (len(tickers),):
                raise RuntimeError(
                    f"{agent_name} returned action shape {action.shape}; "
                    f"expected {(len(tickers),)}"
                )
            action = np.clip(action, -1.0, 1.0)
            status = "model_decision"
            decision_count += 1
        else:
            action = np.zeros(len(tickers), dtype=np.float32)
            status = "scheduled_hold"

        audit_rows.append(
            {
                "date": date,
                "step": step_index,
                "decision_day": bool(decision_day),
                "actions": action.tolist(),
                "status": status,
            }
        )
        observation, _, done, _ = step_environment(environment, action)
        if done and step_index + 1 < total_steps:
            raise RuntimeError(
                f"{agent_name} environment ended at step {step_index}"
            )

    equity = equity_frame(environment)
    executed_actions = action_memory_frame(environment)
    audit = pd.DataFrame(audit_rows)

    if len(equity) != len(dates):
        raise RuntimeError(
            f"{agent_name}: expected {len(dates)} equity rows, got {len(equity)}"
        )
    if len(executed_actions) != total_steps:
        raise RuntimeError(
            f"{agent_name}: expected {total_steps} action rows, "
            f"got {len(executed_actions)}"
        )
    if decision_count != total_decisions:
        raise RuntimeError(
            f"{agent_name}: expected {total_decisions} decisions, got {decision_count}"
        )

    print(
        f"{agent_name}: {decision_count} genuine decisions, "
        f"{total_steps - decision_count} scheduled holds, {len(equity)} valuations."
    )
    return equity, executed_actions, audit


def backup_and_publish_standard(
    agent_key: str,
    equity: pd.DataFrame,
    actions: pd.DataFrame,
) -> None:
    standard_equity = RESULTS_DIR / f"{agent_key}_test_equity_curve.csv"
    standard_actions = RESULTS_DIR / f"{agent_key}_test_actions.csv"
    backup_dir = RESULTS_DIR / "daily_cadence_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for source in (standard_equity, standard_actions):
        if source.exists():
            destination = backup_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)

    atomic_csv_write(equity, standard_equity)
    atomic_csv_write(actions, standard_actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing PPO/A2C models every 5 trading days"
    )
    parser.add_argument(
        "--publish-standard",
        action="store_true",
        help=(
            "Also replace standard PPO/A2C result CSVs after backing up existing "
            "daily-cadence files"
        ),
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_df, tickers, dates = load_test_data()
    print(
        f"Loaded {len(tickers)} tickers across {len(dates)} valuation dates. "
        f"Expected weekly decisions: {(len(dates) - 1 + CADENCE - 1) // CADENCE}."
    )

    specifications = [
        ("ppo", PPO, MODELS_DIR / "ppo_finrl_djia", "PPO"),
        ("a2c", A2C, MODELS_DIR / "a2c_finrl_djia", "A2C"),
    ]
    for key, model_class, model_path, label in specifications:
        equity, actions, audit = run_weekly_backtest(
            model_class,
            model_path,
            label,
            test_df,
            tickers,
            dates,
        )
        atomic_csv_write(
            equity, RESULTS_DIR / f"{key}_weekly_test_equity_curve.csv"
        )
        atomic_csv_write(
            actions, RESULTS_DIR / f"{key}_weekly_test_actions.csv"
        )
        atomic_csv_write(
            audit, RESULTS_DIR / f"{key}_weekly_decision_audit.csv"
        )
        if args.publish_standard:
            backup_and_publish_standard(key, equity, actions)

    print("\nWeekly PPO/A2C backtests completed.")
    if args.publish_standard:
        print("Standard result filenames were updated; prior files were backed up.")
    else:
        print("Weekly-specific files were saved; standard daily files were untouched.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)