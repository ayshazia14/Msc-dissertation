from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
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

MODEL = "openai/gpt-oss-120b"
CADENCE = 5
HMAX = 100
MIN_SECONDS_BETWEEN_CALLS = 32.0
DAILY_TOKEN_SAFETY_BUDGET = 160_000
DEFAULT_MAX_CALLS_PER_PROCESS = 110
MAX_COMPLETION_TOKENS = 2_048

# These values are only used to make an estimated-cost column. They do not
# control billing and can be overridden in the environment if prices change.
INPUT_USD_PER_MILLION = float(
    os.environ.get("GROQ_INPUT_USD_PER_MILLION", "0.15")
)
OUTPUT_USD_PER_MILLION = float(
    os.environ.get("GROQ_OUTPUT_USD_PER_MILLION", "0.60")
)

DATA_PATH = ROOT / "data" / "test_data.csv"
RESULTS_DIR = ROOT / "results"
STATE_PATH = RESULTS_DIR / "llm_weekly_checkpoint.json"
PROGRESS_EQUITY_PATH = RESULTS_DIR / "llm_weekly_progress_equity.csv"
PROGRESS_ACTIONS_PATH = RESULTS_DIR / "llm_weekly_progress_actions.csv"
PROGRESS_REASONING_PATH = RESULTS_DIR / "llm_weekly_progress_reasoning.csv"

WEEKLY_EQUITY_PATH = RESULTS_DIR / "llm_weekly_test_equity_curve.csv"
WEEKLY_ACTIONS_PATH = RESULTS_DIR / "llm_weekly_test_actions.csv"
WEEKLY_REASONING_PATH = RESULTS_DIR / "llm_weekly_reasoning_log.csv"

# Existing dashboard/metrics filenames are published only after a complete run.
FINAL_EQUITY_PATH = RESULTS_DIR / "llm_test_equity_curve.csv"
FINAL_ACTIONS_PATH = RESULTS_DIR / "llm_test_actions.csv"
FINAL_REASONING_PATH = RESULTS_DIR / "llm_reasoning_log.csv"


class LLMCallFailed(RuntimeError):
    """An API failure that must pause the run before the date is processed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def step_environment(environment: StockTradingEnv, action: np.ndarray):
    output = environment.step(np.asarray(action, dtype=np.float32).copy())
    if len(output) == 5:
        observation, reward, terminated, truncated, info = output
        done = bool(terminated or truncated)
    else:
        observation, reward, done, info = output
        done = bool(done)
    return observation, reward, done, info

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

def load_test_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {DATA_PATH}")

    test_df = pd.read_csv(DATA_PATH)
    required = {
        "date",
        "tic",
        "close",
        "macd",
        "rsi_30",
        "cci_30",
        "dx_30",
        "turbulence",
        *INDICATORS,
    }
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

    # FinRL expects each date to share one integer index value.
    test_df.index = test_df["date"].factorize()[0]
    return test_df, tickers, dates

def build_market_prompt(
    date: str,
    stock_data: pd.DataFrame,
    tickers: list[str],
    cash: float,
    holdings: dict[str, int],
) -> str:
    positions = ",".join(
        f"{ticker}:{holdings[ticker]}"
        for ticker in tickers
        if holdings[ticker] != 0
    ) or "none"

    lines = [
        f"Date:{date}",
        "Rebalance cadence: once every 5 trading days; today is a decision day.",
        f"Cash:{cash:.2f}",
        f"Shares:{positions}",
        "Data columns: ticker,close,MACD,RSI30,CCI30,ADX30",
    ]
    indexed = stock_data.set_index("tic")
    for ticker in tickers:
        row = indexed.loc[ticker]
        lines.append(
            f"{ticker},{float(row['close']):.2f},{float(row['macd']):.3f},"
            f"{float(row['rsi_30']):.1f},{float(row['cci_30']):.1f},"
            f"{float(row['dx_30']):.1f}"
        )

    lines.extend(
        [
            f"Turbulence:{float(stock_data['turbulence'].iloc[0]):.2f}",
            "Choose one normalized action per ticker from -1.0 (sell) to +1.0 "
            "(buy); 0.0 means hold. Actions will be held unchanged by taking no "
            "further trades for the next four trading transitions.",
            "Return one JSON object only: "
            '{"rationale":"maximum 20 words","actions":{"AAPL":0.0}}. '
            "Include every ticker exactly once and put no prose outside JSON.",
        ]
    )
    return "\n".join(lines)

def parse_model_response(raw: str, tickers: list[str]) -> tuple[np.ndarray, str]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMCallFailed(f"Model returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), dict):
        raise LLMCallFailed("Model response is missing the required actions object")

    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise LLMCallFailed("Model response is missing the required rationale")

    actions_dict = payload["actions"]
    missing = [ticker for ticker in tickers if ticker not in actions_dict]
    extra = [ticker for ticker in actions_dict if ticker not in tickers]
    if missing or extra:
        raise LLMCallFailed(
            f"Ticker mismatch. Missing={missing or 'none'}; extra={extra or 'none'}"
        )

    values: list[float] = []
    for ticker in tickers:
        value = actions_dict[ticker]
        if isinstance(value, bool):
            raise LLMCallFailed(f"Boolean action is invalid for {ticker}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise LLMCallFailed(f"Non-numeric action for {ticker}: {value!r}") from exc
        if not math.isfinite(numeric):
            raise LLMCallFailed(f"Non-finite action for {ticker}: {numeric}")
        values.append(float(np.clip(numeric, -1.0, 1.0)))

    return np.asarray(values, dtype=np.float32), rationale

def get_llm_action(
    client: Groq,
    date: str,
    stock_data: pd.DataFrame,
    tickers: list[str],
    cash: float,
    holdings: dict[str, int],
) -> tuple[np.ndarray, str, str, float, int, int, int]:
    prompt = build_market_prompt(date, stock_data, tickers, cash, holdings)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative portfolio agent. Return valid JSON "
                        "only, without markdown or analysis outside the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "portfolio_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "actions": {
                                "type": "object",
                                "properties": {
                                    ticker: {"type": "number"}
                                    for ticker in tickers
                                },
                                "required": tickers,
                                "additionalProperties": False,
                            },
                        },
                        "required": ["rationale", "actions"],
                        "additionalProperties": False,
                    },
                },
            },
            reasoning_effort="low",
            reasoning_format="hidden",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.1,
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        message = str(exc)
        if status_code == 429 or "429" in message:
            raise LLMCallFailed(
                "Groq rate limit reached. This date was not processed. "
                f"Resume after the quota resets. Details: {message}"
            ) from exc
        raise LLMCallFailed(f"Groq request failed: {message}") from exc

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise LLMCallFailed("Groq returned an empty response")

    actions, rationale = parse_model_response(raw, tickers)
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or prompt_tokens + completion_tokens
    )
    estimated_cost = (
        prompt_tokens * INPUT_USD_PER_MILLION / 1_000_000
        + completion_tokens * OUTPUT_USD_PER_MILLION / 1_000_000
    )
    return (
        actions,
        rationale,
        raw,
        float(estimated_cost),
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )

def new_state(tickers: list[str], dates: list[str]) -> dict:
    total_action_steps = len(dates) - 1
    total_decisions = (total_action_steps + CADENCE - 1) // CADENCE
    return {
        "schema_version": 1,
        "status": "in_progress",
        "model": MODEL,
        "cadence_trading_days": CADENCE,
        "tickers": tickers,
        "test_start": dates[0],
        "test_end": dates[-1],
        "total_valuation_dates": len(dates),
        "total_action_steps": total_action_steps,
        "total_decisions": total_decisions,
        "next_step": 0,
        "decision_count": 0,
        "total_cost": 0.0,
        "actions_log": [],
        "reasoning_log": [],
        "quota_date_utc": utc_day(),
        "tokens_today": 0,
        "calls_today": 0,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "last_error": None,
    }

def load_state(tickers: list[str], dates: list[str]) -> dict:
    if not STATE_PATH.exists():
        print("No weekly checkpoint found; starting a new weekly-cadence run.")
        return new_state(tickers, dates)

    with STATE_PATH.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    expected = {
        "schema_version": 1,
        "model": MODEL,
        "cadence_trading_days": CADENCE,
        "tickers": tickers,
        "test_start": dates[0],
        "test_end": dates[-1],
        "total_valuation_dates": len(dates),
        "total_action_steps": len(dates) - 1,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(
                f"Weekly checkpoint mismatch for {key}; refusing an unsafe resume"
            )

    next_step = int(state.get("next_step", -1))
    actions_log = state.get("actions_log", [])
    reasoning_log = state.get("reasoning_log", [])
    if next_step < 0 or len(actions_log) != next_step:
        raise RuntimeError("Weekly checkpoint is internally inconsistent")
    if int(state.get("decision_count", -1)) != len(reasoning_log):
        raise RuntimeError("Weekly decision/reasoning counts are inconsistent")
    expected_decisions = sum(
        1 for step in range(next_step) if step % CADENCE == 0
    )
    if int(state["decision_count"]) != expected_decisions:
        raise RuntimeError("Weekly checkpoint decision cadence is inconsistent")

    if state.get("quota_date_utc") != utc_day():
        state["quota_date_utc"] = utc_day()
        state["tokens_today"] = 0
        state["calls_today"] = 0

    print(
        f"Resuming exact weekly state: {next_step}/{len(dates) - 1} transitions; "
        f"{state['decision_count']}/{state['total_decisions']} LLM decisions complete."
    )
    return state

def create_and_replay_environment(
    test_df: pd.DataFrame,
    tickers: list[str],
    state: dict,
) -> tuple[StockTradingEnv, np.ndarray]:
    environment = StockTradingEnv(
        df=test_df,
        **environment_kwargs(len(tickers)),
    )
    reset_output = environment.reset()
    observation = reset_output[0] if isinstance(reset_output, tuple) else reset_output

    for step_number, row in enumerate(state["actions_log"]):
        recorded_step = int(row.get("step", -1))
        if recorded_step != step_number:
            raise RuntimeError("Checkpoint action steps are out of order")
        action = np.asarray(row["actions"], dtype=np.float32)
        if action.shape != (len(tickers),):
            raise RuntimeError(f"Bad checkpoint action shape at step {step_number}")
        observation, _, done, _ = step_environment(environment, action)
        if done and step_number + 1 < int(state["total_action_steps"]):
            raise RuntimeError("Environment ended early while replaying checkpoint")

    return environment, np.asarray(observation, dtype=np.float32)

def portfolio_state(
    environment: StockTradingEnv, tickers: list[str]
) -> tuple[float, dict[str, int]]:
    state_vector = np.asarray(environment.state, dtype=float)
    stock_dimension = len(tickers)
    cash = float(state_vector[0])
    shares = state_vector[1 + stock_dimension : 1 + (2 * stock_dimension)]
    holdings = {
        ticker: int(round(float(shares[position])))
        for position, ticker in enumerate(tickers)
    }
    return cash, holdings

def save_state(state: dict) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json_write(state, STATE_PATH)

def equity_frame(environment: StockTradingEnv) -> pd.DataFrame:
    frame = environment.save_asset_memory().copy()
    if "date" not in frame.columns:
        frame = frame.reset_index()
    return frame[["date", "account_value"]]

def write_progress_files(state: dict, environment: StockTradingEnv) -> None:
    atomic_csv_write(equity_frame(environment), PROGRESS_EQUITY_PATH)
    atomic_csv_write(pd.DataFrame(state["actions_log"]), PROGRESS_ACTIONS_PATH)
    atomic_csv_write(pd.DataFrame(state["reasoning_log"]), PROGRESS_REASONING_PATH)

def publish_completed_results(state: dict, environment: StockTradingEnv) -> None:
    equity = equity_frame(environment)
    actions = pd.DataFrame(state["actions_log"])
    reasoning = pd.DataFrame(state["reasoning_log"])

    atomic_csv_write(equity, WEEKLY_EQUITY_PATH)
    atomic_csv_write(actions, WEEKLY_ACTIONS_PATH)
    atomic_csv_write(reasoning, WEEKLY_REASONING_PATH)

    atomic_csv_write(equity, FINAL_EQUITY_PATH)
    atomic_csv_write(actions, FINAL_ACTIONS_PATH)
    atomic_csv_write(reasoning, FINAL_REASONING_PATH)

    # Synchronize legacy checkpoint CSVs only after all 97 decisions succeed.
    atomic_csv_write(equity, RESULTS_DIR / "llm_checkpoint.csv")
    atomic_csv_write(actions, RESULTS_DIR / "llm_actions_checkpoint.csv")
    atomic_csv_write(reasoning, RESULTS_DIR / "llm_reasoning_checkpoint.csv")

def show_status(tickers: list[str], dates: list[str]) -> None:
    total_steps = len(dates) - 1
    total_decisions = (total_steps + CADENCE - 1) // CADENCE
    if not STATE_PATH.exists():
        print("No weekly checkpoint exists.")
        print(f"Market transitions completed: 0/{total_steps}")
        print(f"LLM decisions completed: 0/{total_decisions}")
        print(f"Next decision date: {dates[0]}")
        return

    with STATE_PATH.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    next_step = int(state.get("next_step", 0))
    decisions = int(state.get("decision_count", 0))
    next_decision_step = (
        next_step
        if next_step % CADENCE == 0
        else next_step + (CADENCE - next_step % CADENCE)
    )
    print(f"Status: {state.get('status', 'unknown')}")
    print(f"Model: {state.get('model', 'unknown')}")
    print(f"Tickers: {len(tickers)}")
    print(f"Cadence: every {CADENCE} trading days")
    print(f"Market transitions completed: {next_step}/{total_steps}")
    print(f"Daily valuation dates available: {min(next_step + 1, len(dates))}/{len(dates)}")
    print(f"LLM decisions completed: {decisions}/{total_decisions}")
    print(f"LLM decisions remaining: {total_decisions - decisions}")
    print(
        "Last valued date: "
        + (dates[min(next_step, len(dates) - 1)] if dates else "none")
    )
    print(
        "Next decision date: "
        + (
            dates[next_decision_step]
            if next_decision_step < total_steps
            else "none"
        )
    )
    print(f"Calls recorded today by this run (UTC): {state.get('calls_today', 0)}")
    print(f"Tokens recorded today by this run (UTC): {state.get('tokens_today', 0)}")
    if state.get("last_error"):
        print(f"Last pause reason: {state['last_error']}")

def run_backtest(
    test_df: pd.DataFrame,
    tickers: list[str],
    dates: list[str],
    max_calls: int,
) -> int:
    state = load_state(tickers, dates)
    environment, _ = create_and_replay_environment(test_df, tickers, state)
    total_steps = len(dates) - 1

    if int(state["next_step"]) >= total_steps:
        state["status"] = "completed"
        save_state(state)
        publish_completed_results(state, environment)
        print("The weekly LLM backtest is already complete.")
        return 0

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=api_key)

    calls_this_process = 0
    last_call_started: float | None = None

    while int(state["next_step"]) < total_steps:
        if state["quota_date_utc"] != utc_day():
            state["quota_date_utc"] = utc_day()
            state["tokens_today"] = 0
            state["calls_today"] = 0

        step_index = int(state["next_step"])
        date = dates[step_index]
        decision_day = step_index % CADENCE == 0
        api_cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if decision_day:
            if calls_this_process >= max_calls:
                print(
                    f"Planned pause after {calls_this_process} successful calls in "
                    "this process."
                )
                state["status"] = "paused"
                save_state(state)
                write_progress_files(state, environment)
                return 0

            if int(state["tokens_today"]) >= DAILY_TOKEN_SAFETY_BUDGET:
                print(
                    f"Planned quota pause at {state['tokens_today']:,} locally "
                    "recorded tokens today. Resume after the quota resets."
                )
                state["status"] = "paused"
                save_state(state)
                write_progress_files(state, environment)
                return 0

            if last_call_started is not None:
                remaining_wait = MIN_SECONDS_BETWEEN_CALLS - (
                    time.monotonic() - last_call_started
                )
                if remaining_wait > 0:
                    time.sleep(remaining_wait)

            day_data = test_df.loc[step_index].copy()
            if isinstance(day_data, pd.Series):
                day_data = day_data.to_frame().T
            available = set(day_data["tic"].astype(str))
            missing = [ticker for ticker in tickers if ticker not in available]
            if missing:
                raise RuntimeError(
                    f"Missing market data on {date}: {', '.join(missing)}"
                )

            cash, holdings = portfolio_state(environment, tickers)
            last_call_started = time.monotonic()
            try:
                (
                    actions,
                    rationale,
                    raw_response,
                    api_cost,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                ) = get_llm_action(
                    client, date, day_data, tickers, cash, holdings
                )
            except LLMCallFailed as exc:
                state["last_error"] = str(exc)
                state["status"] = "paused"
                save_state(state)
                write_progress_files(state, environment)
                print(f"\nPAUSED safely on {date}: {exc}")
                print(
                    "This transition was not processed and no zero-action fallback "
                    "was recorded."
                )
                return 2

            state["decision_count"] = int(state["decision_count"]) + 1
            state["total_cost"] = float(state["total_cost"] + api_cost)
            state["tokens_today"] = int(state["tokens_today"] + total_tokens)
            state["calls_today"] = int(state["calls_today"] + 1)
            state["reasoning_log"].append(
                {
                    "date": date,
                    "step": step_index,
                    "decision_number": int(state["decision_count"]),
                    "reasoning": raw_response,
                    "rationale": rationale,
                    "model": MODEL,
                    "status": "success",
                }
            )
            action_status = "llm_decision"
            calls_this_process += 1
        else:
            actions = np.zeros(len(tickers), dtype=np.float32)
            action_status = "scheduled_hold"

        state["actions_log"].append(
            {
                "date": date,
                "step": step_index,
                "decision_day": bool(decision_day),
                "actions": actions.tolist(),
                "api_cost": float(api_cost),
                "cumulative_cost": float(state["total_cost"]),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "model": MODEL if decision_day else "intentional_hold",
                "status": action_status,
            }
        )

        _, _, done, _ = step_environment(environment, actions)
        state["next_step"] = step_index + 1
        state["last_error"] = None
        state["status"] = "in_progress"
        save_state(state)

        if decision_day:
            latest_value = float(environment.asset_memory[-1])
            print(
                f"Decision {state['decision_count']}/{state['total_decisions']} | "
                f"transition {state['next_step']}/{total_steps} | {date} | "
                f"Portfolio ${latest_value:,.2f} | "
                f"Today {state['tokens_today']:,} tokens"
            )

        if done and int(state["next_step"]) < total_steps:
            raise RuntimeError("FinRL environment ended before the test period")
        if int(state["next_step"]) % 25 == 0:
            write_progress_files(state, environment)

    if len(environment.asset_memory) != len(dates):
        raise RuntimeError(
            f"Expected {len(dates)} equity values, got {len(environment.asset_memory)}"
        )
    if int(state["decision_count"]) != int(state["total_decisions"]):
        raise RuntimeError("Run ended without the expected number of LLM decisions")

    state["status"] = "completed"
    save_state(state)
    write_progress_files(state, environment)
    publish_completed_results(state, environment)
    print(
        "\nWeekly LLM backtest complete: "
        f"{state['total_decisions']} decisions and {len(dates)} daily valuations."
    )
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quota-safe 5-trading-day GPT-OSS backtest"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show weekly-run progress without making an API call",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS_PER_PROCESS,
        help="Maximum successful API calls in this process (default: 110)",
    )
    args = parser.parse_args()
    if args.max_calls < 1:
        parser.error("--max-calls must be at least 1")
    return args

def main() -> int:
    load_dotenv(ROOT / ".env")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    test_df, tickers, dates = load_test_data()
    print(
        f"Loaded {len(tickers)} tickers across {len(dates)} valuation dates; "
        f"weekly cadence requires {(len(dates) - 1 + CADENCE - 1) // CADENCE} "
        "LLM decisions."
    )

    if args.status:
        show_status(tickers, dates)
        return 0
    return run_backtest(test_df, tickers, dates, args.max_calls)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. The last completed transition remains checkpointed.")
        sys.exit(130)
