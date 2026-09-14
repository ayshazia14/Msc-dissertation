# Project configuration
TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2020-07-01"
TEST_START_DATE = "2020-07-01"
TEST_END_DATE = "2022-05-31"  # Exclusive download boundary
TEST_LAST_TRADING_DATE = "2022-05-27"

INITIAL_AMOUNT = 1_000_000
TRANSACTION_COST = 0.001  # 0.1% per trade
DECISION_CADENCE = 5

DJIA_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT"
]

INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]
