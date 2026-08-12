# Project configuration
TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2020-07-01"
TEST_START_DATE = "2020-07-01"
TEST_END_DATE = "2022-05-31"

INITIAL_AMOUNT = 1_000_000
TRANSACTION_COST = 0.001  # 0.1% per trade

DJIA_TICKERS = [
    "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
    "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE",
    "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW"
]

INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]
