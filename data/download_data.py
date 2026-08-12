
import pandas as pd
import yfinance as yf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    DJIA_TICKERS, INDICATORS
)

def download_and_process():
    print("Step 1: Downloading DJIA data from Yahoo Finance...")
    dfs = []
    for tic in DJIA_TICKERS:
        try:
            df_tic = yf.download(tic, start=TRAIN_START_DATE, end=TEST_END_DATE, auto_adjust=True, progress=False)
            if df_tic.empty:
                print(f"No data for {tic}, skipping")
                continue
            df_tic.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_tic.columns]
            df_tic["tic"] = tic
            df_tic.index.name = "date"
            df_tic = df_tic.reset_index()
            df_tic["date"] = df_tic["date"].astype(str).str[:10]
            dfs.append(df_tic)
            print(f"  ✓ {tic}: {len(df_tic)} rows")
        except Exception as e:
            print(f"  ✗ {tic}: {e}")

    df = pd.concat(dfs).sort_values(["date", "tic"]).reset_index(drop=True)
    print(f"\nTotal: {len(df)} rows for {df['tic'].nunique()} tickers")
    print(df.head())

    print("\nStep 2: Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=True,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    print(f"Processed data shape: {processed.shape}")

    print("\nStep 3: Splitting into train and test sets...")
    train = data_split(processed, TRAIN_START_DATE, TRAIN_END_DATE)
    test = data_split(processed, TEST_START_DATE, TEST_END_DATE)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    print("\nStep 4: Saving to disk...")
    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv", index=False)
    processed.to_csv("data/full_data.csv", index=False)
    print("Saved: data/train_data.csv, data/test_data.csv, data/full_data.csv")

    return train, test, processed

if __name__ == "__main__":
    train, test, processed = download_and_process()
    print("\nDone! Data pipeline complete.")