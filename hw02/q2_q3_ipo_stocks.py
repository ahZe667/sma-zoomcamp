import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

URL = "https://www.iposcoop.com/2025-pricings/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OFFER_CUTOFF = "2025-09-01"
ANALYSIS_DAY = "2026-09-11"
RISK_FREE = 0.05
CACHE = Path(__file__).parent / "stocks_df.parquet"


def load_ipos() -> pd.DataFrame:
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    ipos = pd.read_html(io.StringIO(resp.text))[0]
    print("IPOs in 2025:", len(ipos))
    ipos["Offer Date"] = pd.to_datetime(ipos["Offer Date"], format="%m/%d/%Y")
    ipos = ipos[ipos["Offer Date"] < OFFER_CUTOFF]
    ipos = ipos[ipos["Return"].str.strip() != "0.00%"]
    print(f"IPOs before {OFFER_CUTOFF} with non-zero return:", len(ipos))
    return ipos


def download(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = yf.download(
            ticker,
            start="2025-01-01",
            end="2026-09-12",
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            print("no data:", ticker)
            continue
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Ticker"] = ticker
        df["growth_252d"] = df["Close"] / df["Close"].shift(252)
        df["volatility"] = df["Close"].rolling(30).std() * np.sqrt(252)
        df["Sharpe"] = (df["growth_252d"] - RISK_FREE) / df["volatility"]
        for m in range(1, 13):
            df[f"future_growth_{m}_m"] = df["Close"].shift(-21 * m) / df["Close"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if CACHE.exists():
        stocks_df = pd.read_parquet(CACHE)
    else:
        ipos = load_ipos()
        stocks_df = download(ipos["Symbol"].str.strip().tolist())
        stocks_df.to_parquet(CACHE)
    print("tickers with data:", stocks_df["Ticker"].nunique())

    day = stocks_df[stocks_df["Date"] == ANALYSIS_DAY]
    print(f"\nQ2: stats on {ANALYSIS_DAY}")
    print(day[["growth_252d", "volatility", "Sharpe"]].describe().to_string())

    min_dates = stocks_df.groupby("Ticker")["Date"].min().reset_index()
    first_days = stocks_df.merge(min_dates, on=["Ticker", "Date"], how="inner")
    cols = [f"future_growth_{m}_m" for m in range(1, 13)]
    stats = first_days[cols].describe()
    print("\nQ3: growth from first trading day close")
    print(stats.T.to_string())
    print("\nbest median:", stats.loc["50%"].idxmax(), round(stats.loc["50%"].max(), 4))


if __name__ == "__main__":
    main()
