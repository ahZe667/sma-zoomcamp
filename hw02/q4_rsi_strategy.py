from pathlib import Path

import gdown
import pandas as pd

FILE_ID = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
DATA = Path(__file__).parent / "data.parquet"
RSI_THRESHOLD = 30


def main() -> None:
    if not DATA.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}", str(DATA), quiet=False
        )
    df = pd.read_parquet(DATA, engine="pyarrow")
    print(df.shape)
    selected_df = df[
        (df["rsi"] < RSI_THRESHOLD)
        & (df["Date"] >= "2000-01-01")
        & (df["Date"] <= "2025-06-01")
    ]
    growth = selected_df["growth_future_30d"]
    print("trades:", len(selected_df))
    print("avg 30d return:", round((growth.mean() - 1) * 100, 2), "%")
    print("win rate:", round((growth > 1).mean() * 100, 2), "%")
    net_income = 1000 * (growth - 1).sum()
    print("net income $:", round(net_income, 2))


if __name__ == "__main__":
    main()
