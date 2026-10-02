from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

FILE_ID = "1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5"
DATA = Path(__file__).parent / "stocks_df_combined_2026_09_18.parquet.brotli"
TARGET = "is_positive_growth_30d_future"

OHLCV = ["Open", "High", "Low", "Close", "Adj Close_x", "Volume"]
CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]
CUSTOM_NUMERICAL = [
    "SMA10",
    "SMA20",
    "growing_moving_average",
    "high_minus_low_relative",
    "volatility",
    "ln_volume",
]
TECHNICAL_INDICATORS = [
    "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc", "bop", "cci", "cmo",
    "dx", "macd", "macdsignal", "macdhist", "macd_ext", "macdsignal_ext",
    "macdhist_ext", "macd_fix", "macdsignal_fix", "macdhist_fix", "mfi",
    "minus_di", "mom", "plus_di", "dm", "ppo", "roc", "rocp", "rocr", "rocr100",
    "rsi", "slowk", "slowd", "fastk", "fastd", "fastk_rsi", "fastd_rsi", "trix",
    "ultosc", "willr", "ad", "adosc", "obv", "atr", "natr", "ht_dcperiod",
    "ht_dcphase", "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine",
    "ht_sine_leadsine", "ht_trendmod", "avgprice", "medprice", "typprice",
    "wclprice",
]  # fmt: skip
MACRO = [
    "gdppot_us_yoy",
    "gdppot_us_qoq",
    "cpi_core_yoy",
    "cpi_core_mom",
    "FEDFUNDS",
    "DGS1",
    "DGS5",
    "DGS10",
]
HAND_RULES = ["pred0", "pred1", "pred2", "pred3", "pred4"]


def temporal_split(dates: pd.Series, train_prop=0.7, val_prop=0.15) -> pd.Series:
    min_date, max_date = dates.min(), dates.max()
    days = (max_date - min_date).days
    train_end = min_date + pd.Timedelta(days=days * train_prop)
    val_end = train_end + pd.Timedelta(days=days * val_prop)
    return pd.Series(
        np.select(
            [dates <= train_end, dates <= val_end], ["train", "validation"], "test"
        ),
        index=dates.index,
    )


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def test_precision(df: pd.DataFrame, column: str) -> float:
    test = df[df.split == "test"]
    return precision_score(test[TARGET], test[column], zero_division=0)


def prepare() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Return the modelling frame with dummies, split and hand-rule columns."""
    if not DATA.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}", str(DATA), quiet=False
        )
    df_full = pd.read_parquet(DATA)
    print("df_full:", df_full.shape)

    growth = [g for g in df_full if g.startswith("growth_") and "future" not in g]
    patterns = [g for g in df_full if "cdl" in g]
    to_predict = [g for g in df_full if "future" in g]
    numerical = growth + TECHNICAL_INDICATORS + patterns + CUSTOM_NUMERICAL + MACRO

    df_full["ln_volume"] = np.log(df_full.Volume.replace(0, np.nan).fillna(1e-9))
    df = df_full[df_full.Date >= "2000-01-01"].copy()
    df["Weekday"] = df["Weekday"].astype(str)
    df["Month"] = df["Month"].dt.month.astype(str)
    week_of_month = (df.Date.dt.day - 1) // 7 + 1
    df["month_wom"] = df.Date.dt.month_name() + "_w" + week_of_month.astype(str)

    dummies = pd.get_dummies(df[CATEGORICAL], dtype="int32")
    dummy_names = dummies.columns.to_list()
    df = pd.concat([df, dummies], axis=1)

    df["split"] = temporal_split(df.Date)
    df["pred0_manual_cci"] = (df.cci > 200).astype(int)
    df["pred1_manual_prev_g1"] = (df.growth_30d > 1).astype(int)
    df["pred2_manual_prev_g1_and_snp"] = (
        (df.growth_30d > 1) & (df.growth_snp500_30d > 1)
    ).astype(int)
    df["pred3_manual_dgs10_5"] = ((df.DGS10 <= 4.5) & (df.DGS5 <= 4)).astype(int)
    df["pred4_manual_dgs10_fedfunds"] = (
        (df.DGS10 > 4) & (df.FEDFUNDS <= 4.795)
    ).astype(int)
    for pred in [c for c in df if c.startswith("pred")]:
        df[f"is_correct_{pred.split('_')[0]}"] = (df[pred] == df[TARGET]).astype(int)
    return df, numerical, dummy_names, to_predict


def main() -> None:
    df, numerical, dummy_names, to_predict = prepare()
    wom_dummies = [d for d in dummy_names if d.startswith("month_wom_")]
    print(f"dummies: {len(dummy_names)} (month_wom: {len(wom_dummies)})")

    # Q1
    corr = df[numerical + dummy_names + to_predict].corr()[TARGET]
    corr_wom = corr[wom_dummies].to_frame()
    corr_wom["abs_corr"] = corr_wom[TARGET].abs()
    corr_wom = corr_wom.sort_values("abs_corr", ascending=False)
    print("\nQ1: top month_wom correlations\n", corr_wom.head(5))
    print("Q1 answer:", round(corr_wom.abs_corr.iloc[0], 3))

    # Q2
    print("\nsplit counts\n", df.groupby("split").Date.agg(["min", "max", "count"]))
    predictions = [c for c in df if c.startswith("pred")]
    print("\nQ2: precision on TEST (positives predicted / precision)")
    test = df[df.split == "test"]
    for pred in predictions:
        print(f"  {pred:32s} {test[pred].sum():6d}  {test_precision(df, pred):.4f}")

    # Q3
    features = numerical + dummy_names
    train_mask = df.split.isin(["train", "validation"])
    X_all = clean(df[features])
    y_all = clean(df[[TARGET]])[TARGET]
    X_train, y_train = X_all[train_mask], y_all[train_mask]

    clf_10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf_10.fit(X_train, y_train)
    df["pred5_clf_10"] = clf_10.predict(X_all)
    df["is_correct_pred5"] = (df.pred5_clf_10 == df[TARGET]).astype(int)
    print(f"\npred5_clf_10 TEST precision: {test_precision(df, 'pred5_clf_10'):.4f}")

    others_wrong = (df[[f"is_correct_{p}" for p in HAND_RULES]] == 0).all(axis=1)
    df["only_pred5_is_correct"] = ((df.is_correct_pred5 == 1) & others_wrong).astype(
        int
    )
    print("Q3 answer:", df[df.split == "test"].only_pred5_is_correct.sum())

    # Q4
    print("\nQ4: depth -> TEST precision")
    scores = {}
    for depth in range(1, 13):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_all[df.split == "test"])
        scores[depth] = precision_score(
            y_all[df.split == "test"], y_pred, zero_division=0
        )
        print(f"  {depth:2d}  {scores[depth]:.4f}  positives={y_pred.sum()}")
    best_max_depth = max(scores, key=lambda depth: scores[depth])
    print("Q4 answer (best_max_depth):", best_max_depth)

    clf_best = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    clf_best.fit(X_train, y_train)
    df["pred6_clf_best"] = clf_best.predict(X_all)
    print(export_text(clf_best, feature_names=features, max_depth=3))
    print("TEST precision, all predictions:")
    for pred in [c for c in df if c.startswith("pred")]:
        print(f"  {pred:32s} {test_precision(df, pred):.4f}")


if __name__ == "__main__":
    main()
