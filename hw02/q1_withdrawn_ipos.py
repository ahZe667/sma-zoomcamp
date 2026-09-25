import io
import re

import pandas as pd
import requests

URL = "https://www.iposcoop.com/ipos-recently-filed/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
CUTOFF = "2026-09-11"

RULES = [
    (r"\bTechnologies\b", "Technologies"),
    (r"\bAcquisition Corp(oration)?\b|\bCorp\b", "Acquisition Corp"),
    (r"\bInc\b|\bIncorporated\b", "Inc."),
    (r"\bGroup\b", "Group"),
    (r"\bLtd\b|\bLimited\b", "Limited"),
    (r"\bHoldings?\b", "Holdings"),
]


def company_type(name: str) -> str:
    for pattern, label in RULES:
        if re.search(pattern, name):
            return label
    return "Other"


def to_number(value) -> float | None:
    if pd.isna(value):
        return None
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def main() -> None:
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    print(df.columns.tolist(), len(df))
    withdrawn = df[
        df["Expected To Trade"].astype(str).str.strip() == "Withdrawn"
    ].copy()
    print("withdrawn rows (all):", len(withdrawn))
    withdrawn = withdrawn[pd.to_datetime(withdrawn["File Date"]) < CUTOFF]
    print(f"withdrawn rows filed before {CUTOFF}:", len(withdrawn))
    withdrawn["Company Type"] = withdrawn["Company"].astype(str).map(company_type)
    low = withdrawn["Price Low"].map(to_number)
    high = withdrawn["Price High"].map(to_number)
    withdrawn["Avg_price"] = (low.astype(float) + high.astype(float)) / 2
    withdrawn["Shares"] = withdrawn["Shares (millions)"].map(to_number).astype(float)
    withdrawn["Est_vol"] = (
        withdrawn["Est $ Vol (millions)"].map(to_number).astype(float)
    )
    value = withdrawn["Shares"] * withdrawn["Avg_price"]
    withdrawn["Shares_offered_value"] = value.where(value.notna(), withdrawn["Est_vol"])
    print(
        withdrawn[
            [
                "Company",
                "Company Type",
                "Avg_price",
                "Shares",
                "Est_vol",
                "Shares_offered_value",
            ]
        ].to_string()
    )
    print(
        withdrawn.groupby("Company Type")["Shares_offered_value"]
        .sum()
        .sort_values(ascending=False)
    )


if __name__ == "__main__":
    main()
