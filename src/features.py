
import re
import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = [
    "channel_name", "category", "Sub-category", "Customer_City",
    "Product_category", "Agent_name", "Supervisor", "Manager",
    "Tenure Bucket", "Agent Shift"
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-aware features from raw Shopzilla support data."""
    x = pd.DataFrame(index=df.index)
    issue = pd.to_datetime(df["Issue_reported at"], dayfirst=True, errors="coerce")
    responded = pd.to_datetime(df["issue_responded"], dayfirst=True, errors="coerce")
    order = pd.to_datetime(df["order_date_time"], dayfirst=True, errors="coerce")

    x["response_minutes"] = (responded - issue).dt.total_seconds() / 60
    x["order_to_issue_hours"] = (issue - order).dt.total_seconds() / 3600
    x["issue_hour"] = issue.dt.hour
    x["issue_dayofweek"] = issue.dt.dayofweek
    x["issue_is_weekend"] = (issue.dt.dayofweek >= 5).astype(float)
    x["issue_day"] = issue.dt.day
    x["issue_month"] = issue.dt.month
    hour = issue.dt.hour.fillna(0)
    x["issue_hour_sin"] = np.sin(2*np.pi*hour/24)
    x["issue_hour_cos"] = np.cos(2*np.pi*hour/24)

    for c in ["Item_price", "connected_handling_time"]:
        x[c] = pd.to_numeric(df[c], errors="coerce")
        x[c + "_missing"] = x[c].isna().astype(float)

    txt = df["Customer Remarks"].fillna("").astype(str)
    x["remark_missing"] = df["Customer Remarks"].isna().astype(float)
    x["remark_chars"] = txt.str.len().astype(float)
    x["remark_words"] = txt.str.split().str.len().astype(float)
    x["remark_exclaim"] = txt.str.count("!").astype(float)
    x["remark_question"] = txt.str.count(r"\?").astype(float)

    positive = {
        "good","great","excellent","helpful","happy","satisfied","satisfaction",
        "thank","thanks","love","amazing","awesome","quick","fast","perfect",
        "resolved","polite","friendly","best"
    }
    negative = {
        "bad","poor","worst","terrible","awful","unhappy","dissatisfied","angry",
        "slow","late","delay","delayed","problem","issue","wrong","refund",
        "fraud","rude","disappointed","disappointing","not"
    }
    def counts(s):
        words = re.findall(r"[a-zA-Z']+", s.lower())
        return sum(w in positive for w in words), sum(w in negative for w in words)

    lc = txt.map(counts)
    x["remark_pos"] = lc.map(lambda t: t[0]).astype(float)
    x["remark_neg"] = lc.map(lambda t: t[1]).astype(float)

    for c in CATEGORICAL_COLUMNS:
        x[c] = df[c].fillna("__MISSING__").astype(str)
    return x
