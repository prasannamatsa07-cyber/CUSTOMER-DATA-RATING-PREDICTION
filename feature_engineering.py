"""
feature_engineering.py
======================
Shared feature-engineering logic used by both train_model.py and app.py.
Call `engineer_features(df)` on any DataFrame that already has the raw columns.

New columns produced
--------------------
Discount_Pct        : discount as % of (total + discount) – buying behaviour signal
Price_Tier          : 0=budget / 1=mid / 2=premium based on Unit_Price tertiles
Engagement_Score    : normalised combination of Session_Duration & Pages_Viewed
Value_Score         : Total_Amount per page viewed – spend efficiency
High_Discount_Flag  : 1 when Discount_Pct > 20 %
Fast_Delivery_Flag  : 1 when Delivery_Time_Days <= 3
"""

import numpy as np
import pandas as pd


# Price tertile boundaries (fitted on the training set; kept fixed here so
# inference is consistent without needing the training data).
_PRICE_BOUNDARIES = (100.0, 400.0)   # low < 100 | mid 100-400 | premium > 400


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with additional engineered columns appended.

    Works on a copy — the original DataFrame is not mutated.
    Handles single-row DataFrames (inference) and full datasets (training).
    """
    df = df.copy()

    # ── 1. Discount percentage ─────────────────────────────────────────────
    gross = df["Total_Amount"] + df["Discount_Amount"]
    df["Discount_Pct"] = np.where(
        gross > 0,
        (df["Discount_Amount"] / gross) * 100.0,
        0.0,
    ).round(2)

    # ── 2. Price tier (0 = budget, 1 = mid, 2 = premium) ──────────────────
    lo, hi = _PRICE_BOUNDARIES
    df["Price_Tier"] = pd.cut(
        df["Unit_Price"],
        bins=[-np.inf, lo, hi, np.inf],
        labels=[0, 1, 2],
    ).astype(float)

    # ── 3. Engagement score  (0-1 normalised) ─────────────────────────────
    # Simple equal-weight combination; both columns already on similar scales.
    dur_norm   = df["Session_Duration_Minutes"] / (df["Session_Duration_Minutes"].max() + 1e-9)
    pages_norm = df["Pages_Viewed"] / (df["Pages_Viewed"].max() + 1e-9)
    df["Engagement_Score"] = ((dur_norm + pages_norm) / 2.0).round(4)

    # ── 4. Value score  (spend per page) ──────────────────────────────────
    df["Value_Score"] = (
        df["Total_Amount"] / (df["Pages_Viewed"].replace(0, 1))
    ).round(2)

    # ── 5. Binary flag: heavy discount ────────────────────────────────────
    df["High_Discount_Flag"] = (df["Discount_Pct"] > 20).astype(int)

    # ── 6. Binary flag: fast delivery ─────────────────────────────────────
    df["Fast_Delivery_Flag"] = (df["Delivery_Time_Days"] <= 3).astype(int)

    return df


# ── Inference helper ───────────────────────────────────────────────────────
# When doing single-row inference the max() normalisation in Engagement_Score
# would always give 1.0.  We fix that by using known training-set maxima.
_TRAIN_SESSION_MAX = 60.0    # minutes  (safe upper bound from dataset)
_TRAIN_PAGES_MAX   = 20.0    # pages    (safe upper bound from dataset)


def engineer_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Like engineer_features() but uses fixed normalisation constants
    so single-row inference gives meaningful Engagement_Score values."""
    df = df.copy()

    gross = df["Total_Amount"] + df["Discount_Amount"]
    df["Discount_Pct"] = np.where(
        gross > 0,
        (df["Discount_Amount"] / gross) * 100.0,
        0.0,
    ).round(2)

    lo, hi = _PRICE_BOUNDARIES
    df["Price_Tier"] = pd.cut(
        df["Unit_Price"],
        bins=[-np.inf, lo, hi, np.inf],
        labels=[0, 1, 2],
    ).astype(float)

    dur_norm   = df["Session_Duration_Minutes"] / _TRAIN_SESSION_MAX
    pages_norm = df["Pages_Viewed"] / _TRAIN_PAGES_MAX
    df["Engagement_Score"] = ((dur_norm + pages_norm) / 2.0).clip(0, 1).round(4)

    df["Value_Score"] = (
        df["Total_Amount"] / (df["Pages_Viewed"].replace(0, 1))
    ).round(2)

    df["High_Discount_Flag"] = (df["Discount_Pct"] > 20).astype(int)
    df["Fast_Delivery_Flag"] = (df["Delivery_Time_Days"] <= 3).astype(int)

    return df
