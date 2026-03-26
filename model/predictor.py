"""
predictor.py
------------
Loads model.pkl and exposes a clean `predict_colleges()` function
that app.py calls at request time.

Author: Sami Noor Saifi
"""

import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ── Load artifact once at import time ─────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    _artifact = pickle.load(f)

_model       = _artifact["model"]
_scaler      = _artifact["scaler"]
_le_state    = _artifact["le_state"]
_le_type     = _artifact["le_type"]
_le_gender   = _artifact["le_gender"]
_feature_cols= _artifact["feature_cols"]
_df          = _artifact["df"]


def _safe_label_encode(encoder, value, fallback=0):
    """Return encoded label or fallback if unseen."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return fallback


def predict_colleges(
    budget: float,
    preferred_state: str,
    college_type: str,          # "Public/Government" | "Private" | "Any"
    gender: str,                # "Co-Ed" | "Boys" | "Girls"
    min_rating: float = 0.0,
    required_facilities: list = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Returns top_n college recommendations as a list of dicts.

    Strategy
    --------
    1. Hard-filter the dataframe on user constraints.
    2. For each surviving row build a feature vector identical to training.
    3. Use model.predict_proba to rank by confidence, then sort by rating.
    4. Return rich metadata for the UI.
    """
    required_facilities = required_facilities or []
    df = _df.copy()

    # ── 1. Hard filters ───────────────────────────────────────────────────────
    df = df[df["Average Fees"] <= budget]
    df = df[df["Rating"] >= min_rating]

    if college_type != "Any":
        df = df[df["College Type"] == college_type]

    if preferred_state != "Any":
        df = df[df["State"] == preferred_state]

    if gender != "Any":
        df = df[df["Genders Accepted"] == gender]

    for fac in required_facilities:
        df = df[df["Facilities"].str.contains(fac, na=False)]

    if df.empty:
        return []

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    key_facilities = ["Gym", "Sports", "Wi-Fi", "Labs", "Cafeteria",
                      "Boys Hostel", "Girls Hostel", "Library"]

    rows = []
    for _, row in df.iterrows():
        state_enc  = _safe_label_encode(_le_state,  row["State"])
        type_enc   = _safe_label_encode(_le_type,   row["College Type"])
        gender_enc = _safe_label_encode(_le_gender, row["Genders Accepted"])

        fac_flags = [
            int(fac in str(row["Facilities"])) for fac in key_facilities
        ]

        feat = [
            row["Rating"],
            row["Average Fees"],
            row["Campus_Size_Acres"],
            row["Total Faculty"],
            state_enc,
            type_enc,
            gender_enc,
        ] + fac_flags

        rows.append(feat)

    X = _scaler.transform(np.array(rows))

    # ── 3. Score & rank ───────────────────────────────────────────────────────
    proba      = _model.predict_proba(X)          # (n_samples, n_classes)
    max_conf   = proba.max(axis=1)                 # best-class confidence

    df = df.copy()
    df["_confidence"] = max_conf

    df_ranked = df.sort_values(
        ["_confidence", "Rating"], ascending=[False, False]
    ).head(top_n)

    # ── 4. Build output dicts ─────────────────────────────────────────────────
    results = []
    for _, row in df_ranked.iterrows():
        courses_list = [c.strip() for c in str(row["Courses"]).split(";")]
        facilities_list = [f.strip() for f in str(row["Facilities"]).split(";")]

        results.append({
            "name":         row["College Name"],
            "state":        row["State"],
            "city":         row["City"],
            "type":         row["College Type"],
            "rating":       round(float(row["Rating"]), 2),
            "fees":         int(row["Average Fees"]),
            "campus_size":  row["Campus Size"],
            "faculty":      int(row["Total Faculty"]),
            "courses":      courses_list[:6],      # cap for UI
            "facilities":   facilities_list,
            "gender":       row["Genders Accepted"],
            "established":  int(row["Established Year"]) if pd.notna(row.get("Established Year")) else "N/A",
            "match_score":  round(float(row["_confidence"]) * 100, 1),
        })

    return results


def get_all_states() -> list[str]:
    return sorted(_df["State"].dropna().unique().tolist())


def get_all_facilities() -> list[str]:
    all_fac = set()
    for f in _df["Facilities"].dropna():
        for item in f.split(";"):
            all_fac.add(item.strip())
    return sorted(all_fac)


def get_stats() -> dict:
    return {
        "total_colleges": len(_df),
        "states":         _df["State"].nunique(),
        "avg_rating":     round(_df["Rating"].mean(), 2),
        "avg_fees":       int(_df["Average Fees"].mean()),
    }
