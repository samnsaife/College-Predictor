"""
train_model.py
--------------
Trains a Random Forest model to predict/recommend colleges
based on student inputs. Run this once to generate model.pkl.

Author: Sami Noor Saifi
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "colleges.csv")
MODEL_OUT = os.path.join(BASE_DIR, "model.pkl")


def load_and_preprocess(path: str):
    """Load CSV and engineer features used for prediction."""
    df = pd.read_csv(path)

    # ── Derived numeric features ──────────────────────────────────────────────
    # Campus size: strip non-numeric, default 0
    df["Campus_Size_Acres"] = (
        df["Campus Size"]
        .str.extract(r"([\d.]+)", expand=False)
        .astype(float)
        .fillna(0)
    )

    df["Rating"]       = pd.to_numeric(df["Rating"],       errors="coerce").fillna(0)
    df["Average Fees"] = pd.to_numeric(df["Average Fees"], errors="coerce").fillna(0)
    df["Total Faculty"]= pd.to_numeric(df["Total Faculty"],errors="coerce").fillna(0)

    # ── Facility flags ────────────────────────────────────────────────────────
    key_facilities = ["Gym", "Sports", "Wi-Fi", "Labs", "Cafeteria",
                      "Boys Hostel", "Girls Hostel", "Library"]
    for fac in key_facilities:
        col = fac.replace(" ", "_").replace("-", "_")
        df[f"fac_{col}"] = df["Facilities"].str.contains(fac, na=False).astype(int)

    # ── Encode categoricals ───────────────────────────────────────────────────
    le_state  = LabelEncoder()
    le_type   = LabelEncoder()
    le_gender = LabelEncoder()

    df["State_enc"]   = le_state.fit_transform(df["State"].fillna("Unknown"))
    df["Type_enc"]    = le_type.fit_transform(df["College Type"].fillna("Unknown"))
    df["Gender_enc"]  = le_gender.fit_transform(df["Genders Accepted"].fillna("Co-Ed"))

    # ── Feature matrix ────────────────────────────────────────────────────────
    fac_cols = [f"fac_{f.replace(' ','_').replace('-','_')}" for f in key_facilities]
    feature_cols = (
        ["Rating", "Average Fees", "Campus_Size_Acres", "Total Faculty",
         "State_enc", "Type_enc", "Gender_enc"]
        + fac_cols
    )

    X = df[feature_cols].values
    y = df["College Name"].values          # target = college name

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df, scaler, le_state, le_type, le_gender, feature_cols


def train():
    print("Loading data …")
    X, y, df, scaler, le_state, le_type, le_gender, feature_cols = (
        load_and_preprocess(DATA_PATH)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, preds, zero_division=0))

    # ── Persist everything needed at inference time ───────────────────────────
    artifact = {
        "model":        clf,
        "scaler":       scaler,
        "le_state":     le_state,
        "le_type":      le_type,
        "le_gender":    le_gender,
        "feature_cols": feature_cols,
        "df":           df,           # kept for metadata look-ups
    }

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    train()
