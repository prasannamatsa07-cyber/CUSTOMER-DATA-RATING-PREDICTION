"""
train_model.py
==============
Trains ML models on the e-commerce customer behaviour dataset and saves the
best pipeline to models/model.pkl.

Target  : Customer_Rating  (1–5)  → multi-class classification
Features: Age, Gender, City, Product_Category, Unit_Price, Quantity,
          Discount_Amount, Total_Amount, Payment_Method, Device_Type,
          Session_Duration_Minutes, Pages_Viewed, Is_Returning_Customer,
          Delivery_Time_Days

Run with:
    python train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ─────────────────────────────────────────────
# 2. Load data
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 3. Basic cleaning
# ─────────────────────────────────────────────
# Drop identifier / date columns that carry no predictive signal
DROP_COLS = ["Order_ID", "Customer_ID", "Date"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Normalise boolean column
if df["Is_Returning_Customer"].dtype == object:
    df["Is_Returning_Customer"] = df["Is_Returning_Customer"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    )

TARGET = "Customer_Rating"
FEATURE_COLS = [c for c in df.columns if c != TARGET]

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

print(f"\n  Target : {TARGET}")
print(f"  Classes: {sorted(y.unique())}")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ─────────────────────────────────────────────
# 4. Identify column types
# ─────────────────────────────────────────────
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\n  Numeric  : {NUM_COLS}")
print(f"  Categorical: {CAT_COLS}")

# ─────────────────────────────────────────────
# 5. Pre-processing sub-pipelines
# ─────────────────────────────────────────────
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, NUM_COLS),
    ("cat", cat_transformer, CAT_COLS),
], remainder="drop")

# ─────────────────────────────────────────────
# 6. Train / test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 7. Model candidates
# ─────────────────────────────────────────────
CANDIDATES = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    ),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

# ─────────────────────────────────────────────
# 8. Train & compare
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training and evaluating models …\n")

results = {}
best_name, best_score, best_pipeline = None, -1, None

for name, clf in CANDIDATES.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    # 5-fold cross-val on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                scoring="accuracy", n_jobs=-1)
    cv_mean = cv_scores.mean()

    # Fit on full training set
    pipeline.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))

    results[name] = {"cv_mean": cv_mean, "test_acc": test_acc}
    print(f"  {name:25s}  CV={cv_mean:.4f}  Test={test_acc:.4f}")

    if test_acc > best_score:
        best_score    = test_acc
        best_name     = name
        best_pipeline = pipeline

# ─────────────────────────────────────────────
# 9. Detailed report for best model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Best model: {best_name}  (Test accuracy = {best_score:.4f})")
print("\nClassification Report:")
y_pred = best_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 10. Save artefacts
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

artefact = {
    "pipeline":     best_pipeline,
    "feature_cols": FEATURE_COLS,
    "num_cols":     NUM_COLS,
    "cat_cols":     CAT_COLS,
    "target":       TARGET,
    "best_model":   best_name,
    "test_accuracy": best_score,
    "results":      results,
    # Unique category values needed for Streamlit dropdowns
    "cat_uniques":  {col: sorted(df[col].dropna().unique().tolist())
                     for col in CAT_COLS},
    # Numeric range hints for Streamlit sliders
    "num_ranges":   {col: {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                    } for col in NUM_COLS},
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(artefact, f)

print(f"\nModel artefact saved -> {MODEL_PATH}")
print("=" * 60)
print("Done.")
