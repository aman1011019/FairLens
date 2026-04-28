"""
preprocessing.py
────────────────
Handles loading and preprocessing of the Adult Income Dataset (or any
compatible CSV) into the (X, A, y) format required by the FairLens pipeline.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional


# ─── Column constants for the UCI Adult dataset ───────────────────────────────
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

ADULT_URL_TRAIN = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
ADULT_URL_TEST = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)


def load_adult_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the UCI Adult Income dataset.

    Args:
        filepath: Optional path to a local CSV. If None, attempts to load
                  from UCI repository (requires internet).

    Returns:
        Raw DataFrame with stripped whitespace in string columns.
    """
    if filepath:
        df = pd.read_csv(
            filepath,
            header=None,
            names=ADULT_COLUMNS,
            na_values=" ?",
            skipinitialspace=True,
        )
    else:
        df = pd.read_csv(
            ADULT_URL_TRAIN,
            header=None,
            names=ADULT_COLUMNS,
            na_values=" ?",
            skipinitialspace=True,
        )

    # Strip leading/trailing whitespace in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def preprocess(
    df: pd.DataFrame,
    sensitive_attr: str = "gender",
    target_col: str = "income",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline.

    Steps:
        1. Drop rows with missing values.
        2. Encode binary target → {0, 1}.
        3. Encode sensitive attribute → {0, 1}.
        4. One-hot encode remaining categoricals.
        5. Standard-scale numeric features.
        6. Train/test split.

    Args:
        df:              Raw DataFrame (output of load_adult_dataset).
        sensitive_attr:  Column name of the protected attribute (e.g. "gender").
        target_col:      Column name of the label (e.g. "income").
        test_size:       Fraction for test split.
        random_state:    Seed for reproducibility.

    Returns:
        A dict with keys:
            X_train, X_test  – feature matrices (numpy arrays)
            y_train, y_test  – label vectors
            A_train, A_test  – sensitive-attribute vectors {0, 1}
            feature_names    – list of feature column names
            label_encoder    – fitted LabelEncoder for sensitive_attr
            scaler           – fitted StandardScaler
    """
    df = df.copy().dropna()

    # ── 1. Encode target ──────────────────────────────────────────────────────
    # Adult dataset uses ">50K" / "<=50K" (sometimes with trailing dot in test)
    df[target_col] = df[target_col].str.replace(".", "", regex=False)
    df["label"] = (df[target_col] == ">50K").astype(int)

    # ── 2. Encode sensitive attribute ─────────────────────────────────────────
    le_sensitive = LabelEncoder()
    df["sensitive"] = le_sensitive.fit_transform(df[sensitive_attr])
    # Store mapping for interpretability
    sensitive_map = dict(
        zip(le_sensitive.classes_, le_sensitive.transform(le_sensitive.classes_))
    )

    # ── 3. Drop original target + sensitive column ────────────────────────────
    feature_df = df.drop(columns=[target_col, "label", sensitive_attr, "sensitive"])

    # ── 4. One-hot encode categoricals ────────────────────────────────────────
    feature_df = pd.get_dummies(feature_df, drop_first=True)

    # ── 5. Standard scale ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    feature_names = list(feature_df.columns)

    y = df["label"].values
    A = df["sensitive"].values

    # ── 6. Split ───────────────────────────────────────────────────────────────
    (X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
        X, y, A, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "A_train": A_train,
        "A_test": A_test,
        "feature_names": feature_names,
        "label_encoder": le_sensitive,
        "sensitive_map": sensitive_map,
        "scaler": scaler,
    }
