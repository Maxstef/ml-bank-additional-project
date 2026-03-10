from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
from typing import Optional, Tuple

from src.config import DATA_RAW


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Bank Marketing dataset.

    Supports both:
    - bank-additional-full.csv
    - bank-additional-full.csv.zip

    Returns
    -------
    pd.DataFrame
        Raw dataset as a pandas DataFrame.
    """

    csv_path = DATA_RAW / "bank-additional-full.csv"
    zip_path = DATA_RAW / "bank-additional-full.csv.zip"

    if csv_path.exists():
        # default separator is ','
        return pd.read_csv(csv_path)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            # open the CSV inside ZIP
            with z.open("bank-additional-full.csv") as f:
                return pd.read_csv(f)

    raise FileNotFoundError(
        "Raw dataset not found. Expected:\n" f"- {csv_path}\n" f"- {zip_path}"
    )


def split_numeric_categorical(df, target_col="y"):
    """
    Split DataFrame into numeric and categorical features, plus target.

    Returns dict with:
        X_numeric, y_numeric, X_categorical, y_categorical, numeric_cols, categorical_cols
    """
    # Numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col not in numeric_cols:  # include target if not there
        numeric_cols.append(target_col)
    numeric_df = df[numeric_cols].copy()

    # Convert target to 0/1
    numeric_df[target_col] = numeric_df[target_col].replace({"yes": 1, "no": 0})

    X_numeric = numeric_df.drop(columns=[target_col])
    y_numeric = numeric_df[target_col]

    # Categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col not in categorical_cols:  # include target if not there
        categorical_cols.append(target_col)
    categorical_df = df[categorical_cols].copy()

    # Convert target to 0/1 as well
    categorical_df[target_col] = categorical_df[target_col].replace({"yes": 1, "no": 0})

    X_categorical = categorical_df.drop(columns=[target_col])
    y_categorical = categorical_df[target_col]

    return {
        "X_numeric": X_numeric,
        "y_numeric": y_numeric,
        "X_categorical": X_categorical,
        "y_categorical": y_categorical,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def split_train_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset to split.
    test_size : float, default=0.2
        Fraction of data to use as validation set.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify_col : str or None, default=None
        Column name to stratify on (useful for classification).

    Returns
    -------
    train_df : pd.DataFrame
        Training set.
    val_df : pd.DataFrame
        Validation set.
    """

    if stratify_col is not None:
        stratify_vals = df[stratify_col]
    else:
        stratify_vals = None

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_vals
    )
    return train_df, val_df
