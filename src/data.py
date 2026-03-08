from pathlib import Path
import pandas as pd
import zipfile

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
