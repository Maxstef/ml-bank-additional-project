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
        "Raw dataset not found. Expected:\n"
        f"- {csv_path}\n"
        f"- {zip_path}"
    )