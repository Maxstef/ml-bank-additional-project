import pandas as pd


def remove_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Numeric column name to check for outliers.
    k : float, default 1.5
        Multiplier for the IQR. Values outside [Q1 - k*IQR, Q3 + k*IQR] are considered outliers.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing outliers in `col` removed.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    filtered_df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return filtered_df


import pandas as pd


def replace_outliers_iqr(
    df: pd.DataFrame, col: str, replace_with: float = None, k: float = 1.5
) -> pd.DataFrame:
    """
    Replace outliers in a numeric column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Numeric column name to check for outliers.
    replace_with : float, optional
        Value to replace outliers with.
        If None, replaces lower outliers with Q1 - k*IQR, upper outliers with Q3 + k*IQR.
    k : float, default 1.5
        Multiplier for the IQR. Values outside [Q1 - k*IQR, Q3 + k*IQR] are considered outliers.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers replaced in the specified column.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")

    df = df.copy()

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    if replace_with is None:
        # Replace with bounds
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    else:
        # Replace outliers with provided value
        df.loc[df[col] < lower_bound, col] = replace_with
        df.loc[df[col] > upper_bound, col] = replace_with

    return df
