from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Removes specified columns from a DataFrame.
    """

    def __init__(self, columns: list):
        """
        Parameters
        ----------
        columns : list
            List of column names to remove.
        """
        self.columns = columns

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise ValueError("ColumnDropper expects a pandas DataFrame")

        # Validate columns during fit for safer pipelines
        self.columns_ = [col for col in self.columns if col in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        return [c for c in input_features if c not in self.columns_]


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps values in a specified column.
    If `cap` is not provided, calculates it using the IQR method (Q3 + 1.5*IQR by default).
    """

    def __init__(self, column: str, cap: float = None, iqr_multiplier: float = 1.5):
        """
        Parameters
        ----------
        column : str
            Column name to cap.
        cap : float, optional
            Maximum allowed value. If None, calculated using IQR method.
        iqr_multiplier : float, default=1.5
            Multiplier for IQR to determine cap if cap is None.
        """
        self.column = column
        self.cap = cap
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        # Calculate cap if not provided
        if self.cap is None:
            Q1 = X[self.column].quantile(0.25)
            Q3 = X[self.column].quantile(0.75)
            IQR = Q3 - Q1
            self.cap_ = Q3 + self.iqr_multiplier * IQR
        else:
            self.cap_ = self.cap

        return self

    def transform(self, X):
        X = X.copy()
        # Cap values
        X[self.column] = np.minimum(X[self.column], self.cap_)
        # Ensure no negative values
        X[self.column] = np.maximum(X[self.column], 0)
        return X


class ColumnArithmetic(BaseEstimator, TransformerMixin):
    """
    Applies an arithmetic operation (add, subtract, multiply, divide) to a column.
    """

    def __init__(self, column: str, operation: str = "subtract", value: float = 0):
        """
        Parameters
        ----------
        column : str
            Column to apply the operation.
        operation : str, default="subtract"
            Operation to apply: "add", "subtract", "multiply", "divide".
        value : float, default=0
            Value to use in the operation.
        """
        self.column = column
        self.operation = operation
        self.value = value

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")
        return self

    def transform(self, X):
        X = X.copy()
        if self.operation == "subtract":
            X[self.column] = X[self.column] - self.value
        elif self.operation == "add":
            X[self.column] = X[self.column] + self.value
        elif self.operation == "multiply":
            X[self.column] = X[self.column] * self.value
        elif self.operation == "divide":
            X[self.column] = X[self.column] / self.value
        else:
            raise ValueError(f"Unsupported operation '{self.operation}'")

        # Ensure no negative values after operation (optional, can remove if not needed)
        X[self.column] = np.maximum(X[self.column], 0)
        return X


class ConditionalMapper(BaseEstimator, TransformerMixin):
    """
    Applies rule-based mapping to create a new column.

    rules format:
    [
        (condition_function, value),
        ...
    ]
    """

    def __init__(self, column, rules, new_column):
        self.column = column
        self.rules = rules
        self.new_column = new_column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"{self.column} not found in DataFrame")
        return self

    def transform(self, X):
        X = X.copy()

        result = np.full(len(X), None)

        for condition, value in self.rules:
            mask = condition(X[self.column])
            result[mask] = value

        X[self.new_column] = result

        return X


class NumericBinner(BaseEstimator, TransformerMixin):
    """
    Transforms a numeric column into categorical bins based on specified thresholds.
    """

    def __init__(self, column: str, bins: list, labels: list = None):
        """
        Parameters
        ----------
        column : str
            Column to transform
        bins : list of numbers, optional
            Bin edges (including min and max)
        labels : list of str, optional
            Labels for bins. Must be one less than len(bins)
        """
        self.column = column
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):

        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        # Validate bins and labels
        if self.bins is None:
            raise ValueError("Bins must be provided")

        if self.labels is not None and len(self.labels) != len(self.bins) - 1:
            raise ValueError("Number of labels must be one less than number of bins")

        # store validated parameters
        self.bins_ = self.bins
        self.labels_ = self.labels

        return self

    def transform(self, X):
        X = X.copy()

        X[f"{self.column}_group"] = pd.cut(
            X[self.column], bins=self.bins_, labels=self.labels_, include_lowest=True
        )

        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode cyclical features using sine and cosine transformation.
    Examples: month, day_of_week, hour, etc.
    """

    def __init__(self, column, max_value=None):
        """
        Parameters
        ----------
        column : str
            Name of the column to encode
        max_value : int, optional
            Maximum possible value of the cycle (e.g., 12 for month, 7 for day_of_week).
            If None, will use X[column].max()
        """
        self.column = column
        self.max_value = max_value

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")
        if self.max_value is None:
            self.max_value_ = X[self.column].max()
        else:
            self.max_value_ = self.max_value
        return self

    def transform(self, X):
        X = X.copy()
        # Scale column to 0 -> 2pi
        radians = 2 * np.pi * X[self.column] / self.max_value_
        X[f"{self.column}_sin"] = np.sin(radians)
        X[f"{self.column}_cos"] = np.cos(radians)
        return X

