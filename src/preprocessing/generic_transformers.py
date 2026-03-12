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

        # infer dtype from rule values
        values = [value for _, value in self.rules]
        self.dtype_ = np.array(values).dtype

        return self

    def transform(self, X):
        X = X.copy()

        result = np.empty(len(X), dtype=self.dtype_)

        for condition, value in self.rules:
            mask = condition(X[self.column])
            result[mask] = value

        X[self.new_column] = result

        return X


class MappingTransformer(BaseEstimator, TransformerMixin):
    """
    Map values from an existing column to a new column using a dictionary.

    This transformer applies a simple dictionary mapping to the specified
    column and stores the result in a new column. It is useful for converting
    categorical values into numeric representations or grouping categories.

    Parameters
    ----------
    column : str
        Name of the input column to map.

    mapping : dict
        Dictionary defining the mapping from original values to new values.
        Keys correspond to values in the input column and values correspond
        to the mapped output values.

    new_column : str
        Name of the column to create with the mapped values.

    Attributes
    ----------
    column_ : str
        Validated input column name after fitting.

    Examples
    --------
    >>> mapping = {"jan": 1, "feb": 2, "mar": 3}
    >>> transformer = MappingTransformer(
    ...     column="month",
    ...     mapping=mapping,
    ...     new_column="month_num"
    ... )
    >>> transformer.fit_transform(df)

    Notes
    -----
    Values not present in the mapping dictionary will be mapped to NaN,
    consistent with pandas `Series.map()` behavior.
    """

    def __init__(self, column, mapping, new_column):
        self.column = column
        self.mapping = mapping
        self.new_column = new_column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"{self.column} not found in DataFrame")

        self.column_ = self.column
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_column] = X[self.column].map(self.mapping)
        return X


class NumericBinner(BaseEstimator, TransformerMixin):
    """
    Transforms a numeric column into categorical bins based on specified thresholds.
    """

    def __init__(
        self, column: str, bins: list, labels: list = None, new_column: str = None
    ):
        """
        Parameters
        ----------
        column : str
            Column to transform.

        bins : list of numbers
            Bin edges (including min and max).

        labels : list of str, optional
            Labels for bins. Must be one less than len(bins).

        new_column : str, optional
            Name of the new column to create. If not provided,
            the default name will be "{column}_group".
        """
        self.column = column
        self.bins = bins
        self.labels = labels
        self.new_column = new_column

    def fit(self, X, y=None):

        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        if self.bins is None:
            raise ValueError("Bins must be provided")

        if self.labels is not None and len(self.labels) != len(self.bins) - 1:
            raise ValueError("Number of labels must be one less than number of bins")

        self.bins_ = self.bins
        self.labels_ = self.labels

        # determine output column name
        self.new_column_ = self.new_column or f"{self.column}_group"

        return self

    def transform(self, X):
        X = X.copy()

        X[self.new_column_] = pd.cut(
            X[self.column], bins=self.bins_, labels=self.labels_, include_lowest=True
        )

        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode cyclical features using sine and cosine transformation.
    Examples: month, day_of_week, hour, etc.
    """

    def __init__(self, column, max_value=None, drop_original=False):
        """
        Parameters
        ----------
        column : str
            Name of the column to encode
        max_value : int, optional
            Maximum possible value of the cycle (e.g., 12 for month, 7 for day_of_week).
            If None, will use X[column].max()
        drop_original : bool, default False
            Whether to drop the original column after encoding
        """
        self.column = column
        self.max_value = max_value
        self.drop_original = drop_original

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
        X[self.column] = pd.to_numeric(X[self.column], errors="coerce").fillna(0)
        # Scale column to 0 -> 2pi
        radians = 2 * np.pi * X[self.column] / self.max_value_
        X[f"{self.column}_sin"] = np.sin(radians)
        X[f"{self.column}_cos"] = np.cos(radians)

        if self.drop_original:
            X = X.drop(columns=[self.column])

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Return the names of the transformed features (sin and cos)
        """
        return [f"{self.column}_sin", f"{self.column}_cos"]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode categorical columns with options to:
      - Drop specified categories
      - Combine multiple categories into one
    """

    def __init__(self, columns, drop_categories=None, combine_categories=None):
        """
        Parameters
        ----------
        columns : list
            List of categorical columns to encode
        drop_categories : dict, optional
            Dictionary {column_name: [categories_to_drop]}
            Example: {'job': ['unknown'], 'education': ['illiterate']}
        combine_categories : dict, optional
            Dictionary {column_name: {new_category: [old_values]}}
            Example: {'education': {'other': ['unknown', 'illiterate']}}
        """
        self.columns = columns
        self.drop_categories = drop_categories or {}
        self.combine_categories = combine_categories or {}
        self.categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            values = X[col].unique().tolist()

            # Drop unwanted categories
            if col in self.drop_categories:
                values = [v for v in values if v not in self.drop_categories[col]]

            # Combine categories: replace old values with new merged key
            if col in self.combine_categories:
                for new_cat, old_vals in self.combine_categories[col].items():
                    for old_val in old_vals:
                        if old_val in values:
                            values.remove(old_val)
                    values.append(new_cat)

            self.categories_[col] = sorted(values)

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Apply category combination first
            if col in self.combine_categories:
                for new_cat, old_vals in self.combine_categories[col].items():
                    X[col] = X[col].replace(old_vals, new_cat)

            # Drop unwanted categories
            if col in self.drop_categories:
                X[col] = X[col].where(~X[col].isin(self.drop_categories[col]))

            # One-hot encode
            allowed_values = self.categories_[col]
            for val in allowed_values:
                X[f"{col}_{val}"] = (X[col] == val).astype(int)

            # Drop original column
            X = X.drop(columns=[col])

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names after one-hot encoding
        """
        feature_names = []
        for col in self.columns:
            if col in self.categories_:
                feature_names.extend([f"{col}_{val}" for val in self.categories_[col]])
        return np.array(feature_names)


class AutoCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop_categories=None, combine_categories=None):
        self.drop_categories = drop_categories or {}
        self.combine_categories = combine_categories or {}

    def fit(self, X, y=None):
        # Select all object columns at this stage
        self.columns_ = X.select_dtypes(include="object").columns.tolist()

        # Initialize the actual encoder
        self.encoder_ = CategoricalEncoder(
            columns=self.columns_,
            drop_categories=self.drop_categories,
            combine_categories=self.combine_categories,
        )
        self.encoder_.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder_.transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Return the list of feature names produced after encoding.
        Works with sklearn's ColumnTransformer convention.
        """
        # Collect feature names from the encoder
        if hasattr(self.encoder_, "get_feature_names_out"):
            return self.encoder_.get_feature_names_out(input_features)

        # If CategoricalEncoder doesn’t implement get_feature_names_out,
        # fallback: return names for each one-hot column as "<col>_<category>"
        feature_names = []
        for col in self.columns_:
            categories = self.encoder_.categories_[
                col
            ]  # assumed stored inside CategoricalEncoder
            for cat in categories:
                feature_names.append(f"{col}_{cat}")
        return feature_names
