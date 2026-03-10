from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CampaignPrevTransformer(BaseEstimator, TransformerMixin):
    """
    Creates campaign_prev = min(campaign - 1, cap)
    """

    def __init__(self, column="campaign", cap=5, new_column="campaign_prev"):
        self.column = column
        self.cap = cap
        self.new_column = new_column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")
        return self

    def transform(self, X):
        X = X.copy()

        prev_contacts = X[self.column] - 1
        prev_contacts = np.maximum(prev_contacts, 0)
        prev_contacts = np.minimum(prev_contacts, self.cap)

        X[self.new_column] = prev_contacts

        return X


class PdaysTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms pdays into either:
    - binary contacted indicator
    - categorical recency groups
    """

    def __init__(self, column="pdays", mode="binary", recent_days=14):
        self.column = column
        self.mode = mode
        self.recent_days = recent_days

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")
        return self

    def transform(self, X):
        X = X.copy()

        if self.mode == "binary":
            X["pdays_contacted"] = (X[self.column] != 999).astype(int)

        elif self.mode == "group":

            conditions = [
                X[self.column] == 999,
                (X[self.column] < self.recent_days) & (X[self.column] != 999),
                X[self.column] >= self.recent_days,
            ]

            labels = [
                "not_contacted_before",
                "contacted_recently",
                "contacted_long_ago",
            ]

            X["pdays_group"] = np.select(
                conditions, labels, default="not_contacted_before"
            )

        else:
            raise ValueError("mode must be 'binary' or 'group'")

        return X
