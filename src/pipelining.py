import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    ColumnDropper,
    ColumnArithmetic,
    OutlierCapper,
    NumericBinner,
    PdaysTransformer,
    MappingTransformer,
    AutoCategoricalEncoder,
    CyclicalEncoder,
)

from src.preprocessing.mappings import month_map, dow_map

# -------------------------------------------------------
# Socio-economic binning configuration
# -------------------------------------------------------
socioecon_bins = {
    "emp.var.rate": {"bins": [-np.inf, -0.5, np.inf], "labels": ["low", "high"]},
    "nr.employed": {
        "bins": [-np.inf, 5077, 5150, np.inf],
        "labels": ["low", "medium", "high"],
    },
    "euribor3m": {"bins": [-np.inf, 3, np.inf], "labels": ["low", "high"]},
    "cons.price.idx": {
        "bins": [-np.inf, 92.75, 93.5, np.inf],
        "labels": ["low", "medium", "high"],
    },
    "cons.conf.idx": {
        "bins": [-np.inf, -45, -40, -35, np.inf],
        "labels": ["very low", "low", "medium", "high"],
    },
}


# -------------------------------------------------------
# Feature engineering pipeline
# -------------------------------------------------------
def build_feature_engineering_pipeline(
    drop_cols,
    campaign_prev_subtract_value,
    campaign_prev_cap_value,
    pdays_transform_mode,
    pdays_transform_recent_days,
    calendar_cols_mode,
    age_bin_mode,
    soceco_bin_cols,
):

    steps = [
        ("drop_cols_initial", ColumnDropper(columns=drop_cols)),
        (
            "campaign_prev",
            Pipeline(
                [
                    (
                        "subtract_1",
                        ColumnArithmetic(
                            column="campaign",
                            operation="subtract",
                            value=campaign_prev_subtract_value,
                        ),
                    ),
                    (
                        "cap_outliers",
                        OutlierCapper(column="campaign", cap=campaign_prev_cap_value),
                    ),
                ]
            ),
        ),
        (
            "pdays_transform",
            PdaysTransformer(
                mode=pdays_transform_mode, recent_days=pdays_transform_recent_days
            ),
        ),
    ]

    # Calendar numeric mapping
    if calendar_cols_mode != "onehot":
        steps.extend(
            [
                ("month_mapper", MappingTransformer("month", month_map, "month_num")),
                ("dow_mapper", MappingTransformer("day_of_week", dow_map, "dow_num")),
                ("drop_cols_month_day", ColumnDropper(["month", "day_of_week"])),
            ]
        )

    # Age binning
    if age_bin_mode in ("group", "range"):
        bins, labels = (
            ([0, 25, 58, 120], ["young", "middle", "old"])
            if age_bin_mode == "group"
            else (
                [0, 24, 29, 39, 49, 59, 120],
                ["≤24", "25-29", "30-39", "40-49", "50-59", "60+"],
            )
        )
        steps.extend(
            [
                ("age_numbin", NumericBinner(column="age", bins=bins, labels=labels)),
                ("age_drop", ColumnDropper(columns=["age"])),
            ]
        )

    # Socio-economic binning
    if soceco_bin_cols:
        for col in soceco_bin_cols:
            if col in socioecon_bins and col not in drop_cols:
                cfg = socioecon_bins[col]

                steps.append(
                    (
                        f"{col}_numbin",
                        NumericBinner(
                            column=col, bins=cfg["bins"], labels=cfg["labels"]
                        ),
                    ),
                )

        # drop original soc-eco cols after preprocessing
        steps.append(("soceco_drop", ColumnDropper(columns=soceco_bin_cols)))

    return Pipeline(steps)


# -------------------------------------------------------
# Cyclical encoding pipeline
# -------------------------------------------------------
def build_cyclical_pipeline():

    return Pipeline(
        [
            ("month_cyclical", CyclicalEncoder("month_num", 12)),
            ("dow_cyclical", CyclicalEncoder("dow_num", 5)),
            ("drop_month_dow_num", ColumnDropper(["month_num", "dow_num"])),
        ]
    )


# -------------------------------------------------------
# Numeric preprocessing pipeline
# -------------------------------------------------------
def build_numeric_pipeline(scale_mode, poly_degree):

    steps = []

    if scale_mode == "standard":
        steps.append(("scaler", StandardScaler()))

    elif scale_mode == "normalize":
        steps.append(("scaler", MinMaxScaler()))

    if poly_degree > 1:
        steps.append(
            (
                "poly_features",
                PolynomialFeatures(degree=poly_degree, include_bias=False),
            )
        )

    if steps:
        return Pipeline(steps)

    return "passthrough"


# -------------------------------------------------------
# Preprocessor (ColumnTransformer)
# -------------------------------------------------------
def build_preprocessor(cat_encoder, num_encoder):

    return ColumnTransformer(
        [
            (
                "cat",
                cat_encoder,
                make_column_selector(dtype_include=["object", "category"]),
            ),
            ("numeric", num_encoder, make_column_selector(dtype_include="number")),
        ]
    )


# -------------------------------------------------------
# Main pipeline builder
# -------------------------------------------------------
def build_pipeline(
    drop_cols=None,
    campaign_prev_subtract_value=1,
    campaign_prev_cap_value=5,
    pdays_transform_mode="group",
    pdays_transform_recent_days=7,
    soceco_bin_cols=None,
    age_bin_mode=None,
    calendar_cols_mode="onehot",
    drop_cats=None,
    combine_cats=None,
    scale_mode="standard",
    poly_degree=1,
    model=LogisticRegression(random_state=42),
):
    """
    Build a configurable end-to-end machine learning pipeline for preprocessing
    and classification.

    The pipeline consists of several stages:

    1. Feature engineering
       - Column dropping
       - Campaign feature transformation
       - `pdays` transformation
       - Optional calendar feature mapping
       - Optional age binning
       - Optional socio-economic feature binning

    2. Optional cyclical encoding
       - Applies cyclical encoding to `month` and `day_of_week`
       when `calendar_cols_mode="cyclical"`.

    3. Preprocessing
       - Automatic categorical encoding using `AutoCategoricalEncoder`
       - Numeric feature scaling and optional polynomial feature generation
       using a `ColumnTransformer`.

    4. Model
       - Final classification estimator.

    The resulting object is a fully compatible sklearn `Pipeline` that can be
    used with tools such as cross-validation, hyperparameter tuning, and
    model persistence.

    Parameters
    ----------
    drop_cols : list[str] or None, default=None
        Columns to drop during feature engineering.
        If None, defaults to ["duration"].

    campaign_prev_subtract_value : int, default=1
        Value subtracted from the `campaign` column to create a "previous
        campaign contacts" feature.

    campaign_prev_cap_value : int, default=5
        Upper cap applied to the transformed `campaign` feature to limit
        the influence of outliers.

    pdays_transform_mode : {"group", "binary"}, default="group"
        Strategy used to transform the `pdays` feature.

        - "group": creates categorical groups based on recency
        - "binary": indicates whether the client was contacted before

    pdays_transform_recent_days : int, default=7
        Threshold used to define "recent contact" when
        `pdays_transform_mode="group"`.

    soceco_bin_cols : list[str] or None, default=None
        List of socio-economic numeric columns to convert into categorical bins
        using predefined thresholds.

    age_bin_mode : {"group", "range"} or None, default=None
        Optional strategy for binning the `age` column.

        - "group": broad groups (young / middle / old)
        - "range": detailed age ranges
        - None: no age binning applied.

    calendar_cols_mode : {"onehot", "num", "cyclical"}, default="onehot"
        Encoding strategy for calendar features (`month`, `day_of_week`).

        - "onehot": treat them as categorical features
        - "num": map to chronological numeric values
        - "cyclical": apply cyclical encoding (sin/cos)

    drop_cats : dict or None, default=None
        Categories to remove from categorical variables during encoding.

        Example
        -------
        {"job": ["unknown"]}

    combine_cats : dict or None, default=None
        Mapping of categories to combine into a single category.

        Example
        -------
        {"default": {"risk": ["unknown", "yes"]}}

    scale_mode : {"standard", "normalize", None}, default="standard"
        Scaling method applied to numeric features.

        - "standard": standardization using StandardScaler
        - "normalize": min-max scaling using MinMaxScaler
        - None: no scaling applied.

    poly_degree : int, default=1
        Degree of polynomial features generated from numeric variables.
        Values greater than 1 enable polynomial feature expansion.

    model : sklearn estimator, default=LogisticRegression(random_state=42)
        Final classification model used in the pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully constructed preprocessing and modeling pipeline.

    Notes
    -----
    The pipeline is designed to be compatible with hyperparameter search tools
    such as `GridSearchCV`, `RandomizedSearchCV`, and
    optimization frameworks like Optuna.
    """

    # Default params
    if drop_cols is None:
        drop_cols = ["duration"]

    if drop_cats is None:
        drop_cats = {
            "job": ["unknown"],
            "marital": ["unknown"],
            "education": ["unknown", "illiterate"],
            "poutcome": ["nonexistent"],
        }

    if combine_cats is None:
        combine_cats = {"default": {"risk": ["unknown", "yes"]}}

    # --------------------------------------------------
    # Build pipeline components
    # --------------------------------------------------
    feature_engineering = build_feature_engineering_pipeline(
        drop_cols,
        campaign_prev_subtract_value,
        campaign_prev_cap_value,
        pdays_transform_mode,
        pdays_transform_recent_days,
        calendar_cols_mode,
        age_bin_mode,
        soceco_bin_cols,
    )

    cat_encoder = AutoCategoricalEncoder(
        drop_categories=drop_cats, combine_categories=combine_cats
    )

    num_encoder = build_numeric_pipeline(scale_mode, poly_degree)

    preprocessor = build_preprocessor(cat_encoder, num_encoder)

    # --------------------------------------------------
    # Final pipeline
    # --------------------------------------------------
    steps = [("feature_engineering", feature_engineering)]

    if calendar_cols_mode == "cyclical":
        steps.append(("cyclical_encoding", build_cyclical_pipeline()))

    steps.extend([("preprocessing", preprocessor), ("classifier", model)])

    return Pipeline(steps)
