import numpy as np

from imblearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

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
# Sampler factory
# -------------------------------------------------------
def get_sampler(sampler_name, random_state=42):
    samplers = {
        "smote": SMOTE(random_state=random_state),
        "adasyn": ADASYN(random_state=random_state),
        "random_over": RandomOverSampler(random_state=random_state),
        "random_under": RandomUnderSampler(random_state=random_state),
        "tomek": TomekLinks(),
        "smote_tomek": SMOTETomek(random_state=random_state),
        None: None,
    }
    return samplers.get(sampler_name)


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
# Feature engineering steps
# -------------------------------------------------------
def build_feature_engineering_steps(
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
            "campaign_subtract",
            ColumnArithmetic(
                column="campaign",
                operation="subtract",
                value=campaign_prev_subtract_value,
            ),
        ),
        ("campaign_cap", OutlierCapper(column="campaign", cap=campaign_prev_cap_value)),
    ]

    if pdays_transform_mode in ["binary", "group"] and "pdays" not in drop_cols:
        steps.extend(
            [
                (
                    "pdays_transform",
                    PdaysTransformer(
                        mode=pdays_transform_mode,
                        recent_days=pdays_transform_recent_days,
                    ),
                ),
                ("pdays_drop", ColumnDropper(columns=["pdays"])),
            ]
        )

    # Calendar numeric mapping
    if calendar_cols_mode != "onehot":
        steps.extend(
            [
                ("month_mapper", MappingTransformer("month", month_map, "month_num")),
                ("dow_mapper", MappingTransformer("day_of_week", dow_map, "dow_num")),
                ("drop_month_day", ColumnDropper(["month", "day_of_week"])),
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
                ("age_bin", NumericBinner(column="age", bins=bins, labels=labels)),
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
                        f"{col}_bin",
                        NumericBinner(
                            column=col,
                            bins=cfg["bins"],
                            labels=cfg["labels"],
                        ),
                    )
                )

        steps.append(("soceco_drop", ColumnDropper(columns=soceco_bin_cols)))

    return steps


# -------------------------------------------------------
# Cyclical steps
# -------------------------------------------------------
def build_cyclical_steps():
    return [
        ("month_cyclical", CyclicalEncoder("month_num", 12)),
        ("dow_cyclical", CyclicalEncoder("dow_num", 5)),
        ("drop_month_dow_num", ColumnDropper(["month_num", "dow_num"])),
    ]


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
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
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
            ("num", num_encoder, make_column_selector(dtype_include="number")),
        ]
    )


# -------------------------------------------------------
# Main pipeline builder
# -------------------------------------------------------
def build_pipeline(
    drop_cols=None,
    campaign_prev_subtract_value=1,
    campaign_prev_cap_value=5,
    pdays_transform_mode=None,
    pdays_transform_recent_days=7,
    soceco_bin_cols=None,
    age_bin_mode=None,
    calendar_cols_mode="onehot",
    drop_cats=None,
    combine_cats=None,
    scale_mode="standard",
    poly_degree=1,
    sampler_name=None,
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

    steps = []

    # flatten feature engineering
    steps.extend(
        build_feature_engineering_steps(
            drop_cols,
            campaign_prev_subtract_value,
            campaign_prev_cap_value,
            pdays_transform_mode,
            pdays_transform_recent_days,
            calendar_cols_mode,
            age_bin_mode,
            soceco_bin_cols,
        )
    )

    if calendar_cols_mode == "cyclical":
        steps.extend(build_cyclical_steps())

    cat_encoder = AutoCategoricalEncoder(
        drop_categories=drop_cats, combine_categories=combine_cats
    )

    num_encoder = build_numeric_pipeline(scale_mode, poly_degree)

    preprocessor = build_preprocessor(cat_encoder, num_encoder)

    steps.append(("preprocessing", preprocessor))

    sampler = get_sampler(sampler_name)
    if sampler is not None:
        steps.append(("resampling", sampler))

    steps.append(("classifier", model))

    return Pipeline(steps)
