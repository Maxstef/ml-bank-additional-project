import numpy as np
import warnings
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.pipelining import build_pipeline

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------
# Generic tuning function
# ---------------------------
def tune_model(objective_fn, space, max_evals=50, random_state=42):
    """
    Run hyperopt tuning for a given objective function and search space.
    Returns the best parameters and trials object.
    """
    trials = Trials()
    best = fmin(
        fn=objective_fn,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
    )
    return best, trials


# ---------------------------
# Random Forest
# ---------------------------
def objective_rf(params, X_train, X_val, y_train, y_val, drop_cols=["duration"]):
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=42,
        n_jobs=-1,
    )
    pipeline = build_pipeline(drop_cols=drop_cols, model=model)
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(y_val, y_val_pred, pos_label="yes")
    val_auroc = roc_auc_score(y_val, y_val_prob)

    return {
        "loss": -val_f1,
        "status": STATUS_OK,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
    }


space_rf = {
    "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
    "max_depth": hp.choice("max_depth", [5, 8, 12, 16, 20, None]),
    "min_samples_split": hp.choice("min_samples_split", [2, 5, 10, 20]),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 5, 10]),
    "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
    "class_weight": hp.choice(
        "class_weight", ["balanced", {"no": 1, "yes": 2}, {"no": 1, "yes": 4}]
    ),
}


# ---------------------------
# AdaBoost
# ---------------------------
def objective_ada(
    params, X_train, X_val, y_train, y_val, drop_cols=["duration", "loan", "housing"]
):
    base_est = DecisionTreeClassifier(
        max_depth=params["base_max_depth"],
        min_samples_leaf=params["base_min_samples_leaf"],
        class_weight=params["base_class_weight"],
        random_state=42,
    )
    model = AdaBoostClassifier(
        estimator=base_est,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        random_state=42,
    )
    pipeline = build_pipeline(drop_cols=drop_cols, model=model)
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(y_val, y_val_pred, pos_label="yes")
    val_auroc = roc_auc_score(y_val, y_val_prob)

    return {
        "loss": -val_f1,
        "status": STATUS_OK,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
    }


space_ada = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300]),
    "learning_rate": hp.loguniform("learning_rate", -2, 0),  # ~0.135-1.0
    "base_max_depth": hp.choice("base_max_depth", [1, 2, 3, 4]),
    "base_min_samples_leaf": hp.choice("base_min_samples_leaf", [1, 2, 5]),
    "base_class_weight": hp.choice(
        "base_class_weight", ["balanced", {"yes": 4, "no": 1}, {"yes": 2, "no": 1}]
    ),
}


# ---------------------------
# XGBoost
# ---------------------------
def objective_xgb(
    params,
    X_train,
    X_val,
    y_train_bin,
    y_val_bin,
    drop_cols=["duration", "loan", "housing", "euribor3m"],
):
    model = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=int(params["min_child_weight"]),
        scale_pos_weight=params["scale_pos_weight"],
        random_state=42,
        eval_metric="logloss",
    )
    pipeline = build_pipeline(drop_cols=drop_cols, model=model)
    pipeline.fit(X_train, y_train_bin)

    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(y_val_bin, y_val_pred)
    val_auroc = roc_auc_score(y_val_bin, y_val_prob)

    return {
        "loss": -val_f1,
        "status": STATUS_OK,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
    }


space_xgb = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 150, 200]),
    "max_depth": hp.choice("max_depth", [3, 4, 5, 6, 8]),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "min_child_weight": hp.choice("min_child_weight", [1, 2, 5, 10]),
    "scale_pos_weight": hp.choice("scale_pos_weight", [1, 4, 8]),
}

# ---------------------------
# LightGBM
# ---------------------------
space_lgb = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300]),
    "max_depth": hp.choice("max_depth", [4, 6, 8, 10, 12, -1]),
    "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127]),
    "min_child_samples": hp.choice("min_child_samples", [5, 10, 20, 50]),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "scale_pos_weight": hp.uniform("scale_pos_weight", 1, 4),
    "class_weight": hp.choice(
        "class_weight", [{"yes": 1, "no": 1}, {"yes": 2, "no": 1}, {"yes": 4, "no": 1}]
    ),
}


def objective_lgb(
    params,
    X_train,
    X_val,
    y_train,
    y_val,
    drop_cols=["duration", "loan", "housing", "euribor3m"],
):
    model = LGBMClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        learning_rate=params["learning_rate"],
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        class_weight=params["class_weight"],
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    pipeline = build_pipeline(drop_cols=drop_cols, model=model)
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(y_val, y_val_pred, pos_label="yes")
    val_auroc = roc_auc_score(y_val, y_val_prob)

    return {
        "loss": -val_f1,
        "status": STATUS_OK,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
    }
