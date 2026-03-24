import time
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from src.pipelining import build_pipeline

results_df = pd.DataFrame()
experiment_counter = 0


def experiment_logger(func):

    def wrapper(*args, **kwargs):
        global results_df
        global experiment_counter

        start = time.time()

        # Run training function
        pipe, X_train, X_val, y_train, y_val, pipeline_params = func(*args, **kwargs)

        fit_time = time.time() - start

        # Predictions
        y_train_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)

        y_train_prob = pipe.predict_proba(X_train)[:, 1]
        y_val_prob = pipe.predict_proba(X_val)[:, 1]

        pos_label_train = "yes" if "yes" in y_train.values else 1
        pos_label_val = "yes" if "yes" in y_val.values else 1

        train_f1 = f1_score(y_train, y_train_pred, pos_label=pos_label_train)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=pos_label_val)

        train_precision = precision_score(
            y_train, y_train_pred, pos_label=pos_label_train
        )
        val_precision = precision_score(y_val, y_val_pred, pos_label=pos_label_val)

        train_recall = recall_score(y_train, y_train_pred, pos_label=pos_label_train)
        val_recall = recall_score(y_val, y_val_pred, pos_label=pos_label_val)

        train_auc = roc_auc_score(y_train, y_train_prob)
        val_auc = roc_auc_score(y_val, y_val_prob)

        model = pipe.named_steps["classifier"]

        # ------------------------
        # model hyperparameters
        # ------------------------
        default_model = model.__class__()
        params = model.get_params()
        default_params = default_model.get_params()

        model_params = {
            f"model__{k}": v
            for k, v in params.items()
            if k in default_params and v != default_params[k]
        }

        # ------------------------
        # pipeline params
        # ------------------------
        pipe_params = {
            f"pipe__{k}": v for k, v in pipeline_params.items() if k != "model"
        }

        # keep features count
        n_features = len(pipe.named_steps["preprocessing"].get_feature_names_out())

        row = {
            "experiment_id": experiment_counter,
            "model_name": model.__class__.__name__,
            "fit_time": fit_time,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "train_auroc": train_auc,
            "val_auroc": val_auc,
            "train_recall": train_recall,
            "val_recall": val_recall,
            "train_precision": train_precision,
            "val_precision": val_precision,
            "n_features": n_features,
        }

        row.update(pipe_params)
        row.update(model_params)

        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        experiment_counter += 1

        return pipe

    return wrapper


@experiment_logger
def train_pipeline(X_train, X_val, y_train, y_val, pipeline_params):

    pipe = build_pipeline(**pipeline_params)

    pipe.fit(X_train, y_train)

    return pipe, X_train, X_val, y_train, y_val, pipeline_params


def reset_experiments():
    global results_df, experiment_counter
    results_df = pd.DataFrame()
    experiment_counter = 0


def show_results_df(
    model_name=None,  # default None will show all models
    sort_by="val_f1",
    show_cols=[
        "model_name",
        "fit_time",
        "train_f1",
        "val_f1",
        "train_auroc",
        "val_auroc",
    ],
    show_count=10,
    max_experiment_id=np.inf,  # show experiments up to this ID
    min_experiment_id=-1,
):
    # Show full column content without truncation
    with pd.option_context("display.max_colwidth", None):
        df = results_df[
            (results_df["experiment_id"] <= max_experiment_id)
            & (results_df["experiment_id"] >= min_experiment_id)
        ]

        # Filter by model_name only if provided
        if model_name is not None:
            df = df[df["model_name"] == model_name]

        display(df.sort_values(sort_by, ascending=False)[show_cols].head(show_count))
