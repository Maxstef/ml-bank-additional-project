import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_model_and_subset(
    pipeline, X, subset_indices=None, sample_size=100, random_state=42
):
    """
    Extract model and transformed subset of data from a pipeline.

    Returns:
    - model
    - transformed subset as DataFrame (with feature names)
    - transformed full DataFrame (with feature names)
    - subset indices of provided X dataset
    """

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessing"]

    if subset_indices is None:
        sample_size = min(sample_size, len(X))
        subset_indices = np.random.default_rng(random_state).choice(
            X.index, size=sample_size, replace=False
        )

    X_subset_raw = X.loc[subset_indices]

    # transform subset
    X_subset_transformed = pipeline[:-1].transform(X_subset_raw)
    feature_names = preprocessor.get_feature_names_out()

    # transform full
    X_full_transformed = pipeline[:-1].transform(X)

    X_subset_transformed_df = pd.DataFrame(
        X_subset_transformed, columns=feature_names, index=subset_indices
    ).astype(np.float64)
    X_full_transformed_df = pd.DataFrame(X_full_transformed, columns=feature_names)

    return model, X_subset_transformed_df, X_full_transformed_df, subset_indices


def log_odds_to_proba(log_odds):
    """
    Convert log-odds (logits) to probability.

    Parameters:
    - log_odds: float or array-like

    Returns:
    - probability in range [0, 1]
    """
    return 1 / (1 + np.exp(-log_odds))


def get_mean_shap(shap_values, class_idx=None):
    if class_idx is not None:
        values = shap_values.values[:, :, class_idx]
    else:
        values = shap_values.values

    return np.abs(values).mean(axis=0), shap_values.feature_names


def prepare_top_with_other(vals, names, max_display):
    idx = np.argsort(vals)

    # Top (max_display - 1)
    top_idx = idx[-(max_display - 1) :]

    top_vals = vals[top_idx]
    top_names = np.array(names)[top_idx]

    # Remaining ("Other")
    other_val = vals[idx[: -(max_display - 1)]].sum()

    # Combine
    final_vals = np.insert(top_vals, 0, other_val)
    final_names = np.insert(top_names, 0, "Other features")

    return final_vals, final_names


def sort_features(vals, names):
    idx = np.argsort(vals)
    return vals[idx], np.array(names)[idx]


def plot_shap_comparison(shap_rf, shap_xgb, shap_lr, max_display=15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- RF ---
    rf_vals, rf_names = get_mean_shap(shap_rf, class_idx=1)
    rf_vals, rf_names = prepare_top_with_other(rf_vals, rf_names, max_display)

    # print(np.array(rf_names)[rf_idx], rf_vals[rf_idx])

    # axes[0].barh(np.array(rf_names)[rf_idx], rf_vals[rf_idx])
    axes[0].barh(rf_names, rf_vals)
    axes[0].set_title("Random Forest (SHAP)")

    # --- XGB ---
    xgb_vals, xgb_names = get_mean_shap(shap_xgb, class_idx=1)
    xgb_vals, xgb_names = prepare_top_with_other(xgb_vals, xgb_names, max_display)
    # xgb_idx = np.argsort(xgb_vals)[-max_display:]

    # print(np.array(xgb_names)[xgb_idx], xgb_vals[xgb_idx])

    # axes[1].barh(np.array(xgb_names)[xgb_idx], xgb_vals[xgb_idx])
    axes[1].barh(xgb_names, xgb_vals)
    axes[1].set_title("XGBoost (SHAP)")

    # --- LR ---
    lr_vals, lr_names = get_mean_shap(shap_lr)
    lr_vals, lr_names = prepare_top_with_other(lr_vals, lr_names, max_display)
    # lr_idx = np.argsort(lr_vals)[-max_display:]

    # print(np.array(lr_names)[lr_idx], lr_vals[lr_idx])

    # axes[2].barh(np.array(lr_names)[lr_idx], lr_vals[lr_idx])
    axes[2].barh(lr_names, lr_vals)
    axes[2].set_title("Logistic Regression (SHAP)")

    plt.tight_layout()
    plt.show()


def get_shap_per_confusion_matrix(y_true, y_pred, X_subset, shap_values):
    # --- Align predictions ---
    y_pred = pd.Series(y_pred, index=y_true.index)

    # --- Confusion groups ---
    tp_idx = y_true[(y_true == "yes") & (y_pred == "yes")].index
    fp_idx = y_true[(y_true == "no") & (y_pred == "yes")].index
    tn_idx = y_true[(y_true == "no") & (y_pred == "no")].index
    fn_idx = y_true[(y_true == "yes") & (y_pred == "no")].index

    # --- Map index → position ---
    index_to_pos = {idx: pos for pos, idx in enumerate(X_subset.index)}

    def to_pos(indices):
        return [index_to_pos[i] for i in indices]

    tp_pos = to_pos(tp_idx)
    fp_pos = to_pos(fp_idx)
    tn_pos = to_pos(tn_idx)
    fn_pos = to_pos(fn_idx)

    # --- Handle SHAP shape ---
    def extract_shap(pos, class_idx=1):
        if len(shap_values.values.shape) == 3:
            return shap_values.values[pos, :, class_idx]
        else:
            return shap_values.values[pos]

    def extract_base(pos, class_idx=1):
        if len(np.array(shap_values.base_values).shape) == 2:
            return shap_values.base_values[pos, class_idx]
        else:
            return shap_values.base_values[pos]

    # --- Extract ---
    result = {
        "tp": {"pos": tp_pos},
        "fp": {"pos": fp_pos},
        "tn": {"pos": tn_pos},
        "fn": {"pos": fn_pos},
    }

    for key in result:
        pos = result[key]["pos"]
        result[key]["shap"] = extract_shap(pos)
        result[key]["base"] = extract_base(pos)

    # --- Debug counts ---
    # print({k: len(v["pos"]) for k, v in result.items()})

    return result


def get_common_indexes(y_true, preds_dict, true_value="yes", pred_value="yes"):
    """
    Find indices where ALL models agree on a given condition.

    preds_dict: dict like {
        "rf": y_pred_rf,
        "xgb": y_pred_xgb,
        "lr": y_pred_lr
    }
    """

    index_sets = []

    for model_name, y_pred in preds_dict.items():
        y_pred_series = pd.Series(y_pred, index=y_true.index)

        idx = set(y_true[(y_true == true_value) & (y_pred_series == pred_value)].index)

        index_sets.append(idx)

    # intersection across all models
    common_idx = set.intersection(*index_sets)

    return list(common_idx)


def get_single_explanation(shap_values, pos, class_idx=1):
    if len(shap_values.values.shape) == 3:
        return shap_values[pos, :, class_idx]
    else:
        return shap_values[pos]


def plot_confusion_matrices(
    y_true,
    predictions_dict,
    normalize=None,
    labels=["no", "yes"],
    figsize=(15, 8),
    max_cols=3,
):
    """
    Plot confusion matrices for multiple models.

    Parameters
    ----------
    y_true : array-like
        True labels.

    predictions_dict : dict
        Dictionary where:
        key = model name (str)
        value = predicted labels (array-like)

        Example:
        {
            "rf": y_pred_rf,
            "xgb": y_pred_xgb,
            "lr": y_pred_lr
        }

    normalize : str or None
        Normalization mode passed to sklearn.confusion_matrix.
        Options: None, 'true', 'pred', 'all'

    labels : list
        Class labels order.

    figsize : tuple
        Figure size.
    """
    n_models = len(predictions_dict)

    n_cols = min(max_cols, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Make axes iterable and flat
    axes = axes.flatten() if n_models > 1 else [axes]

    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[i]

        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format=".2f" if normalize else "d", colorbar=False)

        ax.set_title(f"{model_name} ({'normalized' if normalize else 'raw'})")

    # Hide unused subplots
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
