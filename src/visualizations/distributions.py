import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def plot_column_distribution(
    df: pd.DataFrame, col: str, bins: int = 30, title: str = None, show_pct: bool = True
):
    """
    Plot the distribution of a column (numeric or categorical) with optional percentage annotations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    col : str
        Column name to plot.
    bins : int, default 30
        Number of bins (used for numeric columns).
    title : str, optional
        Title of the plot. Defaults to "Distribution of {col}".
    show_pct : bool, default True
        Whether to annotate bars/bins with percentage values.
    """

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    data = df[col].dropna()

    if pd.api.types.is_numeric_dtype(data):
        # Numeric column → histogram
        total = len(data)
        ax = sns.histplot(data, bins=bins, kde=False)

        if show_pct:
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2.0,
                        height,
                        f"{height/total:.2%}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_ylabel("Count")
        ax.set_xlabel(col)

    else:
        # Categorical column → countplot
        total = len(data)
        ax = sns.countplot(x=col, data=df)

        if show_pct:
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height / total,
                    f"{height/total:.2%}",
                    ha="center",
                    va="bottom",
                )
        ax.set_ylabel("Count")
        ax.set_xlabel(col)

    if title is None:
        title = f"Distribution of {col}"
    ax.set_title(title)

    plt.show()


def plot_numeric_boxplots_grid(
    df, target_col=None, cols_per_row=2, figsize_per_plot=(6, 3)
):
    """
    Plot numeric columns as boxplots in a grid of subplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric columns.
    target_col : str, optional
        Column to exclude (like target variable).
    cols_per_row : int, default 2
        Number of subplots per row.
    figsize_per_plot : tuple, default (6,3)
        Size of each individual subplot.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if target_col is not None and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    n_cols = len(numeric_cols)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row  # ceiling division

    fig, axes = plt.subplots(
        n_rows,
        cols_per_row,
        figsize=(figsize_per_plot[0] * cols_per_row, figsize_per_plot[1] * n_rows),
    )
    axes = axes.flatten()  # flatten in case of single row

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_histogram(
    df: pd.DataFrame,
    col: str,
    target_col: str = None,
    bins: int = 30,
    title: str = None,
    show_pct: bool = True,
    palette: str = "Set2",
    normalize: bool = False,
):
    """
    Plot a histogram for a numeric column, optionally colored by a target column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column(s).
    col : str
        Numeric column name to plot.
    target_col : str, optional
        Column to color by (categorical/binary). Default None.
    bins : int, default 30
        Number of histogram bins.
    title : str, optional
        Title of the plot. Defaults to "Histogram of {col}".
    show_pct : bool, default True
        Whether to annotate bins with percentage of total observations.
    palette : str, default "Set2"
        Color palette for the target column.
    normalize : bool, default False
        If True, the histogram is normalized to show proportion (density) instead of counts.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    plt.figure(figsize=(8, 5))

    if target_col is None:
        # Simple histogram
        data = df[col].dropna()
        total = len(data)
        ax = sns.histplot(
            data,
            bins=bins,
            kde=False,
            color="skyblue",
            stat="density" if normalize else "count",
        )

        if show_pct:
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    if not normalize:
                        ax.text(
                            p.get_x() + p.get_width() / 2.0,
                            height,
                            f"{height/total:.2%}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

    else:
        # Histogram colored by target
        data = df.dropna(subset=[col, target_col])
        total = len(data)
        ax = sns.histplot(
            data,
            x=col,
            hue=target_col,
            bins=bins,
            palette=palette,
            multiple="stack",
            stat="density" if normalize else "count",
        )

        if show_pct:
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:
                    if not normalize:
                        ax.text(
                            patch.get_x() + patch.get_width() / 2.0,
                            patch.get_y() + height,
                            f"{height/total:.2%}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

    if title is None:
        title = f"Histogram of {col}" + (f" by {target_col}" if target_col else "")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Density" if normalize else "Count")
    plt.show()


def plot_target_rate_by_bin(df, col, target_col, bins=20, title=None):
    """
    Plot percentage of target=1 in each bin of a numeric column.
    """

    data = df[[col, target_col]].dropna().copy()

    # create bins
    data["bin"] = pd.cut(data[col], bins=bins)

    # compute target rate
    rate = data.groupby("bin")[target_col].mean()
    counts = data.groupby("bin")[target_col].count()

    # create rounded labels
    labels = [f"{int(interval.left)}–{int(interval.right)}" for interval in rate.index]

    plt.figure(figsize=(10, 5))
    ax = rate.plot(kind="bar")

    # apply rounded labels
    ax.set_xticklabels(labels, rotation=45)

    for i, v in enumerate(rate):
        plt.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    plt.ylabel("Target rate")
    plt.xlabel(col)
    plt.title(title or f"Target rate by {col} bins")

    plt.xticks(rotation=45)
    plt.show()


def plot_target_distribution_by_value(
    df: pd.DataFrame,
    col: str,
    value,
    target_col: str,
    figsize=(6, 4),
):
    """
    Compare target distribution when `col == value` vs `col != value`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Column to test.
    value : Any
        Value to split the column by.
    target_col : str
        Target column (binary or categorical).
    figsize : tuple
        Figure size.
    """

    cond_true = df[df[col] == value][target_col].value_counts(normalize=True)
    cond_false = df[df[col] != value][target_col].value_counts(normalize=True)

    df_plot = pd.DataFrame(
        {f"{col} = {value}": cond_true, f"{col} != {value}": cond_false}
    ).fillna(0)

    ax = df_plot.T.plot(kind="bar", figsize=figsize)

    # add percentage labels
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v*100:.1f}%" for v in container.datavalues],
            fontsize=9,
        )

    ax.set_ylabel("Percentage")
    ax.set_title(f"Target distribution by {col} condition")
    plt.xticks(rotation=0)
    plt.legend(title=target_col)

    plt.tight_layout()
    plt.show()


def plt_histogram_kde(
    df, col, color_col, labels=None, bins=30, density=True, kde=True, colors=None
):
    """
    Plot histogram distributions of a numeric column grouped by a categorical column,
    with optional Kernel Density Estimate (KDE) curves.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col : str
        Name of the numeric column to visualize.
    color_col : str
        Column used to split the data into groups (e.g., target variable).
        Each unique value will be plotted as a separate histogram.
    labels : dict, optional
        Mapping of class values to custom legend labels.
        Example: {0: "No", 1: "Yes"}.
        If not provided, raw class values will be used.
    bins : int, default=30
        Number of bins used for the histogram.
    density : bool, default=True
        If True, normalize the histogram so the total area equals 1
        (probability density). If False, show raw counts.
    kde : bool, default=True
        If True, overlay a Kernel Density Estimate curve for each class.
    colors : list of str, optional
        List of colors used for plotting the classes. If fewer colors
        are provided than classes, the list will be cycled.

    Returns
    -------
    None
        Displays the histogram plot.

    Notes
    -----
    - Histograms are plotted with transparency (`alpha=0.4`) to allow
      overlapping distributions to be visible.
    - KDE curves are only computed when at least two data points are
      available in the group.
    - Useful for exploratory data analysis to compare feature
      distributions across target classes.
    """

    if labels is None:
        labels = {}

    # Default colors: red for first class, blue for second
    if colors is None:
        colors = ["red", "blue"]

    classes = sorted(df[color_col].dropna().unique())

    plt.figure(figsize=(8, 5))

    for i, c in enumerate(classes):
        data = df[df[color_col] == c][col].dropna()
        color = colors[i % len(colors)]  # cycle if more than 2 classes

        # Histogram
        plt.hist(
            data,
            bins=bins,
            density=density,
            alpha=0.4,
            label=labels.get(c, c),
            color=color,
        )

        # KDE line
        if kde and len(data) > 1:
            kde_est = gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 300)
            plt.plot(x_vals, kde_est(x_vals), color=color)

    plt.xlabel(col)
    plt.ylabel("Density" if density else "Count")
    plt.title(f"{col} Distribution by {color_col}")
    plt.legend()
    plt.show()
