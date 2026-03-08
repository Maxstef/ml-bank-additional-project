import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def draw_countplot(
    df,
    col,
    hue_col,
    normalize=False,
    title=None,
    ax=None,
):
    """
    Draw a bar plot showing counts or normalized percentages by category and hue.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col : str
        Categorical column whose values are counted.
    hue_col : str
        Categorical column used to group bars (hue).
    normalize : bool, default False
        If True, counts are converted to percentages per hue group.
    title : str, optional
        Title of the plot.

    Notes
    -----
    - Bars are grouped by `hue_col` and sorted by the first hue value for consistent ordering.
    - Labels are displayed on top of each bar.
    """

    if ax is None:
        ax = plt.gca()

    hue_values = df[hue_col].dropna().unique()

    if normalize:
        data = df.groupby(hue_col)[col].value_counts(normalize=True).mul(100).round(2)
        label_fmt = "{:,.1f}%"
    else:
        data = df.groupby(hue_col)[col].value_counts()
        label_fmt = ""

    (
        data.unstack(hue_col)
        .sort_values(by=hue_values[0], ascending=False)
        .plot.bar(ax=ax, title=title)
    )

    for container in ax.containers:
        ax.bar_label(container, fmt=label_fmt)

    ax.set_xlabel(col)

    # plt.tight_layout()
    # plt.show()


def plot_target_rate_by_category(
    df,
    col,
    target_col,
    sort=True,
    figsize=(8, 5),
    title=None,
):
    """
    Plot the average target rate for each category of a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    col : str
        Categorical column to group by.
    target_col : str
        Target column (should be binary, e.g. 0/1).
    sort : bool, default True
        Whether to sort categories by target rate.
    figsize : tuple, default (8,5)
        Size of the figure.
    title : str, optional
        Custom plot title. If None, a default title is used.

    Notes
    -----
    The plot shows:
        P(target = 1 | category)

    This helps identify categories associated with higher or lower
    target probability.
    """

    data = df.groupby(col)[target_col].mean()

    if sort:
        data = data.sort_values()

    ax = data.plot.bar(figsize=figsize)

    ax.set_xlabel(col)
    ax.set_ylabel(f"Mean {target_col} (Target Rate)")
    ax.set_title(title or f"Target Rate by {col}")

    # add percentage labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.show()


def plot_target_distribution_by_category(
    df,
    col,
    target_col,
    normalize=True,
    sort=True,
    figsize=(8, 5),
    title=None,
):
    """
    Plot stacked target distribution within each category.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    col : str
        Categorical column used for grouping.
    target_col : str
        Target column (binary or categorical).
    normalize : bool, default True
        If True, show proportions within each category.
        If False, show raw counts.
    sort : bool, default True
        Sort categories by the first target class.
    figsize : tuple, default (8,5)
        Figure size.
    title : str, optional
        Custom plot title.

    Notes
    -----
    Each bar represents one category of `col`.

    - If normalize=True:
        Bars sum to 100% and show target composition.

    - If normalize=False:
        Bars show raw counts.

    Useful for understanding:

        P(target | category)
    """

    data = df.groupby(col)[target_col].value_counts(normalize=normalize).unstack()

    if sort:
        data = data.sort_values(by=data.columns[0], ascending=False)

    ax = data.plot(kind="bar", stacked=True, figsize=figsize)

    ax.set_xlabel(col)
    ax.set_ylabel("Percentage" if normalize else "Count")
    ax.set_title(title or f"{target_col} distribution by {col}")

    # add labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f" if normalize else "%d", label_type="center")

    plt.tight_layout()
    plt.show()


def plot_categorical_heatmap(
    df,
    col1,
    col2,
    figsize=(8, 6),
    cmap="Blues",
    title=None,
    normalize=False,  # accept 'all', 'index', 'columns', or False
    row_order=None,
    col_order=None,
):
    """
    Plot heatmap of counts or normalized percentages for two categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the categorical columns.
    col1 : str
        Column to use as rows.
    col2 : str
        Column to use as columns.
    figsize : tuple, default (8,6)
        Figure size.
    cmap : str, default "Blues"
        Color map for the heatmap.
    title : str, optional
        Custom title for the plot.
    normalize : str or None, default None
        How to normalize counts:
        - 'all'     : proportion of total
        - 'index'   : row-wise proportion
        - 'columns' : column-wise proportion
        - False      : raw counts
    row_order : list, optional
        Explicit order of rows (values in col1).
    col_order : list, optional
        Explicit order of columns (values in col2).
    """

    # Validate normalize
    valid_opts = [False, "all", "index", "columns"]
    if normalize not in valid_opts:
        raise ValueError(f"normalize must be one of {valid_opts}")

    # Convert columns to categorical if order is provided
    df_plot = df.copy()
    if row_order is not None:
        df_plot[col1] = pd.Categorical(
            df_plot[col1], categories=row_order, ordered=True
        )
    if col_order is not None:
        df_plot[col2] = pd.Categorical(
            df_plot[col2], categories=col_order, ordered=True
        )

    ct = pd.crosstab(df_plot[col1], df_plot[col2], normalize=normalize)

    plt.figure(figsize=figsize)
    sns.heatmap(ct, cmap=cmap, annot=True, fmt=".2f" if normalize else "d")

    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.title(title or f"{col1} vs {col2} {'proportion' if normalize else 'frequency'}")
    plt.tight_layout()
    plt.show()


def plot_target_rate_heatmap(
    df,
    col1,
    col2,
    target_col,
    figsize=(8, 6),
    cmap="RdYlGn",
    title=None,
    normalize=None,  # None, 'all', 'index', 'columns'
    row_order=None,
    col_order=None,
):
    """
    Plot heatmap of target mean for combinations of two categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col1 : str
        Column to use as rows.
    col2 : str
        Column to use as columns.
    target_col : str
        Target column (binary or numeric).
    figsize : tuple, default (8,6)
        Figure size.
    cmap : str, default "RdYlGn"
        Color map.
    title : str, optional
        Plot title.
    normalize : str or None, default None
        How to normalize:
        - None     : raw mean
        - 'all'    : divide all values by total sum
        - 'index'  : row-wise proportion
        - 'columns': column-wise proportion
    row_order : list, optional
        Explicit order for rows.
    col_order : list, optional
        Explicit order for columns.
    """

    # Copy df to avoid modifying original
    df_plot = df.copy()

    # Convert to categorical if order provided
    if row_order is not None:
        df_plot[col1] = pd.Categorical(
            df_plot[col1], categories=row_order, ordered=True
        )
    if col_order is not None:
        df_plot[col2] = pd.Categorical(
            df_plot[col2], categories=col_order, ordered=True
        )

    # Pivot table (mean target)
    ct = df_plot.pivot_table(
        index=col1, columns=col2, values=target_col, aggfunc="mean"
    ).astype(float)

    # Apply normalization
    if normalize == "all":
        ct = ct / ct.values.sum()
    elif normalize == "index":
        ct = ct.div(ct.sum(axis=1), axis=0)  # row-wise
    elif normalize == "columns":
        ct = ct.div(ct.sum(axis=0), axis=1)  # column-wise

    plt.figure(figsize=figsize)
    sns.heatmap(ct, annot=True, fmt=".2f", cmap=cmap)

    plt.xlabel(col2)
    plt.ylabel(col1)
    title_suffix = ""
    if normalize is not None:
        title_suffix = f" ({normalize} normalized)"
    plt.title(title or f"{target_col} rate by {col1} and {col2}" + title_suffix)

    plt.tight_layout()
    plt.show()
