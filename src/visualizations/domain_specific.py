import matplotlib.pyplot as plt


def plot_previous_pdays_poutcome(df, target_col="y"):
    """
    Visualize the relationship between previous/pdays, poutcome, and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns 'previous', 'pdays', 'poutcome', and target_col.
    target_col : str
        Name of the target column (default "y").
    """
    # Prepare target mapping
    df_plot = df.copy()
    df_plot[target_col] = df_plot[target_col].map({"yes": 1, "no": 0})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Previous != 0 vs poutcome
    df_prev = df_plot[df_plot["previous"] != 0]
    prev_counts = (
        df_prev.groupby("poutcome")[target_col]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    prev_counts.plot(
        kind="bar", stacked=True, ax=axes[0, 0], color=["skyblue", "salmon"]
    )
    axes[0, 0].set_title("Previous != 0: Target distribution by Poutcome")
    axes[0, 0].set_ylabel("Proportion")

    # Previous == 0 vs poutcome
    df_prev0 = df_plot[df_plot["previous"] == 0]
    prev0_counts = (
        df_prev0.groupby("poutcome")[target_col]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    prev0_counts.plot(
        kind="bar", stacked=True, ax=axes[0, 1], color=["lightgreen", "darkorange"]
    )
    axes[0, 1].set_title("Previous == 0: Poutcome distribution")
    axes[0, 1].set_ylabel("Proportion")

    # Pdays != 999 vs poutcome
    df_pdays = df_plot[df_plot["pdays"] != 999]
    pdays_counts = (
        df_pdays.groupby("poutcome")[target_col]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    pdays_counts.plot(
        kind="bar", stacked=True, ax=axes[1, 0], color=["skyblue", "salmon"]
    )
    axes[1, 0].set_title("Pdays != 999: Target distribution by Poutcome")
    axes[1, 0].set_ylabel("Proportion")

    # Pdays == 999 vs poutcome
    df_pdays999 = df_plot[df_plot["pdays"] == 999]
    pdays999_counts = (
        df_pdays999.groupby("poutcome")[target_col]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    pdays999_counts.plot(
        kind="bar", stacked=True, ax=axes[1, 1], color=["lightgreen", "darkorange"]
    )
    axes[1, 1].set_title("Pdays == 999: Poutcome distribution")
    axes[1, 1].set_ylabel("Proportion")

    plt.tight_layout()
    plt.show()
