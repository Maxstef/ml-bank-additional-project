import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(
    df, figsize=(10, 8), title="Correlation Heatmap of Numeric Features"
):
    corr = df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title)
    plt.show()
