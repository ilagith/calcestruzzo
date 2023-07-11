from matplotlib import pyplot as plt
from missingno import matrix as missing_vals_distr
import numpy as np
import pandas as pd
import seaborn as sns


def data_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Explore general statistics per each variable in DataFrame.
    :param data: DataFrame to gets statistics for
    :return summary_statistics: Summary statistics per each column in data
    """
    summary_statistics = data.describe()
    return summary_statistics


def plot_pairs(data: pd.DataFrame, diagonal_plot_distribution: str) -> None:
    """
    Plot each variable one by one in scatter plots
    and enrich the diagonal with distribution per
    each variable (e.g. histogram or density plots).
    :param data: DataFrame to plot scatters and distributions for
    :param diagonal_plot_distribution: Str of distribution type ('hist' or 'kde')
    :return: None, displays a pair plot
    """
    sns.pairplot(data, diag_kind=diagonal_plot_distribution, corner=False)
    plt.show()


def examine_missing_values(data: pd.DataFrame) -> None:
    """
    Visually explore missing values distributions.
    :param data: Data to plot missing values for
    :return: None, plots missing values in data
    """
    na = missing_vals_distr(data, figsize=(25, 25))
    na = na.get_figure()
    na.show()


def plot_missing_values_percentages(data: pd.DataFrame) -> None:
    """
    Plot percentages of missing vals per column.
    :param data: Data to plot missing values percentages for
    :return: None, barplots with na percentages per column
    """

    na_percentages = data.isna().mean().round(4) * 100

    figure, axes = plt.subplots()

    for index, na_pcts in enumerate(zip(na_percentages.keys(), na_percentages.values)):
        na_pcts = list(na_pcts)
        na_pcts[1] = na_pcts[1].round(2)
        axes.bar(na_pcts[0], na_pcts[1], label=na_pcts[0])
        axes.text(index - 0.25, na_pcts[1] + 1.50, str(na_pcts[1]))  # center percentages

    axes.set_xticklabels([])
    axes.set_xticks([])
    plt.ylim(0, 100)
    plt.ylabel('NA percentage')
    plt.xlabel('Variables')
    plt.title('% of missing values per variable', size=14, fontweight="bold")
    plt.legend()
    plt.show()


def investigate_duplicates(data: pd.DataFrame) -> None:
    """
    Count number of duplicated rows in data without considering strength
    :param data: Original DataFrame
    :return: None, bar plot of duplicated rows
    """
    df_rows_occurrence = data.copy()
    duplicate_counts = df_rows_occurrence.loc[:, df_rows_occurrence.columns != 'strength'] \
        .round(2).duplicated().value_counts()  # True or False

    # Create a bar plot
    plt.bar(duplicate_counts.index.astype(str), duplicate_counts.values)
    plt.xlabel('Duplicates')
    plt.ylabel('Count')
    plt.title('Duplicate Row Counts')
    plt.show()


def inspect_outliers(data: pd.DataFrame) -> None:
    """
    Plot horizontal box plots per each variable with white background
    to inspect outliers.
    :param data: Data to inspect outliers for
    :return: None, shows boxplots
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 5))
    sns.boxplot(data=data, palette="colorblind", orient="h")
    plt.show()


def check_correlations(data: pd.DataFrame) -> None:
    """
    Plot correlation matrix with values below the diagonal only.
    :param data: Data to plot correlations for
    :return: None, displays correlation matrix
    """
    plt.figure(figsize=(30, 20))
    below_diagonal = np.triu(np.ones_like(data.corr(), dtype=bool))
    sns.heatmap(data.corr(), annot=True, linewidth=.50, cmap=sns.color_palette("viridis"), mask=below_diagonal)
    plt.show()



