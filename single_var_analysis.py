import numpy as np
import pandas as pd
import os, math
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib.pyplot as plt
import seaborn as sns
import palettable

from const import *
from filter import get_path


def bar_plot(fre_name: str) -> None:
    dataset_path = os.path.join("..", "output", "dataset", "all-" + fre_name + ".csv")
    fig_dir = os.path.join("..", "output", "figure", "box", "")
    dataset = pd.read_csv(dataset_path)
    dataset.drop("Unnamed: 0", axis=1, inplace=True)
    dataset.replace({"class": label_text}, inplace=True)

    for column_name in dataset.columns:
        plt.figure(dpi=120)
        sns.boxplot(
            data=dataset,
            x="class",
            y=column_name,
            showfliers=False,
            palette=palettable.tableau.TrafficLight_9.mpl_colors,
        )
        plt.title(fre_name + "-" + column_name)
        plt.savefig(fig_dir + fre_name + "-" + column_name + ".png", dpi=660)
        # plt.show()


def anova_analysis(fre_name: str) -> None:
    dataset_path = os.path.join("..", "output", "dataset", "all-" + fre_name + ".csv")
    dataset = pd.read_csv(dataset_path)
    dataset.drop("Unnamed: 0", axis=1, inplace=True)
    dataset.replace({"class": label_text}, inplace=True)
    dataset.rename(columns={"class": "Class"}, inplace=True)
    p1_result = {}

    for column_name in dataset.columns[:-1]:
        current_model = ols(formula=column_name + " ~ C(Class)", data=dataset).fit()
        anova_table = anova_lm(current_model)
        print(fre_name + "-" + column_name + " result:")
        print(anova_table)
        p1_result[column_name] = anova_table["PR(>F)"].iloc[0]

        print(pairwise_tukeyhsd(dataset[column_name], dataset["Class"]))

    p1_result = sorted(p1_result.items(), key=lambda s: (s[1], s[0]))
    valid_count = 0
    for col, pv in p1_result:
        if pv <= 0.05:
            print(CYAN + "{} : {}".format(col, pv) + END)
            valid_count += 1
        else:
            print(RED + "{} : {}".format(col, pv) + END)
    print(HIGHLIGHT + "{} valid features in total.".format(valid_count) + END)


if __name__ == "__main__":
    # bar_plot("delta")
    # bar_plot("theta")
    # bar_plot("alpha")
    # bar_plot("beta")
    anova_analysis("beta")
