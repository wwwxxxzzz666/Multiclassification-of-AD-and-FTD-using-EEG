import numpy as np
import pandas as pd
import os

from const import *
from filter import get_path

feature_set = [
    "first_feature",
    "second_feature",
    "third_feature",
    "forth_feature",
    "fifth_feature",
    "sixth_feature",
]


def assemly_dataset(fre_name: str, nolist: list[int]):
    assemly = None
    target_path = os.path.join("..", "output", "dataset", "all-" + fre_name + ".csv")
    for no in nolist:
        for current_set in feature_set:
            data_path = os.path.join(
                "..", "output", current_set, get_path(no) + "-" + fre_name + ".csv"
            )
            current_df = pd.read_csv(data_path)
            if no < 37:
                current_df.replace({"class": [1, 2]}, 0, inplace=True)  # A
            elif no < 66:
                current_df.replace({"class": [0, 2]}, 1, inplace=True)  # C
            else:
                current_df.replace({"class": [0, 1]}, 2, inplace=True)  # F
            # print(current_df)
            if assemly is None:
                assemly = current_df
            else:
                assemly = pd.concat([assemly, current_df], ignore_index=True)
    assemly.drop("Unnamed: 0", axis=1, inplace=True)
    # assemly.replace({"class": [0, 1, 2]}, 10)
    assemly = assemly.sample(frac=1, random_state=666)
    print(assemly)
    assemly.to_csv(target_path, float_format="%.10f")


if __name__ == "__main__":
    nolist = (
        [i for i in range(1, 1 + 26)]
        + [i for i in range(37, 37 + 26)]
        + [i for i in range(66, 89)]
    )
    assemly_dataset("beta", nolist)
