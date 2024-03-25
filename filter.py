import numpy as np
import random as rd
import mne
import os

RED = "\033[0;31;40m"
YELLOW = "\033[0;33;40m"
CYAN = "\033[0;36;40m"
HIGHLIGHT = "\033[1;4;37;40m"
INFO = "\033[2;3;37;40m"
END = "\033[0m"

annotation_set = set()


def get_path(no: int) -> str:
    path = "sub-"
    if no < 10:
        path += "00"
    elif no < 100:
        path += "0"
    path += str(no)
    return path


def filter_set(no: int, crop_len: int):
    seed_by_no = rd.randint(1, 10000)
    sub_path = get_path(no)
    target_prefix = os.path.join("..", "output", "filter", sub_path)
    # target_prefix = "../output/filter/" + sub_path + "/"
    if not os.path.exists(target_prefix):
        os.mkdir(target_prefix)
    data_prefix = os.path.join(
        "..",
        "dataset",
        "derivatives",
        sub_path,
        "eeg",
        sub_path + "_task-eyesclosed_eeg.set",
    )
    raw = mne.io.read_raw_eeglab(
        data_prefix,
        "auto",
        True,
        "utf-8",
    )
    # raw = mne.io.read_raw_eeglab(
    #     "../dataset/derivatives/"
    #     + sub_path
    #     + "/eeg/"
    #     + sub_path
    #     + "_task-eyesclosed_eeg.set",
    #     "auto",
    #     True,
    #     "utf-8",
    # )
    print(INFO + sub_path + " loaded!" + END)

    # Rename the annotations
    global annotation_set
    mapping_dict = {"boundary": "bad"}
    if len(raw.annotations.count().keys()) > 0:
        raw.annotations.rename(mapping=mapping_dict)
        for x in raw.annotations.count().keys():
            annotation_set.add(x)
    print(INFO + sub_path + "'s annotations have been checked." + END)

    delta = raw.copy().filter(0, 4, method="fir", verbose=False)
    theta = raw.copy().filter(4, 8, method="fir", verbose=False)
    alpha = raw.copy().filter(8, 16, method="fir", verbose=False)
    beta = raw.copy().filter(16, 32, method="fir", verbose=False)

    print(INFO + sub_path + " has been filtered." + END)

    # INFO FOR DEBUG
    # print(delta.info)
    # print(delta.n_times)
    # print(delta.ch_names)
    # print()
    # print(delta.annotations)
    # delta.plot(duration=10, scalings="auto", show_options=True, block=True)

    data = [delta, theta, alpha, beta]
    fre_name = ["delta", "theta", "alpha", "beta"]
    resample_rate = [50, 20, 10, 5]
    for idx, current_data in enumerate(data):
        epochs = mne.make_fixed_length_epochs(
            current_data,
            duration=crop_len,
            overlap=crop_len / 4,
            preload=False,
            reject_by_annotation=True,
            verbose=False,
        )
        np_data = epochs.get_data(copy=True, verbose=False)
        print(
            INFO
            + "The shape of {}'s epochs array is {}".format(
                fre_name[idx], np_data.shape
            )
            + END
        )

        np_data = np.array(
            [
                [np_data[i][j][:: resample_rate[idx]] for j in range(np_data.shape[1])]
                for i in range(np_data.shape[0])
            ]
        )

        for i in range(np_data.shape[0]):
            for j in range(np_data.shape[1]):
                np_data[i][j] = (np_data[i][j] - np_data[i][j].mean()) / np_data[i][
                    j
                ].std()

        np.random.seed(seed_by_no)
        np.random.shuffle(np_data)

        # plt.plot(np_data[0][0][:5000], "k--")
        # plt.plot(np.arange(0, 5000, 50), np_data[0][0][: int(5000 / 50)], "b--")
        # plt.show()

        print(
            INFO
            + "After resampling and normlization, the shape of {}'s epochs array is {}".format(
                fre_name[idx], np_data.shape
            )
            + END
        )

        np.save(os.path.join(target_prefix, fre_name[idx]), np_data)

    # print(delta.annotations)
    # delta.plot(duration=10, scalings="auto", show_options=True, block=True)


if __name__ == "__main__":
    print(HIGHLIGHT + "This is just a test!" + END)
    print(INFO + "This is just a test!" + END)
    print(CYAN + "This is just a test!" + END)
    print(YELLOW + "This is just a test!" + END)
    print(RED + "This is just a test!" + END)
    # mne.viz.set_browser_backend("qt")
    rd.seed(666)
    # for no in range(1, 37):
    # for no in range(37, 66):
    # for no in range(66, 89):
    for no in range(1, 2):
        filter_set(no, 5)
    print(annotation_set)
