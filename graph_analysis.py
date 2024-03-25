import networkx as nx
import numpy as np
import pandas as pd
import os, math, scipy.signal
import scipy.optimize as opt
import scipy.special

import matplotlib.pyplot as plt
import seaborn as sns

from const import *
from filter import get_path


def dd_func(k: float, lamb: float) -> float:
    return (lamb**k) * math.exp(-lamb) / scipy.special.factorial(k)


class feature:
    def __init__(self, channels: int, label: int) -> None:
        self.channels = channels
        self.mcc = None
        self.mpc = None
        self.ddi = np.zeros(channels)
        self.entropy = np.zeros(channels)
        self.awd = np.zeros(channels)
        self.ave_pli = np.zeros(channels)
        self.label = int(label)

    # Calculate the average C_{i,1} for the network G
    def cal_mcc(self, G: list[nx.Graph], layer: int, time_n: int) -> None:
        self.mcc = 0
        for i in range(time_n):
            numerator = 0
            denominator = 0
            for alpha in range(layer):
                neibourhoods = list(G[alpha][i])
                for keppa in range(layer):
                    if keppa == alpha:
                        continue
                    for sub1 in range(len(neibourhoods) - 1):
                        for sub2 in range(sub1 + 1, len(neibourhoods)):
                            j = neibourhoods[sub1]
                            m = neibourhoods[sub2]
                            if m not in list(G[keppa][j]):
                                break
                            numerator += np.cbrt(
                                G[alpha][i][j]["weight"]
                                * G[alpha][i][m]["weight"]
                                * G[keppa][j][m]["weight"]
                            )
                denominator += G[alpha].degree(i) * (G[alpha].degree(i) - 1)  # unsure?
            if denominator == 0:
                self.mcc += 0
            else:
                self.mcc += numerator / ((layer - 1) * denominator)
        self.mcc /= time_n

    def cal_mpc(self, G: list[nx.Graph], layer: int, time_n: int) -> None:
        self.mpc = 0
        for i in range(time_n):
            term = 1
            o_i = 0
            for l in range(layer):
                o_i += G[l].degree(i)
            if o_i > 0:
                for l in range(layer):
                    term -= (G[l].degree(i) / o_i) ** 2
        self.mpc = layer * term / ((layer - 1) * time_n)

    def cal_ddi_entropy(
        self, G: list[nx.Graph], layer: int, time_n: int, no: int, cur_fre: str
    ) -> None:
        x_data = np.arange(20)
        for ch in range(layer):
            dd = nx.degree_histogram(G[ch])[:20]
            for _ in range(len(dd), 20):
                dd.append(0)
            dd = np.array(dd)
            dd = dd / dd.sum()
            # print(RED + str(dd) + END)
            popt, pcov = opt.curve_fit(dd_func, xdata=x_data, ydata=dd, maxfev=2000)
            self.ddi[ch] = popt[0]

            # Draw the degree distribution graph
            # if ch == 0:
            #     plt.figure(figsize=(4, 4), dpi=120)
            #     # plt.plot(x_data, dd, "gray")
            #     plt.scatter(x_data, dd, marker="*")
            #     y_data_esi = [dd_func(k, self.ddi[ch]) for k in x_data]
            #     plt.plot(x_data, y_data_esi, "cyan")
            #     plt.title(cur_fre)  # str(self.ddi[ch]))
            #     plt.savefig("sub-{}-{}-dd_0.svg".format(no, cur_fre))
            #     # plt.show()

            self.entropy[ch] = 0
            for p_k in dd:
                if p_k > 0:
                    self.entropy[ch] -= p_k * math.log2(p_k)

    def cal_awd(self, G: list[nx.Graph], layer: int, time_n: int) -> None:
        for ch in range(layer):
            self.awd[ch] = 0
            for i in range(time_n):
                neibourhoods = list(G[ch][i])
                for j in neibourhoods:
                    self.awd[ch] += G[ch][i][j]["weight"]
            self.awd[ch] /= time_n

    def cal_ave_pli(self, pli_mat: list[list[np.float32]], layer: int) -> None:
        for ch1 in range(layer):
            self.ave_pli[ch1] = 0
            for ch2 in range(0, ch1):
                self.ave_pli[ch1] += pli_mat[ch2][ch1]
            for ch2 in range(ch1 + 1, layer):
                self.ave_pli[ch1] += pli_mat[ch1][ch2]
            self.ave_pli[ch1] /= layer

    def extract_numpy(self) -> list[np.float32]:
        return np.concatenate(
            [
                [self.mcc, self.mpc],
                self.ddi,
                self.entropy,
                self.awd,
                self.ave_pli,
                [self.label],
            ],
            dtype=np.float32,
        )

    def extract_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self.extract_numpy()],
            columns=["mcc", "mpc"]
            + ["ddi_" + str(i) for i in range(self.channels)]
            + ["entropy_" + str(i) for i in range(self.channels)]
            + ["awd_" + str(i) for i in range(self.channels)]
            + ["awe_pli_" + str(i) for i in range(self.channels)]
            + ["class"],
        )


def build_graph(no: int, label: int):
    sub_path = get_path(no)
    input_path = os.path.join("..", "output", "filter", sub_path)
    extract_dir = os.path.join("..", "output", "sixth_feature", "")
    fre_name = ["delta", "theta", "alpha", "beta"]
    for cur_fre in fre_name:
        results_df = None
        np_data = np.load(os.path.join(input_path, cur_fre + ".npy"))
        print(
            INFO
            + "{}'s epochs have been loaded, whose shape is {}".format(
                cur_fre, np_data.shape
            )
            + END
        )

        for epoch in range(25, 26):  # np_data.shape[0]):
            G = [nx.Graph() for _ in range(np_data.shape[1])]
            # Calculate the weight visibility graph in each channel
            for ch in range(np_data.shape[1]):
                for t2 in range(np_data.shape[2]):
                    G[ch].add_node(t2)
                    flag = True
                    fail_tk = -1000
                    for t in range(max(0, t2 - 251), t2 - 1):
                        if flag == False and t < fail_tk:
                            continue
                        flag = (
                            True  # Used to check whether t and t2 could see each other
                        )
                        r_term = (np_data[epoch][ch][t] - np_data[epoch][ch][t2]) / (
                            t - t2
                        )
                        for t_k in range(t + 1, t2):
                            if (np_data[epoch][ch][t] - np_data[epoch][ch][t_k]) / (
                                t - t_k
                            ) >= r_term:
                                flag = False
                                fail_tk = t_k
                                break
                        if flag:
                            G[ch].add_edge(
                                # str(ch) + "_" + str(t),
                                # str(ch) + "_" + str(t2),
                                t,
                                t2,
                                weight=abs(
                                    math.atan2(
                                        np_data[epoch][ch][t] - np_data[epoch][ch][t2],
                                        t - t2,
                                    )
                                ),
                            )

            # Draw the WVG
            # for u, v in G[0].edges():
            #     G[0][u][v]["weight"] = (G[0][u][v]["weight"] / np.pi) ** 3
            # edge_weights = [G[0][u][v]["weight"] for u, v in G[0].edges()]
            # # print(edge_weights)
            # pos = nx.spring_layout(G[0], iterations=50)
            # plt.figure(figsize=(4, 4), dpi=120)
            # nx.draw_networkx_edges(
            #     G[0],
            #     pos=pos,
            #     edge_color="dodgerblue",
            #     alpha=[weight / max(edge_weights) for weight in edge_weights],
            #     arrows=True,
            #     connectionstyle="arc3,rad=0.2",
            #     node_size=0,
            # )
            # # labels = {node: str(node) for node in G[0].nodes()}
            # # nx.draw_networkx_labels(G[0], pos, labels=labels, font_size=10)
            # plt.title(cur_fre)
            # plt.savefig("sub-{}-{}-wvg_0.svg".format(no, cur_fre))
            # plt.show()

            print(
                INFO
                + "The weight visibility graph has been built successfully. Current Epoch:{}:{}".format(
                    epoch, np_data.shape[0]
                )
                + END
            )

            # Calculate the PLI matrix between different channels
            pli_mat = np.zeros((np_data.shape[1], np_data.shape[1]), dtype=np.float32)
            analytic_sig = []
            phase = []
            for ch in range(np_data.shape[1]):
                analytic_sig.append(scipy.signal.hilbert(np_data[epoch][ch]))
                # np.imag(analytic_sig[ch]) / np.real(analytic_sig[ch])
                phase.append(np.angle(analytic_sig[ch]))
            for ch1 in range(0, np_data.shape[1] - 1):
                for ch2 in range(
                    0, np_data.shape[1] - 1
                ):  # (ch1 + 1, np_data.shape[1]):
                    pli_mat[ch1][ch2] = abs(
                        np.sum(np.sign(np.imag(np.exp(1j * (phase[ch1] - phase[ch2])))))
                        / np_data.shape[2]
                    )
                    # for t in range(np_data.shape[2]):
                    #     G.add_edge(
                    #         str(ch1) + "_" + str(t),
                    #         str(ch2) + "_" + str(t),
                    #         weight=pli_mat[ch1][ch2],
                    #     )

            # Draw the PLI matrix heatmap
            plt.figure(figsize=(4, 4), dpi=120)
            sns.heatmap(
                pli_mat,
                cmap=sns.light_palette("#2ecc71", as_cmap=True),
                linewidths=0.2,
                vmin=0,
                vmax=1,
                annot=False,
                annot_kws={"size": 5, "weight": "normal", "color": "green"},
            )
            plt.title(cur_fre)  # "PLI matrix")
            plt.savefig("sub-{}-{}-pli.svg".format(no, cur_fre))
            # plt.show()

            print(
                INFO
                + "The PLI matrix has been calculated completely. Current Epoch:{}:{}".format(
                    epoch, np_data.shape[0]
                )
                + END
            )

            # Analyze and calculate the features from the network
            result = feature(np_data.shape[1], label=label)
            result.cal_mcc(G, np_data.shape[1], np_data.shape[2])
            result.cal_mpc(G, np_data.shape[1], np_data.shape[2])
            result.cal_ddi_entropy(G, np_data.shape[1], np_data.shape[2], no, cur_fre)
            result.cal_awd(G, np_data.shape[1], np_data.shape[2])
            result.cal_ave_pli(pli_mat=pli_mat, layer=np_data.shape[1])

            print(
                INFO
                + "The features has been calculated completely. Current Epoch:{}:{}".format(
                    epoch, np_data.shape[0]
                )
                + END
            )

        #     if results_df is None:
        #         results_df = result.extract_df()
        #     else:
        #         results_df = pd.concat(
        #             [results_df, result.extract_df()], ignore_index=True
        #         )
        # results_df.to_csv(
        #     extract_dir + sub_path + "-" + cur_fre + ".csv", float_format="%.10f"
        # )


if __name__ == "__main__":
    print(INFO + "Test Starts!" + END)
    # for no in range(1, 16):
    build_graph(17, 0)
    build_graph(67, 0)
    build_graph(52, 0)
    # for no in range(1, 1 + 26):
    #     build_graph(no, 0)  #!!! THE LABEL MAY NEED TO BE CHANGED
    # for no in range(37, 37 + 26):
    #     build_graph(no, 1)  #!!! THE LABEL MAY NEED TO BE CHANGED
    # for no in range(66, 66 + 23):
    #     build_graph(no, 2)  #!!! THE LABEL MAY NEED TO BE CHANGED
