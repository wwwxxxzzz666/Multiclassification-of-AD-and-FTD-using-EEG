import numpy as np
import pandas as pd
import os, math
import codecs

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    fbeta_score,
    recall_score,
    auc,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import FastICA, KernelPCA, IncrementalPCA, PCA, SparsePCA
from sklearn.dummy import DummyClassifier


from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


from const import *


def get_feature_weight_list(X_train, y_train):
    forest_clf = RandomForestClassifier(n_estimators=200, criterion="entropy")
    forest_clf.fit(X_train, y_train)
    feature_weight_list = []
    for term1, term2 in zip(
        forest_clf.feature_importances_, forest_clf.feature_names_in_
    ):
        feature_weight_list.append((term2, term1))
    feature_weight_list.sort(key=lambda u: u[1], reverse=True)
    print(feature_weight_list)
    print(np.array(feature_weight_list)[:, 0])
    y_prediction = cross_val_predict(forest_clf, X_train, y_train, cv=5)
    print(f1_score(y_train, y_prediction, average="macro"))
    print(accuracy_score(y_train, y_prediction))


def train() -> None:
    fre_names = ["theta", "delta", "alpha", "beta"]
    dataset = None
    for cur_fre in fre_names:
        cur_dataset_path = os.path.join(
            "..", "output", "dataset", "all-" + cur_fre + ".csv"
        )
        cur_dataset = pd.read_csv(cur_dataset_path).drop(
            columns=["Unnamed: 0"], inplace=False
        )
        rename_dict = {}
        for column in cur_dataset.columns:
            rename_dict[column] = cur_fre + "_" + column
        cur_dataset.rename(columns=rename_dict, inplace=True)
        if dataset is None:
            dataset = cur_dataset
        else:
            dataset = pd.concat([dataset, cur_dataset], axis=1)
    dataset.drop(columns=["theta_class", "delta_class", "alpha_class"], inplace=True)
    dataset.rename(columns={"beta_class": "class"}, inplace=True)
    print(dataset)
    dataset_lables = dataset["class"]
    dataset_inputs = dataset.drop(columns=["class"])
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_inputs, dataset_lables, test_size=0.2, random_state=666
    )

    X_columns = X_train.columns
    pipeline = ColumnTransformer(
        [
            ("std_scaler", StandardScaler(), X_train.columns),
        ]
    )
    # X_train = pd.DataFrame(pipeline.fit_transform(X_train), columns=X_columns)

    get_feature_weight_list(X_train, y_train)

    # Dummy Classifier
    dummy_clf_list = [
        multiclass_test_module(
            DummyClassifier(strategy="prior", random_state=666), "DummyClassifier_prior"
        ),
        multiclass_test_module(
            DummyClassifier(strategy="uniform", random_state=666),
            "DummyClassifier_uniform",
        ),
        multiclass_test_module(
            DummyClassifier(strategy="stratified", random_state=666),
            "DummyClassifier_stratified",
        ),
    ]

    # Linear Classifier
    linear_clf_list = [
        multiclass_test_module(
            SGDClassifier("hinge", random_state=666), "SGDClassifier_hinge"
        ),
        multiclass_test_module(
            SGDClassifier("log_loss", random_state=666), "SGDClassifier_logistic"
        ),
        multiclass_test_module(
            SGDClassifier("perceptron", random_state=666), "SGDClassifier_perceptron"
        ),
        multiclass_test_module(
            PassiveAggressiveClassifier(random_state=666), "PassiveAggressiveClassifier"
        ),
        multiclass_test_module(RidgeClassifier(random_state=666), "RidgeClassifier"),
    ]

    # Tree Classifier
    tree_clf_list = [
        multiclass_test_module(
            DecisionTreeClassifier(criterion="gini", random_state=666),
            "DecisionTreeClassifier_gini",
        ),
        multiclass_test_module(
            DecisionTreeClassifier(criterion="entropy", random_state=666),
            "DecisionTreeClassifier_entropy",
        ),
        multiclass_test_module(
            DecisionTreeClassifier(criterion="log_loss", random_state=666),
            "DecisionTreeClassifier_log_loss",
        ),
        multiclass_test_module(
            ExtraTreeClassifier(criterion="gini", random_state=666),
            "ExtraTreeClassifier_gini",
        ),
        multiclass_test_module(
            ExtraTreeClassifier(criterion="entropy", random_state=666),
            "ExtraTreeClassifier_entropy",
        ),
        multiclass_test_module(
            ExtraTreeClassifier(criterion="log_loss", random_state=666),
            "ExtraTreeClassifier_log_loss",
        ),
    ]

    # SVC
    svc_list = [
        multiclass_test_module(SVC(kernel="rbf", random_state=666), "SVC_rbf"),
        # multiclass_test_module(SVC(kernel="poly", random_state=666), "SVC_poly"),
        multiclass_test_module(SVC(kernel="sigmoid", random_state=666), "SVC_sigmoid"),
        multiclass_test_module(NuSVC(kernel="rbf", random_state=666), "NuSVC_rbf"),
        # multiclass_test_module(NuSVC(kernel="poly", random_state=666), "NuSVC_poly"),
        # multiclass_test_module(
        #     NuSVC(kernel="sigmoid", random_state=666), "NuSVC_sigmoid"
        # ),
        multiclass_test_module(LinearSVC(random_state=666), "LinearSVC"),
    ]

    # SVC_ovo_ovr
    svc_ov_list = [
        multiclass_test_module(
            SVC(kernel="rbf", decision_function_shape="ovr", random_state=666),
            "SVC_rbf_ovr",
        ),
        multiclass_test_module(
            SVC(kernel="rbf", decision_function_shape="ovo", random_state=666),
            "SVC_rbf_ovo",
        ),
        multiclass_test_module(
            NuSVC(kernel="rbf", decision_function_shape="ovr", random_state=666),
            "NuSVC_rbf_ovr",
        ),
        multiclass_test_module(
            NuSVC(kernel="rbf", decision_function_shape="ovo", random_state=666),
            "NuSVC_rbf_ovo",
        ),
    ]

    # SVC_c
    svc_c_list = [
        # multiclass_test_module(
        #     SVC(kernel="rbf", C=0.5, random_state=666),
        #     "SVC_C=0.5",
        # ),
        # multiclass_test_module(
        #     SVC(kernel="rbf", C=1, random_state=666),
        #     "SVC_C=1",
        # ),
        multiclass_test_module(
            SVC(kernel="rbf", C=1.5, random_state=666),
            "SVC_C=1.5",
        ),
        multiclass_test_module(
            SVC(kernel="rbf", C=2, random_state=666),
            "SVC_C=2",
        ),
        multiclass_test_module(
            SVC(kernel="rbf", C=3, random_state=666),
            "SVC_C=3",
        ),
        # multiclass_test_module(
        #     NuSVC(kernel="rbf", nu=0.3, random_state=666),
        #     "NuSVC_nu=0.3",
        # ),
        # multiclass_test_module(
        #     NuSVC(kernel="rbf", nu=0.5, random_state=666),
        #     "NuSVC_nu=0.5",
        # ),
        multiclass_test_module(
            NuSVC(kernel="rbf", nu=0.7, random_state=666),
            "NuSVC_nu=0.7",
        ),
        multiclass_test_module(
            NuSVC(kernel="rbf", nu=0.8, random_state=666),
            "NuSVC_nu=0.8",
        ),
        multiclass_test_module(
            NuSVC(kernel="rbf", nu=0.9, random_state=666),
            "NuSVC_nu=0.9",
        ),
    ]

    # Neighbor Classifier
    neighbor_clf_list = [
        multiclass_test_module(
            KNeighborsClassifier(n_neighbors=5), "KNeighborsClassifier_5"
        ),
        multiclass_test_module(
            KNeighborsClassifier(n_neighbors=10), "KNeighborsClassifier_10"
        ),
        multiclass_test_module(
            KNeighborsClassifier(n_neighbors=25), "KNeighborsClassifier_25"
        ),
        multiclass_test_module(
            KNeighborsClassifier(n_neighbors=50), "KNeighborsClassifier_50"
        ),
        multiclass_test_module(
            KNeighborsClassifier(n_neighbors=100), "KNeighborsClassifier_100"
        ),
        # multiclass_test_module(   # No neighbor with large space
        #     RadiusNeighborsClassifier(radius=1), "RadiusNeighborsClassifier_1"
        # ),
        # multiclass_test_module(
        #     RadiusNeighborsClassifier(radius=5), "RadiusNeighborsClassifier_5"
        # ),
    ]

    # Naive Bayes Classifier
    bayes_clf_list = [
        multiclass_test_module(GaussianNB(), "GaussianNB"),
        multiclass_test_module(BernoulliNB(), "BernoulliNB"),
        # multiclass_test_module(ComplementNB(), "ComplementNB"),   # Negative values in X
        # multiclass_test_module(CategoricalNB(), "CategoricalNB"),
        # multiclass_test_module(MultinomialNB(), "MultinomialNB"),
    ]

    # MLP Classifier
    mlp_clf_list = [
        multiclass_test_module(
            MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=666),
            "MLPClassifier_50",
        ),
        multiclass_test_module(
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=666),
            "MLPClassifier_100",
        ),
        multiclass_test_module(
            MLPClassifier(hidden_layer_sizes=(150,), max_iter=2000, random_state=666),
            "MLPClassifier_150",
        ),
        # multiclass_test_module(
        #     MLPClassifier(hidden_layer_sizes=(200,), max_iter=5000, random_state=666),
        #     "MLPClassifier_200",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(hidden_layer_sizes=(300,), max_iter=5000, random_state=666),
        #     "MLPClassifier_300",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(hidden_layer_sizes=(400,), max_iter=5000, random_state=666),
        #     "MLPClassifier_400",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(hidden_layer_sizes=(500,), max_iter=5000, random_state=666),
        #     "MLPClassifier_500",
        # ),
        multiclass_test_module(
            MLPClassifier(
                hidden_layer_sizes=(150, 100), max_iter=5000, random_state=666
            ),
            "MLPClassifier_150_100",
        ),
        multiclass_test_module(
            MLPClassifier(
                hidden_layer_sizes=(200, 100), max_iter=5000, random_state=666
            ),
            "MLPClassifier_200_100",
        ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(200, 150), max_iter=5000, random_state=666
        #     ),
        #     "MLPClassifier_200_150",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(250, 200), max_iter=5000, random_state=666
        #     ),
        #     "MLPClassifier_250_200",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(300, 250), max_iter=5000, random_state=666
        #     ),
        #     "MLPClassifier_300_250",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(350, 300), max_iter=5000, random_state=666
        #     ),
        #     "MLPClassifier_350_300",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(100, 200, 100), max_iter=5000, random_state=666
        #     ),
        #     "MLPClassifier_100_200_100",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(150, 100),
        #         activation="tanh",
        #         max_iter=2000,
        #         random_state=666,
        #     ),
        #     "MLPClassifier_150_100_tanh",
        # ),
        # multiclass_test_module(
        #     MLPClassifier(
        #         hidden_layer_sizes=(150, 100),
        #         activation="logistic",
        #         max_iter=2000,
        #         random_state=666,
        #     ),
        #     "MLPClassifier_150_100_logistic",
        # ),
    ]

    # Ensemble Classifier
    ensemble_clf_list = [
        multiclass_test_module(
            AdaBoostClassifier(random_state=666), "AdaBoostClassifier"
        ),
        multiclass_test_module(
            BaggingClassifier(random_state=666), "BaggingClassifier"
        ),
        multiclass_test_module(
            ExtraTreesClassifier(random_state=666), "ExtraTreesClassifier"
        ),
        multiclass_test_module(
            ExtraTreesClassifier(criterion="entropy", random_state=666),
            "ExtraTreesClassifier_entropy",
        ),
        multiclass_test_module(
            GradientBoostingClassifier(random_state=666), "GradientBoostingClassifier"
        ),
        multiclass_test_module(
            HistGradientBoostingClassifier(random_state=666),
            "HistGradientBoostingClassifier",
        ),
        multiclass_test_module(
            RandomForestClassifier(random_state=666), "RandomForestClassifier"
        ),
        multiclass_test_module(
            RandomForestClassifier(criterion="entropy", random_state=666),
            "RandomForestClassifier_entropy",
        ),
        # multiclass_test_module(StackingClassifier(), "StackingClassifier"),
        # multiclass_test_module(VotingClassifier(),"VotingClassifier")
    ]

    # Gaussian Process Classifier
    gp_clf_list = [
        multiclass_test_module(
            GaussianProcessClassifier(random_state=666), "GaussianProcessClassifier"
        )
    ]

    # Voting Classifier
    svm_clf = SVC(C=1.5, kernel="rbf", probability=True, random_state=666)
    svm_nu_clf = NuSVC(nu=0.7, kernel="rbf", probability=True, random_state=666)
    forest_clf = RandomForestClassifier(criterion="entropy", random_state=666)
    extratrees_clf = ExtraTreesClassifier(criterion="entropy", random_state=666)
    hgb_clf = HistGradientBoostingClassifier(random_state=666)
    bag_clf = BaggingClassifier(random_state=666)
    knn_clf = KNeighborsClassifier(n_neighbors=25)
    gp_clf = GaussianProcessClassifier(random_state=666)

    vo_clf_list = [
        multiclass_test_module(
            VotingClassifier(
                [
                    ("svm", svm_clf),
                    ("svm_nu", svm_nu_clf),
                    ("forest", forest_clf),
                    ("extratrees", extratrees_clf),
                    ("hgb", hgb_clf),
                    ("bag", bag_clf),
                    ("knn", knn_clf),
                    ("gp", gp_clf),
                ],
                voting="soft",
            ),
            "VotingClassifier_soft",
        )
    ]

    group_names = [
        # "dummy",
        # "linear",
        # "tree",
        "svc_no_norm"
        # "svm_c",
        # "neighbour",
        # "bayes",
        # "mlp",
        # "ensemble",
        # "gaussian",
        # "voting_ori",
    ]
    m_lists = [
        # dummy_clf_list,
        # linear_clf_list,
        # tree_clf_list,
        svc_list,
        # svc_c_list,
        # neighbor_clf_list,
        # bayes_clf_list,
        # mlp_clf_list,
        # ensemble_clf_list,
        # gp_clf_list,
        # vo_clf_list,
    ]

    for group_name, m_list in zip(group_names, m_lists):
        # group_name = "voting"
        # m_list = svc_list
        for i in range(len(m_list)):
            print(INFO + "Starting the test of {}".format(m_list[i].name) + END)
            m_list[i].cv_test(all_sel_feature, 5, X_train, y_train)
            with codecs.open(
                os.path.join("..", "output", "performance", "")
                + "all_"
                + m_list[i].name
                + ".txt",
                "w",
                "utf-8",
            ) as file:
                file.writelines(
                    "The F1 Score result of {} :".format(m_list[i].name) + "\n"
                )
                file.writelines("{}".format(m_list[i].f1) + "\n")
                file.writelines(
                    "{:.9f}\t{:.9f}".format(
                        np.average(m_list[i].f1), np.max(m_list[i].f1)
                    )
                    + "\n"
                )
                file.writelines("\n")
                file.writelines(
                    "The Accuracy result of {} :".format(m_list[i].name) + "\n"
                )
                file.writelines("{}".format(m_list[i].acc) + "\n")
                file.writelines(
                    "{:.9f}\t{:.9f}".format(
                        np.average(m_list[i].acc), np.max(m_list[i].acc)
                    )
                    + "\n"
                )

        plot_result("all", group_name, m_list)
        for i in range(len(m_list)):
            with codecs.open("all_metrics.txt", "a", "utf-8") as file:
                file.writelines(
                    "The F1 Score result of {} :".format(m_list[i].name) + "\n"
                )
                file.writelines(
                    "{:.9f}\t{:.9f}".format(
                        np.average(m_list[i].f1), np.max(m_list[i].f1)
                    )
                    + "\n"
                )
        for i in range(len(m_list)):
            with codecs.open("all_metrics.txt", "a", "utf-8") as file:
                file.writelines(
                    "The Accuracy result of {} :".format(m_list[i].name) + "\n"
                )
                file.writelines(
                    "{:.9f}\t{:.9f}".format(
                        np.average(m_list[i].acc), np.max(m_list[i].acc)
                    )
                    + "\n"
                )


if __name__ == "__main__":
    train()
