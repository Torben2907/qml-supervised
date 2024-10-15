import os
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures/", exist_ok=True)

matplotlib.rcParams["mathtext.fontset"] = "stix"
sns.set_theme(font_scale=1.3)
sns.set_style("white")

color_palette = sns.color_palette("deep")

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


def plot_2d_data(
    X: np.ndarray,
    y: np.ndarray,
    data_name: Optional[str] = None,
    save_plot: bool = False,
):
    assert X.shape[1] == 2
    assert len(np.unique(y)) == 2
    assert +1 in y
    assert -1 in y

    fig = plt.figure()
    ax = plt.gca()

    X_pos = X[y == +1]
    X_neg = X[y == -1]

    plt.scatter(
        X_pos[:, 0],
        X_pos[:, 1],
        c=np.array(color_palette[0]).reshape(1, -1),
        marker=".",
    )
    plt.scatter(
        X_neg[:, 0],
        X_neg[:, 1],
        c=np.array(color_palette[1]).reshape(1, -1),
        marker=".",
    )

    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20)
    plt.tight_layout()

    if save_plot:
        if data_name is None:
            raise ValueError("cannot write file when data_name isn't provided.")
        plt.savefig(f"figures/{data_name}" + "_plot", dpi=300)

    plt.show()


def plot_2d_data_with_train_test_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_name: Optional[str] = None,
    save_plot: bool = False,
):
    assert X_train.shape[1] == 2 and X_test.shape[1] == 2
    assert len(np.unique(y_train)) == 2 and len(np.unique(y_test)) == 2
    assert +1 in y_train and -1 in y_train
    assert +1 in y_test and -1 in y_test

    X_train_pos = X_train[y_train == +1]
    X_train_neg = X_train[y_train == -1]
    X_test_pos = X_test[y_test == +1]
    X_test_neg = X_test[y_test == -1]

    plt.scatter(
        X_train_pos[:, 0],
        X_train_pos[:, 1],
        c=np.array(color_palette[0]).reshape(1, -1),
        marker=".",
        label=r"$+1 \, Training$",
    )
    plt.scatter(
        X_train_neg[:, 0],
        X_train_neg[:, 1],
        c=np.array(color_palette[1]).reshape(1, -1),
        marker=".",
        label=r"$-1 \, Training$",
    )
    plt.scatter(
        X_test_pos[:, 0],
        X_test_pos[:, 1],
        c=np.array(color_palette[0]).reshape(1, -1),
        marker="x",
        label=r"$+1 \, Test$",
    )
    plt.scatter(
        X_test_neg[:, 0],
        X_test_neg[:, 1],
        c=np.array(color_palette[1]).reshape(1, -1),
        marker="x",
        label=r"$-1 \, Test$",
    )

    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20)

    if save_plot:
        if data_name is None:
            raise ValueError("cannot write file when data_name isn't provided.")
        # legend is only going to be displayed fully when rendered to .png
        # so I only include it here
        plt.legend(bbox_to_anchor=(1.05, 1), loc="best", borderaxespad=0.0)
        plt.savefig(
            f"figures/{data_name}" + "_train_test_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
