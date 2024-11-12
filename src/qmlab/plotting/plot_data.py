import os
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .plot_utils import set_figure_params

plt.style.use("dark_background")
os.makedirs("figures/", exist_ok=True)

cmap = sns.color_palette("Spectral")


def plot_2d_data(
    X: np.ndarray,
    y: np.ndarray,
    data_name: Optional[str] = None,
    save_plot: bool = False,
):
    assert X.shape[1] == 2 and X.ndim == 2
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
        c=np.array(cmap[0]).reshape(1, -1),
        s=150,
        marker=".",
    )
    plt.scatter(
        X_neg[:, 0],
        X_neg[:, 1],
        c=np.array(cmap[1]).reshape(1, -1),
        s=150,
        marker="X",
    )

    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20, rotation=0)
    plt.tight_layout()

    if save_plot:
        if data_name is None:
            raise ValueError("cannot save the figure when `data_name` isn't provided.")
        plt.savefig(f"figures/{data_name}" + "_plot", dpi=300)

    plt.show()


def plot_2d_data_with_train_test_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_name: Optional[str] = None,
    separate_plots: bool = False,
    save_plot: bool = False,
):
    assert X_train.shape[1] == 2 and X_test.shape[1] == 2
    assert len(np.unique(y_train)) == 2 and len(np.unique(y_test)) == 2
    assert +1 in y_train and -1 in y_train
    assert +1 in y_test and -1 in y_test

    if separate_plots:
        set_figure_params()
        _, ax = plt.subplots(1, 2, figsize=[11, 5])
        ax[0].scatter(
            X_train[:, 0],
            X_train[:, 1],
            marker="o",
            c=np.array(y_train, dtype=np.float32),
            s=150,
            cmap="plasma",
            edgecolors="k",
        )
        ax[0].set_title("Training Data")
        ax[1].set_title("Test Data")
        ax[1].scatter(
            X_test[:, 0],
            X_test[:, 1],
            marker="X",
            c=np.array(y_test, dtype=np.float32),
            s=150,
            cmap="plasma",
            edgecolors="k",
        )
        for i in range(2):
            ax[i].set_xlabel(r"$x_1$", fontsize=20)
            ax[i].set_ylabel(r"$x_2$", fontsize=20, rotation=0)
        plt.tight_layout()

    else:

        X_train_pos = X_train[y_train == +1]
        X_train_neg = X_train[y_train == -1]
        X_test_pos = X_test[y_test == +1]
        X_test_neg = X_test[y_test == -1]

        plt.scatter(
            X_train_pos[:, 0],
            X_train_pos[:, 1],
            c=np.array(cmap[0]).reshape(1, -1),
            s=150,
            marker="o",
            label=r"$+1 \, Training$",
            edgecolors="k",
        )
        plt.scatter(
            X_train_neg[:, 0],
            X_train_neg[:, 1],
            c=np.array(cmap[1]).reshape(1, -1),
            s=150,
            marker="o",
            label=r"$-1 \, Training$",
            edgecolors="k",
        )
        plt.scatter(
            X_test_pos[:, 0],
            X_test_pos[:, 1],
            c=np.array(cmap[0]).reshape(1, -1),
            s=150,
            marker="X",
            label=r"$+1 \, Test$",
            edgecolors="k",
        )
        plt.scatter(
            X_test_neg[:, 0],
            X_test_neg[:, 1],
            c=np.array(cmap[1]).reshape(1, -1),
            s=200,
            marker="X",
            label=r"$-1 \, Test$",
            edgecolors="k",
        )

        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

    if save_plot:
        if data_name is None:
            raise ValueError("cannot write file when data_name isn't provided.")
        # legend is only going to be displayed fully when rendered to .png
        # so I only include it here
        plt.legend(bbox_to_anchor=(1.05, 1), loc="best", borderaxespad=0.0)
        plt.savefig(
            f"figures/{data_name}" + "_2d_train_test_plot.png",
            dpi=300,
            bbox_inches="tight",
        )


def plot_3d_data_with_train_test_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_name: Optional[str] = None,
    save_plot: bool = False,
) -> None:
    assert (
        X_train.shape[1] == 3 and X_test.shape[1] == 3
    ), f"{X_train} and {X_test} must have exactly 3 features!"
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    X_train_pos = X_train[y_train == +1]
    X_train_neg = X_train[y_train == -1]
    X_test_pos = X_test[y_test == +1]
    X_test_neg = X_test[y_test == -1]

    ax.scatter3D(
        X_train_pos[:, 0],
        X_train_pos[:, 1],
        X_train_pos[:, 2],
        marker=".",
        color=np.array(cmap[0]).reshape(1, -1),
        label=r"$+1 \, Training$",
    )
    ax.scatter3D(
        X_train_neg[:, 0],
        X_train_neg[:, 1],
        X_train_neg[:, 2],
        marker=".",
        color=np.array(cmap[1]).reshape(1, -1),
        label=r"$-1 \, Training$",
    )
    ax.scatter3D(
        X_test_pos[:, 0],
        X_test_pos[:, 1],
        X_test_pos[:, 2],
        marker="x",
        color=np.array(cmap[0]).reshape(1, -1),
        label=r"$+1 \, Test$",
    )
    ax.scatter3D(
        X_test_neg[:, 0],
        X_test_neg[:, 1],
        X_test_neg[:, 2],
        marker="x",
        color=np.array(cmap[1]).reshape(1, -1),
        label=r"$-1 \, Test$",
    )

    ax.set_xlabel("$x_1$", fontsize=25)
    ax.set_ylabel("$x_2$", fontsize=25)
    ax.set_zlabel("$x_3$", fontsize=25)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.labelpad = -20
    ax.yaxis.labelpad = -20
    ax.zaxis.labelpad = -20

    plt.legend()
    if save_plot:
        if data_name is None:
            raise ValueError("cannot write file when data_name isn't provided.")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="best", borderaxespad=0.0)
        plt.savefig(
            f"figures/{data_name}" + "3d_train_test_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
