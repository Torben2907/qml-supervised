import os
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..preprocessing import parse_biomed_data_to_ndarray
import plotly.express as px

plt.style.use("dark_background")
os.makedirs("figures/", exist_ok=True)

color_palette = sns.color_palette("bright")


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
        marker="X",
        label=r"$+1 \, Test$",
    )
    plt.scatter(
        X_test_neg[:, 0],
        X_test_neg[:, 1],
        c=np.array(color_palette[1]).reshape(1, -1),
        marker="X",
        label=r"$-1 \, Test$",
    )

    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20, rotation=0)

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

    plt.show()


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
        color=np.array(color_palette[0]).reshape(1, -1),
        label=r"$+1 \, Training$",
    )
    ax.scatter3D(
        X_train_neg[:, 0],
        X_train_neg[:, 1],
        X_train_neg[:, 2],
        marker=".",
        color=np.array(color_palette[1]).reshape(1, -1),
        label=r"$-1 \, Training$",
    )
    ax.scatter3D(
        X_test_pos[:, 0],
        X_test_pos[:, 1],
        X_test_pos[:, 2],
        marker="x",
        color=np.array(color_palette[0]).reshape(1, -1),
        label=r"$+1 \, Test$",
    )
    ax.scatter3D(
        X_test_neg[:, 0],
        X_test_neg[:, 1],
        X_test_neg[:, 2],
        marker="x",
        color=np.array(color_palette[1]).reshape(1, -1),
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
