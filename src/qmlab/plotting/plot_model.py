import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC
from .plot_utils import set_figure_params


def plot_decision_boundaries(
    clf: SVC,
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
) -> None:
    # Create a 10x10 mesh in the data plane
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    margin = 0.2
    XX, YY = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, 10),
        np.linspace(y_min - margin, y_max + margin, 10),
    )

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z_qke = Z.reshape(XX.shape)

    # visualize the decision function and boundary
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    set_figure_params()
    plt.contourf(XX, YY, Z_qke, vmin=-1.0, vmax=1.0, levels=20, cmap="plasma", alpha=1)
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        marker="o",
        s=100,
        c=np.array(y_train, dtype=np.float32),
        cmap="plasma",
        edgecolor="k",
    )
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker="X",
        s=100,
        c=np.array(y_test, dtype=np.float32),
        cmap="plasma",
        edgecolor="k",
    )
    plt.contour(
        XX,
        YY,
        Z_qke,
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
        levels=[-0.2, 0, 0.2],
    )
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.tight_layout()
    plt.show()
