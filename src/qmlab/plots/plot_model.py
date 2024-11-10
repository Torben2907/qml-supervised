import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


def plot_decision_boundaries(
    clf: ClassifierMixin | BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """Visualize decision function, boundary, and margins of +- 0.2"""

    # Create a 10x10 mesh in the data plan
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    margin = 0.2
    XX, YY = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, 10),
        np.linspace(y_min - margin, y_max + margin, 10),
    )

    # Calculate the decision function value on the 10x10 mesh
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z_qke = Z.reshape(XX.shape)

    # visualize the decision function and boundary
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    set_figure_params()
    plt.figure(figsize=(7, 7))
    plt.contourf(
        XX, YY, Z_qke, vmin=-1.0, vmax=1.0, levels=20, cmap=plt.cm.coolwarm, alpha=1
    )
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        marker="o",
        s=200,
        c=plt.cm.coolwarm(np.array(y_train, dtype=np.float32)),
        edgecolor="k",
    )
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker="X",
        s=200,
        c=plt.cm.coolwarm(np.array(y_test, dtype=np.float32)),
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


def set_figure_params():
    """Set output figure parameters"""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "lines.linewidth": 2,
            "axes.titlesize": 24,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.markersize": 10,
        }
    )
