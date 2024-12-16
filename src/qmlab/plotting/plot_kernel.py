import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import set_figure_params


def linear_kernel(X_i, X_j):
    # initialize gram matrix G with zeros
    G = np.zeros(shape=(len(X_i), len(X_j)))
    for i, x_i in X_i:
        for j, x_j in X_j:
            G[i, j] = np.dot(x_i, x_j)
    return G


def radial_basis_function_kernel(X_i, X_j, sigma: float = 2.0):
    gamma = 1 / (2 * sigma**2)
    # initialize gram matrix G with zeros
    G = np.zeros(shape=(X_i, X_j))
    for i, x_i in X_i:
        for j, x_j in X_j:
            G[i, j] = np.exp(-gamma * np.linalg.norm(x_i - x_j) ** 2)
    return G


def plot_kernel_heatmap(kernel_train: np.ndarray, kernel_test: np.ndarray):
    set_figure_params()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(
        np.asmatrix(kernel_train), interpolation="nearest", origin="upper", cmap="Blues"
    )
    axs[0].set_title(r"$\hat{\boldsymbol{K}}$ for Training.")

    axs[1].imshow(
        np.asmatrix(kernel_test), interpolation="nearest", origin="upper", cmap="Reds"
    )
    axs[1].set_title(r"$\hat{\boldsymbol{K}}$ for Testing.")

    plt.show()
