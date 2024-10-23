import numpy as np


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
    G = np.zeros(shape=len(X_i, X_j))
    for i, x_i in X_i:
        for j, x_j in X_j:
            G[i, j] = np.exp(-gamma * np.linalg.norm(x_i - x_j) ** 2)
    return G
