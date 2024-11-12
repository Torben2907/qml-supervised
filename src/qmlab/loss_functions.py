# Loss functions for the quantum kernel optimzation.
# Added visualization method: the loss function as a function of the first
# variational parameter.

from .kernel.kernel_loss import KernelLoss
from .kernel.quantum_kernel import QuantumKernel
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Sequence


class LossPlot:
    def plot(
        self,
        quantum_kernel: QuantumKernel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: np.ndarray,
        grid: list = [0.1, 8, 50],
        show: bool = True,
    ) -> np.ndarray:
        kernel_loss = partial(
            self.evaluate,
            quantum_kernel=quantum_kernel,
            data=X_train,
            labels=y_train,
        )

        theta = np.linspace(grid[0], grid[1], int(grid[2]))
        loss_values = np.zeros(len(theta))

        for i, val in enumerate(theta):
            params[0] = val
            loss_values[i] = kernel_loss(params)

        if show:
            plt.rcParams["font.size"] = 15
            plt.figure(figsize=(8, 4))
            plt.plot(theta, loss_values)
            plt.xlabel("θ[0]")
            plt.ylabel("Kernel Loss")
            plt.show()

        return loss_values


class SVCLoss(KernelLoss, LossPlot):
    """
    User defined Kernel Loss class that can be used to modify the loss function and
    output it for plotting.
    Adopted from https://github.com/qiskit-community/prototype-quantum-kernel-training/blob/main/docs/how_tos/create_custom_kernel_loss_function.ipynb
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def evaluate_kernel(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Evaluate the SVC loss of a trainable quantum kernel.
        """
        # Bind the user parameter values
        kernel_matrix = quantum_kernel.evaluate_kernel(data)

        # Train a quantum support vector classifier
        svc = SVC(kernel="precomputed", **self.kwargs)
        svc.fit(kernel_matrix, labels)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Prune kernel matrix of non-support-vector entries
        kmatrix = kernel_matrix[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (
            0.5 * (dual_coefs.T @ kmatrix @ dual_coefs)
        )

        return loss
