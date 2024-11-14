from abc import ABC, abstractmethod
from functools import partial
from typing import Sequence
import numpy as np
from .trainable_kernel import TrainableQuantumKernel
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class KernelLoss(ABC):
    def __call__(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableQuantumKernel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        return self.evaluate_kernel(parameter_values, quantum_kernel, X, y)

    @abstractmethod
    def evaluate_kernel(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableQuantumKernel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        raise NotImplementedError

    def plot_loss(
        self,
        quantum_kernel: TrainableQuantumKernel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        parameter: Sequence[float],
        grid: list = [0.1, 8, 50],
        show: bool = True,
    ) -> np.ndarray:

        kernel_loss = partial(
            self.evaluate_kernel,
            quantum_kernel=quantum_kernel,
            X=X_train,
            y=y_train,
        )

        theta = np.linspace(grid[0], grid[1], int(grid[2]))
        loss_values = np.zeros(len(theta))

        for i, val in enumerate(theta):
            parameter[0] = val
            loss_values[i] = kernel_loss(parameter)

        if show:
            plt.rcParams["font.size"] = 15
            plt.figure(figsize=(8, 4))
            plt.plot(theta, loss_values)
            plt.xlabel("θ[0]")
            plt.ylabel("Kernel Loss")
            plt.show()

        return loss_values


class SVCLoss(KernelLoss):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def evaluate_kernel(
        self,
        parameter_values: np.ndarray,
        quantum_kernel: TrainableQuantumKernel,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Evaluate the SVC loss of a trainable quantum kernel.
        """
        # Bind the user parameter values
        quantum_kernel.assign_training_parameters(parameter_values)
        kernel_matrix = quantum_kernel.evaluate_kernel(X)

        # Train a quantum support vector classifier
        svc = SVC(kernel="precomputed", **self.kwargs)
        svc.fit(kernel_matrix, y)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Prune kernel matrix of non-support-vector entries
        kernel_matrix = kernel_matrix[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (
            0.5 * (dual_coefs.T @ kernel_matrix @ dual_coefs)
        )

        return loss


class KTALoss(KernelLoss):
    def __init__(self, rescale=True):
        self.rescale = rescale

    def evaluate_kernel(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableQuantumKernel,
        X: np.ndarray,
        y: np.ndarray,
    ):
        quantum_kernel.assign_training_parameters(parameter_values)
        K = quantum_kernel.evaluate_kernel(X)

        if self.rescale:
            nplus = np.count_nonzero(np.array(y) == 1)
            nminus = len(y) - nplus
            _Y = np.array([y_i / nplus if y_i == 1 else y_i / nminus for y_i in y])
        else:
            _Y = np.array(y)

        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        inner_product = inner_product / norm

        return -inner_product
