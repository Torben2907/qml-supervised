from typing import Any
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseQSVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        svm: Any,
        quantum_kernel: QuantumKernel,
        **kwargs,
    ):
        if quantum_kernel is None:
            raise ValueError("`quantum_kernel` must be provided.")
        self._quantum_kernel = quantum_kernel
        self._svm = svm(
            kernel=self._quantum_kernel.evaluate, probability=True, **kwargs
        )

    @property
    def quantum_kernel(self) -> QuantumKernel:
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        self._quantum_kernel = quantum_kernel
        self._svm.kernel = self._quantum_kernel.evaluate

    def _check_fitted(self) -> None:
        if "X_train" not in self._params:
            raise ValueError("Model needs to be fitted before you can evaluate it.")

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        self._params = {"X_train": X}
        self._quantum_kernel.initialize(X.shape[1], np.unique(y))
        self._svm.fit(X, y, sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._svm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._svm.predict_proba(X)

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return self._svm.score(X, y, sample_weight)


class QSVC(BaseQSVM):
    def __init__(self, quantum_kernel: QuantumKernel, **kwargs):
        super().__init__(SVC, quantum_kernel, **kwargs)
