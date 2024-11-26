from typing import Any, Dict
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from ..exceptions import NotFittedError, QMLabError


class BaseQSVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        svm: Any,
        quantum_kernel: QuantumKernel,
        random_state: int = 42,
        **svm_kwargs,
    ) -> None:
        if quantum_kernel is None:
            raise QMLabError("Parameter `quantum_kernel` must be provided.")
        elif not isinstance(quantum_kernel, QuantumKernel):
            raise QMLabError(
                "Parameter `quantum_kernel` must be of type QuantumKernel."
            )
        self._quantum_kernel = quantum_kernel
        self._random_state = random_state
        self._svm = svm(
            kernel=self._quantum_kernel.evaluate, probability=True, **svm_kwargs
        )
        self.params_: Dict[str, np.ndarray] = {}

    @property
    def quantum_kernel(self) -> QuantumKernel:
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        self._quantum_kernel = quantum_kernel
        self._svm.kernel = self._quantum_kernel.evaluate

    @property
    def random_state(self) -> int:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: int) -> None:
        self._random_state = random_state

    def check_if_fitted(self) -> None:
        if "X_train" not in self.params_:
            raise NotFittedError(
                "Model needs to be fitted on the data before evaluating it."
            )

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "BaseQSVM":
        self._svm.random_state = self._random_state
        self.params_ = {"X_train": X}
        self._quantum_kernel.initialize_params(
            feature_dimension=X.shape[1], class_labels=np.unique(y).tolist()
        )
        self._svm.fit(X, y, sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.check_if_fitted()
        return self._svm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.check_if_fitted()
        return self._svm.predict_proba(X)

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return self._svm.score(X, y, sample_weight)


class QSVC(BaseQSVM):
    def __init__(
        self, quantum_kernel: QuantumKernel, random_state: int = 42, **svm_kwargs
    ):
        super().__init__(SVC, quantum_kernel, random_state, **svm_kwargs)
