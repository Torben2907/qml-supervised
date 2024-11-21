from typing import Dict
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from pennylane import QNode


from abc import ABC, abstractmethod


class QSVC(ABC, BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        svm=SVC(kernel="precomputed", probability=True),
        reps: int = 2,
        C: float = 1.0,
        jit: bool = False,
        random_state: int = 42,
        max_vmap: int = 250,
        dev_type: str = "default.qubit",
        qnode_kwargs: dict[str, str | None] = {
            "interface": "jax-jit",
            "diff_method": None,
        },
    ) -> None:
        self.svm = svm
        self.reps = reps
        self.C = C
        self.jit = jit
        self.random_state = random_state
        self.max_vmap = max_vmap
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.circuit = None

        self.parameters: Dict[str, np.ndarray] = {}

    def create_random_key(self) -> jnp.ndarray:
        return jax.random.key(np.random.default_rng().integers(self.random_state))

    @abstractmethod
    def build_circuit(self) -> QNode:
        raise NotImplementedError()

    def initialize_params(
        self, feature_dimension: int, class_labels: np.ndarray | None
    ) -> None:
        if class_labels is None:
            class_labels = np.asarray([-1, 1])

        self.classes_ = class_labels
        self.num_classes = len(self.classes_)
        assert self.num_classes == 2, "Only binary classification supported."
        assert (
            -1 in self.classes_ and +1 in self.classes_
        ), "labels must be in {-1, +1}!"
        self.num_qubits = feature_dimension

        self.build_circuit()

    @abstractmethod
    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVC":
        raise NotImplementedError()

    def _check_fitted(self) -> None:
        if "X_train" not in self.parameters:
            raise ValueError(
                "Model cannot predict without being fitted on the data beforehand."
            )

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
