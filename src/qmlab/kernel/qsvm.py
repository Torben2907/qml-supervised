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

    def create_random_key(self) -> jnp.ndarray:
        return jax.random.key(np.random.default_rng().integers(self.random_state))

    @abstractmethod
    def build_circuit(self) -> QNode:
        raise NotImplementedError()

    @abstractmethod
    def initialize_params(
        self, feature_dimension: int, class_labels: np.ndarray
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVC":
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    # @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
