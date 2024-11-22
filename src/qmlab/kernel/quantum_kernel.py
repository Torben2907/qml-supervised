from typing import Any, Dict, List, Tuple
import numpy as np
import logging
from abc import abstractmethod, ABC
from pennylane import QNode
from pennylane.operation import Operation
import pennylane as qml
import jax


class QuantumKernel(ABC):
    def __init__(
        self,
        *,
        embedding: Operation = None,
        device: str = "default.qubit",
        enforce_psd: bool = True,
        jit: bool = True,
        max_vmap: int = 250,
        qnode_kwargs: Dict[str, str | None] = {
            "interface": "jax-jit",
            "diff_method": None,
        },
    ) -> None:
        self._embedding = embedding
        self._num_wires = self._embedding.num_wires
        self._device = qml.device(device, wires=self._num_wires)
        self._enforce_psd = enforce_psd
        self._jit = jit
        self._max_vmap = max_vmap
        self._qnode_kwargs = qnode_kwargs

    @abstractmethod
    def build_circuit(self) -> QNode:
        raise NotImplementedError()

    def initialize(
        self, feature_dimension: int, class_labels: List[int] | np.ndarray | None = None
    ) -> None:
        if class_labels is None:
            class_labels = [-1, 1]

        self.classes_ = (
            class_labels.tolist()
            if isinstance(class_labels, np.ndarray)
            else class_labels
        )
        self.n_classes_ = len(self.classes_)
        assert +1 and -1 in self.classes_
        assert self.n_classes_ == 2
        self.num_qubits = feature_dimension

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        if y is None:
            logging.warning(
                "You've passed one state vector to the"
                + f"kernel computation, i. e. evaluating self inner product of {x}."
            )
        raise NotImplementedError(
            "You're trying to call an abstract method of the base quantum kernel class."
        )

    def create_random_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    @property
    def embedding(self) -> Operation:
        return self._embedding

    @embedding.setter
    def embedding(self, embedding: Operation) -> None:
        self._embedding = embedding

    @property
    def num_wires(self) -> int:
        return self._num_wires

    @property
    def device(self) -> Any:
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        self._device = qml.device(device, wires=self._num_wires)

    @property
    def max_vmap(self) -> int:
        return self._max_vmap

    @max_vmap.setter
    def max_vmap(self, max_vmap: int) -> None:
        self._max_vmap = max_vmap

    def _validate_inputs(
        self,
        x: np.ndarray | List[List[float]],
        y: np.ndarray | List[List[float]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        if x.ndim > 2:
            raise ValueError(
                f"{x} must be a one or two-dimensional array but has size {x.ndim}!"
            )

        if x.ndim == 1:
            x = x.reshape(-1, len(x))

        if x.shape[1] != self._num_wires:
            try:
                self._embedding.num_wires = x.shape[1]
            except AttributeError as ae:
                raise ValueError(
                    f"Incompatible dimensions found between {x} and class {self._embedding.name}."
                    f"{x} has {x.shape[1]} but {self._embedding.name} has "
                    f"{self._embedding.num_qubits}."
                ) from ae

        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)

            if y.ndim > 2:
                raise ValueError(
                    f"{y} must be a one or two-dimensional array but has size {y.ndim}!"
                )

            if y.ndim == 1:
                y = y.reshape(-1, len(y))

        return x, y

    def _make_psd(self, kernel_matrix: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eig(kernel_matrix)
        m = v @ np.diag(np.maximum(0, w)) @ v.transpose()
        return m.real
