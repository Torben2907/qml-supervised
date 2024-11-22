import logging
from abc import abstractmethod, ABC
from typing import Any, List, Tuple

import jax
import numpy as np
import pennylane as qml
from pennylane import QNode
from pennylane.operation import Operation


class QuantumKernel(ABC):
    def __init__(
        self,
        *,
        embedding: Operation = None,
        device: str = "default.qubit",
        enforce_psd: bool = False,
        jit: bool = True,
        max_vmap: int = 250,
        interface: str = "jax-jit",
    ) -> None:
        self._embedding = embedding
        self._num_wires = self._embedding.num_wires
        self._device = qml.device(device, wires=self._num_wires)
        self._enforce_psd = enforce_psd
        self._jit = jit
        self._max_vmap = max_vmap
        self.interface = interface

        self.classes_: List[int] | None = None
        self.n_classes_: int | None = None

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
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> None:
        if y is None:
            logging.warning(
                "You've passed one state vector to the"
                + f"kernel computation, i. e. evaluating self inner product of {x}."
            )
        raise NotImplementedError(
            "You're trying to call an abstract method of the base quantum kernel class."
        )

    @staticmethod
    def create_random_key():
        return jax.random.PRNGKey(np.random.default_rng().integers(1000000))

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
        y: np.ndarray | List[List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = self.check_type_and_dimension(x)
        if x.shape[1] != self._num_wires:
            try:
                self._embedding.num_wires = x.shape[1]
            except AttributeError as ae:
                raise ValueError(
                    f"Incompatible dimensions found between {x} and class {self._embedding.name}."
                    f"{x} has {x.shape[1]} but {self._embedding.name} has "
                    f"{self._embedding.num_wires}."
                ) from ae

        if y is not None:
            y = self.check_type_and_dimension(y)
        return x, y

    @staticmethod
    def check_type_and_dimension(data: np.ndarray | List[List[float]]) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim > 2:
            raise ValueError(
                f"{data} must be a one or two-dimensional array but has dimension: {data.ndim}!"
            )
        if data.ndim == 1:
            data = data.reshape(-1, len(data))
        return data

    @staticmethod
    def make_psd(kernel_matrix: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eig(kernel_matrix)
        m = v @ np.diag(np.maximum(0, w)) @ v.transpose()
        return m.real
