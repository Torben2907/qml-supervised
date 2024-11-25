import logging
from abc import abstractmethod, ABC
from typing import Any, Callable, List, Tuple

import jax
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pennylane as qml
from pennylane import QNode
from pennylane.operation import Operation
from pennylane.capture import ABCCaptureMeta
from ..exceptions import InvalidEmbeddingError


class QuantumKernel(ABC):
    def __init__(
        self,
        *,
        data_embedding: Operation | str = None,
        device_type: str = "default.qubit",
        enforce_psd: bool = False,
        jit: bool = True,
        max_vmap: int = 250,
        interface: str = "jax-jit",
    ) -> None:
        self._available_embeddings: Tuple[str, ...] = ("Amplitude", "Angle", "IQP")
        self._data_embedding = self.initialize_embedding(data_embedding)
        self._num_wires = self._data_embedding.num_wires
        self._enforce_psd = enforce_psd
        self._jit = jit
        self._max_vmap = max_vmap
        self.interface = interface
        self._device_type = device_type

        self.classes_: List[int] | None = None
        self.n_classes_: int | None = None
        self.num_qubits: int | None = None
        self.batched_circuit: Callable | None = None
        self.circuit: Callable | None = None
        self.device: Any | None = None

    def initialize_embedding(self, embedding: str | Operation) -> Operation:
        if isinstance(embedding, str):
            if embedding not in self._available_embeddings:
                raise InvalidEmbeddingError(
                    f"{embedding} embedding isn't available. Choose from {self._available_embeddings}."
                )
            match embedding:
                case "Amplitude":
                    return qml.AmplitudeEmbedding
                case "Angle":
                    return qml.AngleEmbedding
                case "IQP":
                    return qml.IQPEmbedding
        elif isinstance(embedding, ABCCaptureMeta):
            """actually passing in `qml.operation.Operation` should work here, but when passing
            in the argument without brackets it creates this abstract
            class from the capture module. might have to investigate this further."""
            return embedding
        else:
            raise InvalidEmbeddingError(f"{embedding} is an invalid embedding type.")

    @abstractmethod
    def build_circuit(self) -> QNode:
        raise NotImplementedError()

    @abstractmethod
    def initialize(
        self,
        feature_dimension: int,
        class_labels: List[int] | None = None,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, x: NDArray, y: NDArray) -> None:
        if y is None:
            logging.warning(
                "You've passed one state vector to the"
                + f"kernel computation, i. e. evaluating self inner product of {x}."
            )
        raise NotImplementedError(
            "You're trying to call the `evaluate` method of the base quantum kernel class, which is abstract."
        )

    @staticmethod
    def create_random_key() -> jax.Array:
        return jax.random.PRNGKey(np.random.default_rng().integers(1000000))

    @property
    def data_embedding(self) -> Operation:
        return self._data_embedding

    @data_embedding.setter
    def data_embedding(self, data_embedding: Operation | str) -> None:
        self._data_embedding = data_embedding

    @property
    def num_wires(self) -> int:
        return self._num_wires

    @property
    def device_type(self) -> str:
        return self._device_type

    @device_type.setter
    def device_type(self, device_type: str) -> None:
        self._device_type = device_type

    @property
    def max_vmap(self) -> int:
        return self._max_vmap

    @max_vmap.setter
    def max_vmap(self, max_vmap: int) -> None:
        self._max_vmap = max_vmap

    @property
    def available_embeddings(self) -> Tuple[str, ...]:
        return self._available_embeddings

    def _validate_inputs(
        self,
        x: NDArray | ArrayLike,
        y: NDArray | ArrayLike,
    ) -> Tuple[NDArray, NDArray]:
        x = self._check_type_and_dimension(x)
        if x.shape[1] != self._num_wires:
            try:
                self._data_embedding.num_wires = x.shape[1]
            except AttributeError as ae:
                raise ValueError(
                    f"Incompatible dimensions found between {x} and class {self._data_embedding.name}."
                    f"{x} has {x.shape[1]} but {self._data_embedding.name} has "
                    f"{self._data_embedding.num_wires}."
                ) from ae

        if y is not None:
            y = self._check_type_and_dimension(y)
        return x, y

    @staticmethod
    def _check_type_and_dimension(data: NDArray | ArrayLike) -> NDArray:
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
    def make_psd(kernel_matrix: NDArray) -> NDArray:
        w, v = np.linalg.eig(kernel_matrix)
        m = v @ np.diag(np.maximum(0, w)) @ v.transpose()
        return m.real
