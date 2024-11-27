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
    """An abstract class for Quantum Kernels, every quantum kernel proposed by the thesis will
    inherit from this superclass.

    Parameters
    ----------
    ABC : `abstract base class` used to show that this class cannot be instantiated on its
        own but rather serves as the blue print containing the necessary methods for
        estimating a quantum kernel.
    """

    def __init__(
        self,
        *,
        data_embedding: Operation | str = None,
        device_type: str = "default.qubit",
        reps: int = 2,
        enforce_psd: bool = False,
        jit: bool = True,
        max_batch_size: int = 256,
        interface: str = "jax",
    ) -> None:
        """Constructor of Quantum Kernel. Contains defaults.

        Parameters
        ----------
        data_embedding : Operation | str, optional
            Data Embedding just as described in the text, by default None
        device_type : str, optional
            Quantum Device used for simulation of the computations, by default "default.qubit"
        reps : int, optional
            Number of reps when IQP is used to embed data, by default 2 and will
            be ignored when any other embedding than IQP is specified.
        enforce_psd : bool, optional
            Ensures that gram matrix is positive semi-definite, by default False
        jit : bool, optional
            Activates or deactivates JAX's just-in-time compilation, by default True
        max_batch_size : int, optional
            Maximum batch size that a JAX vmap function can process in a single call, by default 256
            - Too small values will result to overhead
            - Too large values may exceed memory or computational limits
        interface : str, optional
            Interface that will be used for computations, by default "jax"
        """
        self._available_embeddings: Tuple[str, ...] = ("Amplitude", "Angle", "IQP")
        self._data_embedding = self.initialize_embedding(data_embedding)
        self._num_wires = self._data_embedding.num_wires
        self.reps = reps
        self._enforce_psd = enforce_psd
        self._jit = jit
        self._max_batch_size = max_batch_size
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
            class from the capture module."""
            return embedding
        else:
            raise InvalidEmbeddingError(f"{embedding} is an invalid embedding type.")

    @abstractmethod
    def build_circuit(self) -> QNode:
        """All quantum kernels will need a method of how they get computed.
        This is given by a quantum circuit.

        Returns
        -------
        QNode
            Object from PennyLane. To cite their documentation:
            >>> A quantum node contains a quantum function [...] and the computational device it is
            executed on. (https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html)
            All implementations of this method will have to define a circuit function, i.e. what the
            PennyLane devs refer to as quantum function, as well as a PennyLane-Device, which the user
            can specify via setting the `device_str` parameter in the construction of each quantum kernel.
        Raises
        ------
        NotImplementedError
            abstract method, i.e. will be defined individually on each kernel.
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_params(
        self,
        feature_dimension: int,
        class_labels: List[int] | None = None,
    ) -> None:
        """Initialization of the data dependent attributes, like number of features
        and class labels. Depending on which data embedding has been specified by the user
        it will also initialize the number of qubits. When working with a quantum kernel
        it's mandatory to first call this method before trying to evaluate the gram matrix!

        Parameters
        ----------
        feature_dimension : int
            Number of features in the data domain.
        class_labels : List[int] | None, optional
            Class labels, by default None, will be [-1, +1] throughout the study.

        Raises
        ------
        NotImplementedError
            Abstract method
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, x: NDArray, y: NDArray) -> NDArray:
        """Abstract method for the evaluation of the kernel matrix K.

        Parameters
        ----------
        x : NDArray
            Feature matrix of shape (m, d), where
            m is the number of examples, d is the number of features.
        y : NDArray
            Feature matrix of shape (m, d), where
            m is the number of examples, d is the number of features.
        Returns
        -------
        NDArray
            2D Gram matrix of shape (m, m), where m refers to the number of examples.

        Raises
        ------
        NotImplementedError
            Abstract method.
        """
        if np.array_equal(x, y):
            logging.info(
                "You've passed the same vector twice"
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
        return self._max_batch_size

    @max_vmap.setter
    def max_vmap(self, max_vmap: int) -> None:
        self._max_batch_size = max_vmap

    @property
    def available_embeddings(self) -> Tuple[str, ...]:
        return self._available_embeddings

    def _validate_inputs(
        self,
        x: NDArray | ArrayLike,
        y: NDArray | ArrayLike,
    ) -> Tuple[NDArray, NDArray]:
        """Ensures that the arguments for the `evaluate` method are valid.

        Parameters
        ----------
        x : NDArray | ArrayLike
            1D or 2D Array of shape (m, d), where m is the number of examples,
            d the number of features.
            In the 1D case (which will raise a warning so the user is aware)
            the method will try to reshape the 1D array to the 2D array of shape (m, n).
        y : NDArray | ArrayLike
            1D or 2D Matrix of shape (m, d), where m are the number of examples,
            d the number of features.
            The same reshaping rule applies for y too. See in the definition of x.

        Returns
        -------
        Tuple[NDArray, NDArray]
            The validated and possibly reshaped inputs for the `evaluate`-method.

        Raises
        ------
        ValueError
            If incompatible dimensions have been found during the validation procedure.
        """
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
        """Will check if the inputs are NDArrays. If they're not, we try to convert
        them to one (note that this will most likely only work for Iterable Python objects and
        will throw an error otherwise).

        Parameters
        ----------
        data : NDArray | ArrayLike
            A feature vector (1D array), which will be reshaped to fit the dimension (m, d).
            A feature matrix of shape (m, d), where m denotes the number of features
            and d the number of features.

        Returns
        -------
        NDArray
            The validated and possibly reshaped input.

        Raises
        ------
        ValueError
            If the number of dimensions is greater than one, i.e. a 3D or higher dimensional array has
            been provided.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim > 2:
            raise ValueError(
                f"{data} must be a one or two-dimensional array but has dimension: {data.ndim}!"
            )
        if data.ndim == 1:
            data = data.reshape(-1, data.shape[1])
            logging.warning(
                f"You specified a 1D input, that was now reshaped into a 2D array of shape: {data.shape}"
            )
        return data

    @staticmethod
    def make_psd(kernel_matrix: NDArray) -> NDArray:
        w, v = np.linalg.eig(kernel_matrix)
        m = v @ np.diag(np.maximum(0, w)) @ v.transpose()
        return m.real
