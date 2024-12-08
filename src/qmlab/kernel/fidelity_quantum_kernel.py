from typing import List
import jax
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import pennylane as qml
from .quantum_kernel import QuantumKernel
from pennylane import QNode
from pennylane.operation import Operation
from pennylane.measurements import ProbabilityMP
from jax.sharding import PartitionSpec as P, NamedSharding, PositionalSharding
from .kernel_utils import vmap_batch, mesh_sharding
from ..exceptions import InvalidEmbeddingError, QMLabError

# need to put this here for computation on a GPU
jax.config.update("jax_default_matmul_precision", "highest")


class FidelityQuantumKernel(QuantumKernel):
    r"""Fidelity Quantum Kernel.

    It is defined as the overlap of two pure quantum states (fidelity):

    .. math::

        \hat{\kappa}(\boldsymbol{x}, \boldsymbol{x}')
        = \left|\Braket{\psi(\boldsymbol{x})|\psi(\boldsymbol{x}')}\right|^2.

    These quantum states are created by applying a parameterized
    unitary to the ground state of a quantum circuit:

    ..math::
        U(\boldsymbol{x}) \Ket{0} = \Ket{\psi({\boldsymbol{x})},
        U(\boldsymbol{x}') \Ket{0} = \Ket{\psi(\boldsymbol{x})}.

    For a detailed introduction to quantum kernels we refer to the main paper.

    Parameters
    ----------
    QuantumKernel : Abstract class for a quantum kernel.
        FidelityQuantumKernel inherits a lot of the functionality
        from QuantumKernel.
    """

    def __init__(
        self,
        *,
        data_embedding: Operation | str,
        device_type: str = "default.qubit",
        reps: int = 2,
        rotation: str = "Z",
        enforce_psd: bool = False,
        jit: bool = True,
        max_batch_size: int = 256,
        evaluate_duplicates: str = "off_diagonal",
        interface: str = "jax",
    ):
        """Constructor of FidelityQuantumKernel.

        Parameters
        ----------
        data_embedding : Operation | str
            Data embedding just like we described in the main text.
        device_type : str, optional
            Quantum Device used for the simulation of the computations, by default "default.qubit"
        reps : int, optional
            Number of repitions by embeddings like IQP, by default 2. For embeddings
            that don't specify a number of repititions like Angle this is ignored.
        rotation : str, optional
            The Axis about which the rotation will occur, by default "Z".
            Use:
            - "X" for the R_x gates,
            - "Y" for the R_y gates,
            - "Z" for the R_z gates.
            For embeddings that don't specify rotations, this is ignored.
        enforce_psd : bool, optional
            Make gram matrix positive semi-definite, by default False
        jit : bool, optional
            Use JAX's just-in-time compilation, by default True
        max_batch_size : int, optional
            Maximum batch size that a JAX vmap function can process in a single call, by default 256
                - Too small values will result to overhead
                - Too large values may exceed memory or computational limits
        evaluate_duplicates : str, optional
            whether (in the case of the gram matrix) the symmetric values
            should be computed too, by default "off_diagonal"
        interface : str, optional
            Interface used for the evaluation of kernel entries, by default "jax"

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__(
            data_embedding=data_embedding,
            device_type=device_type,
            reps=reps,
            rotation=rotation,
            enforce_psd=enforce_psd,
            jit=jit,
            max_batch_size=max_batch_size,
            interface=interface,
        )
        evaluate_duplicates = evaluate_duplicates.lower()
        if evaluate_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Value {evaluate_duplicates} isn't supported for attribute `eval_duplicates`!"
            )
        self._evaluate_duplicates = evaluate_duplicates

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
        InvalidEmbeddingError
            When an invalid embedding has been provided by the user.
            Check `self._available_embeddings` for an overview of
            all embeddings that are currently implemented.
        """
        if class_labels is None:
            class_labels = [-1, 1]

        self.classes_ = class_labels
        self.n_classes_ = len(self.classes_)
        assert +1 and -1 in self.classes_
        assert self.n_classes_ == 2

        if (
            self._data_embedding == qml.IQPEmbedding
            or self._data_embedding == qml.AngleEmbedding
        ):
            self.num_qubits = feature_dimension
        elif self._data_embedding == qml.AmplitudeEmbedding:
            if feature_dimension == 1:
                self.num_qubits = 1
            else:
                num_qubits_ae = int(np.ceil(np.log2(feature_dimension)))
                num_qubits = 2 ** int(np.ceil(np.log2(num_qubits_ae)))
                self.num_qubits = num_qubits
        else:
            raise InvalidEmbeddingError("Invalid embedding. Stop.")

    def build_circuit(self) -> QNode:
        """Builds the quantum circuit for computing the
        fidelity.

        This is based on Algorithm 1 of the thesis.

        Returns
        -------
        QNode
            Object from PennyLane. To cite their documentation:
            >>> A quantum node contains a quantum function [...] and the computational device it is
            executed on. (https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html)
            We define a circuit function, i.e. what the
            PennyLane devs refer to as quantum function, as well as a PennyLane-Device,
            which the user can specify via setting the `device_str` parameter
            in the construction of the fidelity quantum kernel.

        Raises
        ------
        QMLabError
            Thrown when the `initialize_params`-method hasn't been called before
            constructing the circuit and therefore data-dependent
            attributes haven't been set yet.
        """
        self.device = qml.device(self._device_type, wires=self.num_qubits)

        @qml.qnode(self.device, interface=self.interface, diff_method=None)
        def circuit(combined_input: jax.Array) -> ProbabilityMP:
            if self.num_qubits is None:
                raise QMLabError(
                    "Number of qubits hasn't been specified before building the circuit!"
                )
            match self._data_embedding:
                case qml.AngleEmbedding:
                    # noinspection PyCallingNonCallable
                    self._data_embedding(
                        features=combined_input[: self.num_qubits],
                        wires=range(self.num_qubits),
                        rotation=self.rotation,
                    )
                    # noinspection PyCallingNonCallable
                    qml.adjoint(
                        self._data_embedding(
                            features=combined_input[self.num_qubits :],
                            wires=range(self.num_qubits),
                            rotation=self.rotation,
                        )
                    )
                case qml.IQPEmbedding:
                    # noinspection PyCallingNonCallable
                    self._data_embedding(
                        features=combined_input[: self.num_qubits],
                        wires=range(self.num_qubits),
                        n_repeats=self.reps,
                    )
                    # noinspection PyCallingNonCallable
                    qml.adjoint(
                        self._data_embedding(
                            features=combined_input[self.num_qubits :],
                            wires=range(self.num_qubits),
                            n_repeats=self.reps,
                        )
                    )
                case qml.AmplitudeEmbedding:
                    # noinspection PyCallingNonCallable
                    self._data_embedding(
                        features=combined_input[: 2**self.num_qubits],
                        normalize=True,
                        wires=range(self.num_qubits),
                    )
                    # noinspection PyCallingNonCallable
                    qml.adjoint(
                        self._data_embedding(
                            features=combined_input[2**self.num_qubits :],
                            normalize=True,
                            wires=range(self.num_qubits),
                        )
                    )
            return qml.probs()

        self.circuit = circuit
        if self._jit:
            circuit = jax.jit(circuit)

        return circuit

    def evaluate(self, x: NDArray, y: NDArray) -> NDArray:
        """Returns the quantum kernel matrix.
        For x = y this is precisely the quantum gram matrix we refer to in the main text.
        In the context of this study we only use x = y.

        Parameters
        ----------
        x : NDArray
            Dataset of shape (m, d), where m denotes the number of examples
            and d the number of features.
        y : NDArray
            Dataset of shape (m', d'), where m' denotes the number of examples
            and d' the number of features.


        Returns
        -------
        NDArray
            Quantum Kernel Matrix. In our case Quantum Gram matrix of shape
            (m, m), where m denotes the number of examples.
        """
        x, y = self._validate_inputs(x, y)
        kernel_matrix_shape = (
            len(x),
            len(y) if y is not None else len(x),
        )

        combined_input = jnp.array(
            [np.concatenate((x[i], y[j])) for i in range(len(x)) for j in range(len(y))]
        )

        # for faster computation run on multiple GPUs
        sharding = PositionalSharding(jax.devices())
        num_devices = jax.local_device_count()
        sharded_input = jax.device_put(combined_input, sharding.reshape(1, num_devices))

        circuit = self.build_circuit()
        self.batched_circuit = vmap_batch(
            jax.vmap(circuit, 0), start=0, max_batch_size=self._max_batch_size
        )

        # we are only interested in measuring |0^n>
        # n refers to the number of qubits.
        kernel_values = self.batched_circuit(combined_input)[:, 0]
        kernel_matrix = np.reshape(kernel_values, kernel_matrix_shape)

        if self._enforce_psd:
            kernel_matrix = self.make_psd(kernel_matrix)

        return kernel_matrix

    def _is_trivial(
        self, i: int, j: int, psi_i: NDArray, phi_j: NDArray, symmetric: bool
    ) -> bool:
        """This method is experimental and hasn't been tested (yet),
        neither is it used for the study (for now).
        The idea is to simplify computation by computing only the upper or
        lower triangle in the case of the gram matrix.
        For now computations are working just fine on the small-scale
        datasets we use and number of qubits we simulate.

        Parameters
        ----------
        i : int
            _description_
        j : int
            _description_
        psi_i : NDArray
            _description_
        phi_j : NDArray
            _description_
        symmetric : bool
            _description_

        Returns
        -------
        bool
            _description_
        """
        if self._evaluate_duplicates == "all":
            return False
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True
        if np.array_equal(psi_i, phi_j) and self._evaluate_duplicates == "none":
            return True
        return False

    @property
    def evaluate_duplicates(self) -> str:
        return self._evaluate_duplicates

    @evaluate_duplicates.setter
    def evaluate_duplicates(self, evaluate_duplicates: str) -> None:
        self._evaluate_duplicates = evaluate_duplicates
