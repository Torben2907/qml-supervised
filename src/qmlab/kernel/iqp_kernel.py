import time
import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.svm import SVC
from qmlab.utils import chunk_vmapped_fn
from pennylane.measurements import ProbabilityMP
from pennylane import QNode
from sklearn.kernel_approximation import Nystroem
from .qsvm import QSVC

jax.config.update("jax_enable_x64", True)


class FidelityIQPKernel(QSVC):
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
        nystroem: bool = False,
    ) -> None:
        r"""
        Kernel version of the classifier from https://arxiv.org/pdf/1804.11326v2.pdf.
        The kernel function is given by

        .. math::
            k(x,x')=\vert\langle 0 \vert U^\dagger(x')U(x)\vert 0 \rangle\vert^2

        where :math:`U(x)` is an IQP circuit implemented via Pennylane's `IQPEmbedding`.

        We precompute the kernel matrix from the data directly, and pass it to scikit-learn's support vector
        classifier SVC. This  allows us to benefit from JAX parallelisation when computing the kernel
        matrices.

        The model requires evaluating a number of circuits given by the square of the number of data
        samples, and is therefore only appropriate for relatively small datasets.

        Args:
            svm (sklearn.svm.SVC): scikit-learn SVM class object used to fit the model from the kernel matrix
            repeats (int): number of times the IQP structure is repeated in the embedding circuit.
            C (float): regularization parameter for SVC. Lower values imply stronger regularization.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            qnode_kwargs (str): the key word arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): seed used for reproducibility.
        """
        # attributes that do not depend on data
        self.reps = reps
        self.C = C
        self.jit = jit
        self.max_vmap = max_vmap
        self.svm = svm
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.nystroem = nystroem
        # data-dependant attributes
        # that will be initialised by calling "fit"
        self.circuit = None

    def create_random_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def build_circuit(self) -> QNode:
        dev = qml.device(self.dev_type, wires=self.num_qubits)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(combined_inputs: np.ndarray) -> ProbabilityMP:
            """
            circuit used to precomute the kernel matrix K(x_1,x_2).
            Args:
                x (np.array): vector of length 2*num_feature that is the concatenation of x_1 and x_2

            Returns:
                (float) the value of the kernel fucntion K(x_1,x_2)
            """
            qml.IQPEmbedding(
                combined_inputs[: self.num_qubits],
                wires=range(self.num_qubits),
                n_repeats=self.reps,
            )
            qml.adjoint(
                qml.IQPEmbedding(
                    combined_inputs[self.num_qubits :],
                    wires=range(self.num_qubits),
                    n_repeats=self.reps,
                )
            )
            return qml.probs()

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)
        return circuit

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """
        compute the kernel matrix relative to data sets X1 and X2
        Args:
            X1 (np.array): first dataset of input vectors
            X2 (np.array): second dataset of input vectors
        Returns:
            kernel_matrix (np.array): matrix of size (len(X1),len(X2)) with elements K(x_1,x_2)
        """
        # these are both data-dependent
        left_inputs = len(x_vec)
        right_inputs = len(y_vec)

        # concatenate all pairs of vectors
        Z = jnp.array(
            [
                np.concatenate((x_vec[i], y_vec[j]))
                for i in range(left_inputs)
                for j in range(right_inputs)
            ]
        )

        circuit = self.build_circuit()
        self.batched_circuit = chunk_vmapped_fn(
            jax.vmap(circuit, 0), start=0, max_vmap=self.max_vmap
        )

        # remember from the derivation in the thesis,
        # we are only interested in measuring |0>
        kernel_values = self.batched_circuit(Z)[:, 0]

        kernel_matrix_shape = (left_inputs, right_inputs)
        kernel_matrix = np.reshape(kernel_values, kernel_matrix_shape)

        return kernel_matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FidelityIQPKernel":
        """Fit the model to data X and labels y. Uses sklearn's SVM classifier

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.svm.random_state = self.rng.integers(100000)
        self.svm.C = self.C

        self.initialize_params(X.shape[1], np.unique(y))
        self.parameters = {"X_train": X}
        gram_matrix = self.evaluate(X, X)

        start = time.time()

        if self.nystroem is True:
            nyst = Nystroem(kernel="precomputed", n_components=500)
            reduced_gram_matrix = nyst.fit_transform(gram_matrix)
            self.svm.fit(reduced_gram_matrix, y)
        else:
            self.svm.fit(gram_matrix, y)

        end = time.time()

        self.training_time = end - start

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        self._check_fitted()
        kernel_matrix = self.evaluate(X, self.parameters["X_train"])
        return self.svm.predict(kernel_matrix)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict label probabilities for data X.
        note that this may be inconsistent with predict; see the sklearn docummentation for details.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        self._check_fitted()
        kernel_matrix = self.evaluate(X, self.parameters["X_train"])
        return self.svm.predict_proba(kernel_matrix)
