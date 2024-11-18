import time
from typing import Callable, List
import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils import gen_batches
from pennylane.measurements import ProbabilityMP

jax.config.update("jax_enable_x64", True)


def chunk_vmapped_fn(vmapped_fn, start: int, max_vmap: int):
    """
    Convert a vmapped function to an equivalent function that evaluates in chunks of size
    max_vmap. The behaviour of chunked_fn should be the same as vmapped_fn, but with a
    lower memory cost.

    The input vmapped_fn should have in_axes = (None, None, ..., 0,0,...,0)

    Args:
        vmapped (func): vmapped function with in_axes = (None, None, ..., 0,0,...,0)
        start (int): The index where the first 0 appears in in_axes
        max_vmap (int) The max chunk size with which to evaluate the function

    Returns:
        chunked version of the function
    """

    def chunked_fn(*args):
        batch_len = len(args[start])
        batch_slices = list(gen_batches(batch_len, max_vmap))
        res = [
            vmapped_fn(*args[:start], *[arg[slice] for arg in args[start:]])
            for slice in batch_slices
        ]
        # jnp.concatenate needs to act on arrays with the same shape, so pad the last array if necessary
        if batch_len / max_vmap % 1 != 0.0:
            diff = max_vmap - len(res[-1])
            res[-1] = jnp.pad(
                res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)]
            )
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn


class IQPKernelClassifier(BaseEstimator, ClassifierMixin):
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
    ):
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
        self.repeats = reps
        self.C = C
        self.jit = jit
        self.max_vmap = max_vmap
        self.svm = svm
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # data-dependant attributes
        # that will be initialised by calling "fit"
        self.parameters = None
        # self.num_qubits = None
        self.circuit = None
        self.training_time = None

    def create_random_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def build_circuit(self):
        dev = qml.device(self.dev_type, wires=self.num_qubits)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(x_vec: np.ndarray) -> ProbabilityMP:
            """
            circuit used to precomute the kernel matrix K(x_1,x_2).
            Args:
                x (np.array): vector of length 2*num_feature that is the concatenation of x_1 and x_2

            Returns:
                (float) the value of the kernel fucntion K(x_1,x_2)
            """
            qml.IQPEmbedding(
                x_vec[: self.num_qubits],
                wires=range(self.num_qubits),
                n_repeats=self.repeats,
            )
            qml.adjoint(
                qml.IQPEmbedding(
                    x_vec[self.num_qubits :],
                    wires=range(self.num_qubits),
                    n_repeats=self.repeats,
                )
            )
            return qml.probs()

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)
        return circuit

    def precompute_kernel(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """
        compute the kernel matrix relative to data sets X1 and X2
        Args:
            X1 (np.array): first dataset of input vectors
            X2 (np.array): second dataset of input vectors
        Returns:
            kernel_matrix (np.array): matrix of size (len(X1),len(X2)) with elements K(x_1,x_2)
        """
        left_parameters = len(x_vec)
        right_parameters = len(y_vec)

        # concatenate all pairs of vectors
        Z = jnp.array(
            [
                np.concatenate((x_vec[i], y_vec[j]))
                for i in range(left_parameters)
                for j in range(right_parameters)
            ]
        )

        circuit = self.build_circuit()
        self.batched_circuit = chunk_vmapped_fn(
            jax.vmap(circuit, 0), start=0, max_vmap=self.max_vmap
        )
        kernel_values = self.batched_circuit(Z)[:, 0]

        kernel_matrix_shape = (left_parameters, right_parameters)
        kernel_matrix = np.reshape(kernel_values, kernel_matrix_shape)
        return kernel_matrix

    def initialize(
        self, feature_dimension: int, class_labels: List[int] | None = None
    ) -> None:
        """Initialize attributes that depend on the number of features and the class labels.

        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """
        if class_labels is None:
            class_labels = [-1, 1]

        self.classes_ = class_labels
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_
        self.num_qubits = feature_dimension

        self.build_circuit()

    def fit(self, X, y):
        """Fit the model to data X and labels y. Uses sklearn's SVM classifier

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.svm.random_state = self.rng.integers(100000)

        self.initialize(X.shape[1], np.unique(y))
        self.parameters = {"x_train": X}
        kernel_matrix = self.precompute_kernel(X, X)

        start = time.time()
        # we are updating this value here, in case it was
        # changed after initialising the model
        self.svm.C = self.C
        self.svm.fit(kernel_matrix, y)
        end = time.time()
        self.training_time = end - start

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        kernel_matrix = self.precompute_kernel(X, self.parameters["x_train"])
        return self.svm.predict(kernel_matrix)

    def predict_proba(self, X):
        """Predict label probabilities for data X.
        note that this may be inconsistent with predict; see the sklearn docummentation for details.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """

        if "x_train" not in self.parameters:
            raise ValueError("Model cannot predict without fitting to data first.")

        kernel_matrix = self.precompute_kernel(X, self.parameters["x_train"])
        return self.svm.predict_proba(kernel_matrix)
