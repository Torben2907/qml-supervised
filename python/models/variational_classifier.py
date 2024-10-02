import pennylane as qml
from jax import numpy as jnp
import jax
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class VariationalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        num_reps: int = 1,
        num_layers: int = 10,
        lr: float = 0.01,
        batch_size: int = 20,
        device: str = "default.qubit",
        interface: str = "jax",
        jit: bool = True,
        random_seed: int = 0,
    ) -> None:
        super().__init__()

        # HYPERPARAMETER
        self.num_reps: int = num_reps
        self.num_layers: int = num_layers
        self.lr: float = lr

        self.batch_size: int = batch_size
        self.device: str = device
        self.jit: bool = jit
        self.interface: str = interface
        self.range: int = np.random.default_rng(random_seed)

        # data-dependent attributes
        self.num_qubits_: int = None

    def generate_random_key(self):
        return jax.random.PRNGKey(self.range.integers(100000))

    def build_model(self):
        dev = qml.device(self.device, wires=self.num_qubits)

        @qml.qnode(dev, interface=self.interface)
        def circuit(params, x):
            qml.IQPEmbedding(x, wires=range(self.num_qubits), n_repeats=self.num_reps)
            qml.StronglyEntanglingLayers(
                params["weights"], wires=range(self.num_qubits), imprimitive=qml.CZ
            )
            return qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        self.circuit = circuit
        if self.jit:
            circuit = jax.jit(circuit)
        self.forward = jax.vmap(circuit, in_axes=(None, 0))

        return self.forward

    def initialize(self, num_features: int, classes: list = [-1, +1]):
        self.classes_ = classes
        self.num_qubits_ = num_features
        self.num_classes = len(self.classes_)
        assert (
            self.num_classes == 2
        ), "binary classification requires exactly two classes"
        assert 1 in self.classes_
        assert -1 in self.classes_

        self._initialize_params()
        self.build_model()

    def _initialize_params(self):
        self.weights = (
            2
            * jnp.pi
            * jax.random.uniform(
                shape=(self.num_layers, self.num_qubits_, 3),
                key=self.generate_random_key(),
            )
        )
        self.params_ = {"weights", self.weights}

    def fit(self, X, y):
        num_features = X.shape[1]
        num_classes = np.unique(y)
        self.initialize(num_features, num_classes)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def transform(self, X, preprocess=True):
        pass
