import pennylane as qml
from jax import numpy as jnp
import jax
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
import optax
import torch
from models.model_utils import chunk_vmapped_fn, train_with_jax, quantum_model_train
from typing import Optional


class VariationalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        num_reps: int = 1,
        num_layers: int = 10,
        lr: float = 0.01,
        batch_size: int = 20,
        quantum_device: str = "default.qubit",
        interface: str = "torch",
        gpu_device: str = "mps",
        diff_method: str = "parameter-shift",
        jit: bool = True,
        random_seed: int = 0,
        max_vmap: Optional[int] = None,
        convergence_threshold: int = 10,
        scaling: float = 1.0,
        num_steps: int = 2000,
    ) -> None:
        super().__init__()

        # HYPERPARAMETER
        self.num_reps: int = num_reps
        self.num_layers: int = num_layers
        self.lr: float = lr
        self.scaling: float = scaling
        self.random_seed: int = random_seed

        # for pennylane
        self.quantum_device: str = quantum_device
        self.diff_method: str = diff_method

        # choose your player: jax or torch
        self.interface: str = interface

        # I use torch and Mac M3
        self.gpu_device: str = gpu_device

        # parameters that will be passed to the optimization
        # framework
        self.batch_size: int = batch_size
        self.convergence_threshold = convergence_threshold
        self.num_steps: int = num_steps

        # JAX-specific type shit
        if interface == "jax":
            self.range: int = np.random.default_rng(self.random_seed)
            self.jit: bool = jit
            if max_vmap:
                self.max_vmap = self.batch_size
            else:
                self.max_vmap = max_vmap
        else:
            self.jit: bool = None
            self.range: int = None
            np.random.seed(self.random_seed)
            if max_vmap:
                self.max_vmap = None
                raise ValueError("don't provide max_vmap when interface isn't jax.")

        # data-dependent attributes
        self.params_: dict = None
        self.num_qubits_: int = None
        self.scaler = None
        self.circuit = None
        self.training_time_ = None
        self.loss_record = None

    def build_model(self):
        dev = qml.device(self.quantum_device, wires=self.num_qubits_)

        @qml.qnode(dev, interface=self.interface, diff_method=self.diff_method)
        def circuit(params, x):
            qml.IQPEmbedding(x, wires=range(self.num_qubits_), n_repeats=self.num_reps)
            qml.StronglyEntanglingLayers(
                params["weights"], wires=range(self.num_qubits_), imprimitive=qml.CZ
            )
            return qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        self.circuit = circuit

        if self.interface == "jax":
            if self.jit:
                circuit = jax.jit(circuit)
            self.forward = jax.vmap(circuit, in_axes=(None, 0))
            self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)
            return self.forward
        else:
            return self.circuit

    def initialize(self, num_features: int, classes=None):
        if classes is None:
            classes = [-1, 1]
        self.classes_ = classes
        self.num_qubits_ = num_features
        self.num_classes = len(self.classes_)
        assert (
            self.num_classes == 2
        ), "binary classification requires exactly two classes"
        assert 1 in self.classes_
        assert -1 in self.classes_

        self.initialize_params()
        self.build_model()

    def initialize_params(self):
        if self.interface == "jax":
            init_weights = (
                2
                * jnp.pi
                * jax.random.uniform(
                    shape=(self.num_layers, self.num_qubits_, 3),
                    key=self._generate_random_key(),
                )
            )
        elif self.interface == "torch":
            init_weights = (
                2 * np.pi * np.random.random((self.num_layers, self.num_qubits_, 3))
            )
            init_weights = torch.tensor(
                init_weights,
                requires_grad=True,
                device=self.gpu_device,
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError("use `jax` or `torch` for the interface.")
        self.params_ = {"weights": init_weights}

    def fit(self, X, y):
        num_features = X.shape[1]
        num_classes = np.unique(y)
        self.initialize(num_features, num_classes)

        self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.scaler.fit(X)
        X = self.transform(X)

        if self.interface == "jax":
            optimizer = optax.adam
        else:
            optimizer = torch.optim.Adam

        if self.interface == "jax":

            def loss_fn(params, X, y):
                expvals = self.forward(params, X)
                probs = (1 - expvals * y) / 2
                return jnp.mean(probs)

            if self.jit:
                loss_fn = jax.jit(loss_fn)

            self.params_ = train_with_jax(
                self,
                loss_fn,
                optimizer,
                X,
                y,
                self._generate_random_key,
                self.convergence_threshold,
            )
        elif self.interface == "torch":

            def loss_fn(params, X, y):
                expvals = self.circuit(params, X)
                probs = (1 - expvals * y) / 2
                return torch.mean(probs)

            self.params_ = quantum_model_train(
                self,
                loss_fn,
                optimizer,
                X,
                y,
                self.random_seed,
                self.convergence_threshold,
                self.gpu_device,
            )
        else:
            raise NotImplementedError(
                "no other interfaces than `jax` or `torch` are supported currently"
            )

        return self

    def predict(self, X):
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        # mapped predictions as indices to take from self.classes_
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        X = self.transform(X)
        if self.interface == "jax":
            predictions = self.chunked_forward(self.params_, X)
        if self.interface == "torch":
            predictions = self.circuit(self.params_, X)
            predictions = predictions.detach().numpy()
        predictions_2d = np.c_[(1 - predictions) / 2, (1 + predictions) / 2]
        return predictions_2d

    def transform(self, X, preprocess=True):
        if preprocess:
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
                self.scaler.fit(X)
            X = self.scaler.transform(X)

        return X * self.scaling

    def _generate_random_key(self):
        return jax.random.PRNGKey(self.range.integers(100000))
