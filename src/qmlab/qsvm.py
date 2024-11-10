import numpy as np
from sklearn.svm import SVC
from .kernel.fidelity_quantum_kernel import FidelityQuantumKernel
from .kernel.quantum_kernel import QuantumKernel
from .qfm import QuantumFeatureMap
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService


class QSVC(SVC):
    def __init__(
        self,
        *,
        quantum_kernel: QuantumKernel | None = None,
        num_qubits: int = 2,
        reps: int = 2,
        feature_map: list[str] = ["H", "RZ", "CZ"],
        entanglement: str = "linear",
        alpha: float | None = 2.0,
        quantum_backend: Sampler | QiskitRuntimeService = None,
        random_state: int | None = None,
        **kwargs,
    ):
        if "kernel" in kwargs:
            del kwargs["kernel"]
            raise ValueError(
                "do not use the `kernel` argument - use `quantum_kernel` instead!"
            )

        SVC.__init__(self, **kwargs)

        self.num_qubits = num_qubits
        self.reps = reps
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.alpha = alpha
        self.random_state = random_state
        self.quantum_kernel = (
            quantum_kernel if quantum_kernel else FidelityQuantumKernel()
        )

        # if no quantum device is setup,
        # simulate quantum backend classically:
        if not quantum_backend:
            np.random.seed(self.random_state)
            algorithm_globals._random_seed = self.random_state
            self.quantum_backend = Sampler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 and y.ndim == 1
        assert np.array_equal(np.unique(y), [-1, +1])

        num_features = X.shape[1]
        if isinstance(self.feature_map, list):
            self._qfm = QuantumFeatureMap(
                feature_dimension=num_features,
                num_qubits=self.num_qubits,
                reps=self.reps,
                gates=[g.upper() for g in self.feature_map],
                entanglement=self.entanglement,
                alpha=self.alpha,
                repeat=True,
                data_scaling=False,
            )
        elif isinstance(self.feature_map, QuantumCircuit):
            self._qfm = self.feature_map
        else:
            raise ValueError(
                "Feature Map must be either given as a list containing"
                "names of the gates as strings or directly via a `QuantumCircuit`-object!"
            )

        self.kernel = self.quantum_kernel.evaluate_kernel

        SVC.fit(self, X, y)

        return self

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
