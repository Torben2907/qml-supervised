from typing import Sequence
import numpy as np
from qmlab_testcase import QMLabTest
from qmlab.kernel import FidelityQuantumKernel
from qiskit.circuit.library import PauliFeatureMap
from qiskit import QuantumCircuit
from sklearn.svm import SVC
from qiskit_algorithms.state_fidelities import BaseStateFidelity, StateFidelityResult
from qiskit_algorithms.algorithm_job import AlgorithmJob


class TestFidelityQuantumKernel(QMLabTest):
    def setUp(self):
        super().setUp()
        # XOR-Data
        self.X_train = np.asarray(
            [
                [1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
            ]
        )
        self.y_train = np.asarray([+1, +1, -1, -1])
        self.X_test = np.asarray([[0.9, 2.1], [0.8, -1.6]])
        self.y_test = np.asarray([-1, +1])
        self.qfm = PauliFeatureMap(2)

    def test_kernel_callable_from_svc(self):
        quantum_kernel = FidelityQuantumKernel(feature_map=self.qfm)
        svc = SVC(kernel=quantum_kernel.evaluate_kernel)
        svc.fit(self.X_train, self.y_train)
        score = svc.score(self.X_test, self.y_test)
        assert score == 1.0

    def test_kernel_precomputed_for_svc(self):
        kernel = FidelityQuantumKernel(feature_map=self.qfm)
        kernel_train = kernel.evaluate_kernel(self.X_train)
        kernel_test = kernel.evaluate_kernel(self.X_test, self.X_train)

        svc = SVC(kernel="precomputed")
        svc.fit(kernel_train, self.y_train)
        score = svc.score(kernel_test, self.y_test)

        self.assertEqual(score, 1.0)

    def test_default_values(self):
        features = np.random.randn(10, 2)
        labels = np.sign(features[:, 0])

        quantum_kernel = FidelityQuantumKernel()
        svc = SVC(kernel=quantum_kernel.evaluate_kernel)
        svc.fit(features, labels)
        score = svc.score(features, labels)
        assert score > 0.5

    def test_enforce_psd(self):
        kernel = FidelityQuantumKernel(fidelity=MockFidelity(), enforce_psd=False)
        gram_matrix = kernel.evaluate_kernel(self.X_train)
        eigen_values = np.linalg.eigvals(gram_matrix)
        self.assertFalse(np.all(np.greater_equal(eigen_values, -1e-10)))


class MockFidelity(BaseStateFidelity):
    """Custom fidelity that returns -0.5 for any input."""

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        raise NotImplementedError()

    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ) -> StateFidelityResult | AlgorithmJob:
        values = np.asarray(values_1)
        fidelities = np.full(values.shape[0], -0.5)
        return AlgorithmJob(MockFidelity._call, fidelities, options)

    @staticmethod
    def _call(fidelities, options) -> StateFidelityResult:
        return StateFidelityResult(fidelities, [], {}, options)
