import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qmlab.qfm import QuantumFeatureMap
from qmlab.qsvm import QSVC
from qiskit_algorithms.utils import algorithm_globals
from qmlab_testcase import QMLabTest
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliFeatureMap
from qmlab.kernel.fidelity_statevector_kernel import FidelityStatevectorKernel

# fix seed so tests are deterministic
random_state = 12345


class TestQSVC(QMLabTest):
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = random_state

        self.X = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.y = np.array([-1, +1])

        self.num_features = self.X.shape[1]
        self.num_qubits = self.num_features
        self.reps = 1
        self.alpha = 1.0
        self.C = 1.0

    def test_init_directly_passing_feature_map(self):
        qfm = QuantumFeatureMap(
            feature_dimension=self.num_features,
            num_qubits=self.num_qubits,
            reps=self.reps,
            gates=["H", "RZ", "CZ"],
            entanglement="linear",
            alpha=self.alpha,
        )

        qsvc = QSVC(feature_map=qfm, C=self.C, random_state=random_state)

        assert isinstance(qfm, QuantumCircuit)
        assert qsvc.feature_map.feature_dimension == self.num_features
        assert qsvc.feature_map.num_qubits == self.num_qubits
        assert qsvc.C == self.C
        assert qsvc.random_state == random_state

    def test_init_creating_feature_map_internally(self):
        qsvc = QSVC(
            num_qubits=self.num_qubits,
            reps=self.reps,
            feature_map=["H", "RZ", "CZ"],
            C=self.C,
            random_state=random_state,
        )

        assert qsvc.num_qubits == self.num_qubits
        assert qsvc.reps == self.reps
        assert qsvc.C == self.C
        assert qsvc.random_state == random_state

    def test_kernel_values(self):
        quantum_kernel = FidelityStatevectorKernel()
        qsvc = QSVC(
            num_qubits=2,
            reps=1,
            feature_map=["H", "RZ"],
            C=self.C,
            alpha=2.0,
            entanglement="linear",
            random_state=random_state,
            quantum_kernel=quantum_kernel,
        )
        qsvc.fit(self.X, self.y)
        print(qsvc._qfm.draw())
        x1 = np.array([0.0, 0.0])
        x2 = np.array([[0.0, 0.0], [np.pi, 0.0], [2 * np.pi, 0.0]])
        desired_kernel_values = np.array([1.0, 0.0, 1.0])
        for i, xi in enumerate(x2):
            print(i, xi)
            assert qsvc.kernel(x1, xi) == desired_kernel_values[i]

    def test_predictions(self):
        qsvc = QSVC(
            num_qubits=self.num_qubits,
            reps=self.reps,
            feature_map=["H", "RZ", "CZ"],
            alpha=self.alpha,
            C=self.C,
            random_state=random_state,
        )
        qsvc.fit(self.X, self.y)
        X_test = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.2],
                [0.4, 0.4],
                [0.6, 0.6],
                [0.8, 0.8],
                [1.0, 1.0],
            ]
        )
        y_test = np.array([-1, -1, -1, +1, +1, +1])
        for i, xi in enumerate(X_test):
            print(i, xi)
            assert qsvc.predict(xi) == y_test[i]
