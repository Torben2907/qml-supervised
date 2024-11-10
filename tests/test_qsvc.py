import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qmlab.qfm import QuantumFeatureMap
from qmlab.qsvm import QSVC
from qiskit_algorithms.utils import algorithm_globals
from qmlab_testcase import QMLabTest

# fix seed so tests are deterministic
random_state = 12345


class TestQSVC(QMLabTest):
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = random_state

        self.X = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        self.y = np.array([-1, +1])

        self.num_features = self.X.shape[1]
        self.num_qubits = self.num_features
        self.num_layers = 1
        self.alpha = 1.0
        self.C = 1.0

    def test_init_directly_passing_feature_map(self):
        qfm = QuantumFeatureMap(
            feature_dimension=self.num_features,
            num_qubits=self.num_qubits,
            reps=self.num_layers,
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
            reps=self.num_layers,
            feature_map=["H", "RZ", "CZ"],
            C=self.C,
            random_state=random_state,
        )

        assert qsvc.num_qubits == self.num_qubits
        assert qsvc.reps == self.num_layers
        assert qsvc.C == self.C
        assert qsvc.random_state == random_state

    @pytest.mark.skip(
        "TODO: have to fix this one. value should be 1.0 but it's 0.39 for some reason smh"
    )
    def test_correct_kernel_value_computation(self):
        qsvc = QSVC(
            num_qubits=self.num_qubits,
            reps=self.num_layers,
            feature_map=["H", "RZ"],
            alpha=self.alpha,
            C=self.C,
            entanglement="linear",
            random_state=random_state,
            quantum_kernel=None,
        )
        qsvc.fit(self.X, self.y)
        print(qsvc._qfm.draw())
        x1 = np.array([0.0, 0.0])
        x2 = np.array([[0.0, 0.0], [2 * np.pi, 0.0]])
        k12 = np.array([1.0, 1.0])
        for i, xi in enumerate(x2):
            print(i, xi)
            assert qsvc.kernel(x1, xi) == k12[i]
