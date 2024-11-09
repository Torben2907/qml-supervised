import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qmlab.kernel.qfm import QuantumFeatureMap
from qmlab.kernel.qsvc import QSVC
from qiskit_algorithms.utils import algorithm_globals
from qmlab_testcase import QMLabTestCase

seed = 12345


class TestQSVC(QMLabTestCase):
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = seed

        X = np.asarray(
            [
                [0.63091571, 0.39476126],
                [0.31647494, 1.01113425],
                [2.21722301, 1.0098378],
                [-0.36094076, 0.64501089],
            ]
        )
        y = np.array([-1, +1, +1, +1])

        self.num_features = X.shape[1]
        self.num_qubits = self.num_features
        self.num_layers = 2
        self.alpha = 2.0
        self.C = 1.0

    def test_initialize_explicit_feature_map(self):
        qfm = QuantumFeatureMap(
            num_features=self.num_features,
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            gates=["H", "RZ", "CZ"],
            entanglement="linear",
            alpha=self.alpha,
        )

        qsvc = QSVC(feature_map=qfm, C=self.C, random_state=seed)

        assert isinstance(qfm, QuantumCircuit)
        assert qsvc.feature_map.num_features == self.num_features
        assert qsvc.feature_map.num_qubits == self.num_qubits
        assert qsvc.C == self.C
        assert qsvc.random_state == seed

    def test_initialize_implicit_feature_map(self):
        qsvc = QSVC(
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            feature_map=["H", "RZ", "CZ"],
            C=self.C,
            random_state=seed,
        )

        assert qsvc.num_qubits == self.num_qubits
        assert qsvc.num_layers == self.num_layers
        assert qsvc.C == self.C
        assert qsvc.random_state == seed
