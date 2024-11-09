from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qmlab.kernel.qfm import QuantumFeatureMap
from qmlab.kernel.qsvc import QSVC
from qiskit_algorithms.utils import algorithm_globals
from .qmlab_testcase import QMLabTestCase

seed = 12345


class TestQSVC(QMLabTestCase):
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = seed

        X = [[0.0, 0.0], [1.0, 1.0]]
        y = [-1, +1]
        self.num_features = X.shape[1]
        self.num_qubits = self.num_features
        self.num_layers = 2
        self.alpha = 2.0
        self.C = 1.0

    def test_initialize_no_params(self):
        qfm = QuantumFeatureMap(
            num_features=self.num_features,
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            gates=["H", "RZ", "CZ"],
            entanglement="linear",
        )

        qsvc = QSVC(feature_map=qfm, alpha=self.alpha, C=self.C, random_state=seed)

        assert isinstance(qfm, QuantumCircuit)
        assert qsvc.feature_map.num_features == self.num_features
        assert qsvc.feature_map.num_qubits == self.num_qubits
        assert qsvc.alpha == self.alpha
        assert qsvc.C == self.C
        assert qsvc.random_state == seed
        assert isinstance(qsvc.backend, Sampler)
