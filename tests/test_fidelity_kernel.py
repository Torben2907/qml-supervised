import numpy as np
from qmlab_testcase import QMLabTest
from qiskit_algorithms.utils import algorithm_globals
from qmlab.kernel.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit.circuit.library import PauliFeatureMap
from sklearn.svm import SVC


class TestFidelityQuantumKernel(QMLabTest):
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345

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

    def test_defaults(self):
        features = np.random.randn(10, 2)
        labels = np.sign(features[:, 0])

        quantum_kernel = FidelityQuantumKernel()
        svc = SVC(kernel=quantum_kernel.evaluate_kernel)
        svc.fit(features, labels)
        score = svc.score(features, labels)
        assert score >= 0.5
