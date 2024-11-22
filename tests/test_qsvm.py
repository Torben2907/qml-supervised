import sklearn
import pytest
import numpy as np
import pennylane as qml
from qmlab_testcase import QMLabTest
from qmlab.kernel import QSVC
from qmlab.kernel import FidelityQuantumKernel

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([-1, -1, 1, 1])


class TestQSVC(QMLabTest):

    def setUp(self) -> None:
        super().setUp()
        self.qkernel = FidelityQuantumKernel(embedding=qml.IQPEmbedding)
        self.X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        self.y = np.array([-1, -1, 1, 1])

    def test_no_kernel_provided(self) -> None:
        with pytest.raises(ValueError):
            QSVC(quantum_kernel=None)  # type: ignore[arg-type]

    def test_default_initialization(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel, random_state=self.random_state)
        assert qsvm.quantum_kernel == self.qkernel
        assert qsvm._random_state == self.random_state
        assert isinstance(qsvm._svm, sklearn.svm.SVC)

    def test_eval_before_fitted(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        with pytest.raises(ValueError):
            qsvm.predict(self.X)

    def test_fit(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        svm_fit = self.mocker.patch.object(qsvm._svm, "fit", autospec=True)
        qsvm.fit(self.X, self.y)
        svm_fit.assert_called_once_with(self.X, self.y, None)
