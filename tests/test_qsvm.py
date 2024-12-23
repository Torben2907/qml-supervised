import pytest
import numpy as np
import pennylane as qml
import sklearn.datasets
import sklearn.model_selection
from .qmlab_testcase import QMLabTest
from qmlab.kernel import QSVC
from qmlab.kernel import FidelityQuantumKernel
from qmlab.exceptions import NotFittedError, QMLabError


class TestQSVC(QMLabTest):

    def setUp(self) -> None:
        super().setUp()
        self.qkernel = FidelityQuantumKernel(data_embedding="IQP", reps=1)
        self.X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        self.y = np.array([-1, -1, 1, 1])

    def test_no_kernel_provided(self) -> None:
        with pytest.raises(QMLabError):
            QSVC(quantum_kernel=None)  # type: ignore[arg-type]

    def test_no_valid_kernel_provided(self) -> None:
        with pytest.raises(QMLabError):
            QSVC(quantum_kernel="rbf")  # type: ignore[arg-type]

    def test_default_initialization(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel, random_state=self.random_state)

        assert qsvm.quantum_kernel == self.qkernel
        assert qsvm._random_state == self.random_state
        assert isinstance(qsvm._svm, sklearn.svm.SVC)

    def test_eval_before_fitted(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        with pytest.raises(NotFittedError):
            qsvm.predict(self.X)

    def test_qsvm_fit_calls_svm_fit(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        svm_fit = self.mocker.patch.object(qsvm._svm, "fit", autospec=True)

        qsvm.fit(self.X, self.y)

        svm_fit.assert_called_once_with(self.X, self.y, None)

    def test_qsvm_predict_calls_svm_predict(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        svm_predict = self.mocker.patch.object(qsvm._svm, "predict", autospec=True)
        qsvm.fit(self.X, self.y)

        qsvm.predict(self.X)

        svm_predict.assert_called_once_with(self.X)

    def test_qsvm_predict_proba_calls_svm_predict_proba(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        svm_predict_proba = self.mocker.patch.object(
            qsvm._svm, "predict_proba", autospec=True
        )
        qsvm.fit(self.X, self.y)

        qsvm.predict_proba(self.X)

        svm_predict_proba.assert_called_once_with(self.X)

    def test_qsvm_fit_calls_kernel_initialize(self) -> None:
        qsvm = QSVC(quantum_kernel=self.qkernel)
        qkernel_initialize = self.mocker.spy(self.qkernel, "initialize_params")

        qsvm.fit(self.X, self.y)

        qkernel_initialize.assert_called_once_with(self.X.shape[1], [-1, +1])

    def test_correctly_classify_dummy_data(self) -> None:
        qsvc = QSVC(quantum_kernel=self.qkernel)
        X = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        y = np.asarray([-1.0, 1.0])
        qsvc.fit(X, y)

        x_test = [
            np.asarray([[0.0, 0.0]]),
            np.asarray([[0.2, 0.2]]),
            np.asarray([[0.4, 0.4]]),
            np.asarray([[0.6, 0.6]]),
            np.asarray([[0.8, 0.8]]),
            np.asarray([[1.0, 1.0]]),
        ]
        y_test = [-1, -1, -1, 1, 1, 1]
        for i, xi in enumerate(x_test):
            assert qsvc.predict(xi) == y_test[i]
