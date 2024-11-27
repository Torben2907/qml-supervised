from typing import Any, Dict
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from ..exceptions import NotFittedError, QMLabError
import pennylane as qml
from qmlab.preprocessing import pad_and_normalize_data


class BaseQSVM(BaseEstimator, ClassifierMixin):
    """An abstract class for a quantum support vector machine. Written with the intention
    of making it extendable for regression problems in the future.

    Parameters
    ----------
    BaseEstimator : sklearn.base.BaseEstimator
        Base class for all estimators in scikit-learn.
    ClassifierMixin : sklearn.base.ClassifierMixin
        Mixin class for all classifiers in scikit-learn.
    """

    def __init__(
        self,
        svm: Any,
        quantum_kernel: QuantumKernel,
        random_state: int = 42,
        **svm_kwargs,
    ) -> None:
        """Constructor of the quantum support vector machine (QSVM).

        Parameters
        ----------
        svm : sklearn.svm.SVC for this study.
            classical support vector machine.
        quantum_kernel : QuantumKernel
            Quantum Kernel implementation, has to inherit from abstract class QuantumKernel.
        random_state : int, optional
            Fixing seed for reproducable results, by default 42

        Raises
        ------
        QMLabError
            Error when no quantum kernel is provided.
        QMLabError
            Error when no valid quantum kernel is provided.
        """
        if quantum_kernel is None:
            raise QMLabError("Parameter `quantum_kernel` must be provided.")
        elif not isinstance(quantum_kernel, QuantumKernel):
            raise QMLabError(
                "Parameter `quantum_kernel` must be of type QuantumKernel."
            )
        self._quantum_kernel = quantum_kernel
        self._random_state = random_state
        self._svm = svm(
            kernel=self._quantum_kernel.evaluate, probability=True, **svm_kwargs
        )
        """This dict will save the feature matrix used for training of the quantum 
        kernel. This way we can check whether the QSVM is fitted or not."""
        self.params_: Dict[str, NDArray] = {}

    @property
    def quantum_kernel(self) -> QuantumKernel:
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        self._quantum_kernel = quantum_kernel
        self._svm.kernel = self._quantum_kernel.evaluate

    @property
    def random_state(self) -> int:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: int) -> None:
        self._random_state = random_state

    def check_if_fitted(self) -> None:
        """If "X_train" is not in the parameter-dictionary we haven't called fit yet,
        i.e. the model is untrained, which is not desired for obtaining optimal results.
        Therefore we throw an error here.

        Raises
        ------
        NotFittedError
            Gets thrown when the predict or score method gets called before the model is fitted.
        """
        if "X_train" not in self.params_:
            raise NotFittedError(
                "Model needs to be fitted on the data before evaluating it."
            )

    def fit(
        self, X: NDArray, y: NDArray, sample_weight: NDArray | None = None
    ) -> "BaseQSVM":
        """Fit the model to the training data. Ideally this will do 3 things:
        1. save the feature matrix used for training so we can check if the model
        is fitted later on,
        2. Initialize a valid quantum kernel
        3. Evaluate the quantum kernel, i.e. obtain a real-valued kernel matrix, that can
        be used by the scikit-learn implementation of the traditional support vector machine.

        Parameters
        ----------
        X : NDArray
            Feature matrix / 2D Array of shape (m, d) used for training/fitting of the QSVM.
        y : NDArray
            Label vector / 1D Array of shape (m,) used for training/fitting of the QSVM.
        sample_weight : NDArray | None, optional
            _description_, by default None

        Returns
        -------
        BaseQSVM
            The model itself fitted on the data now. Usually this output can be safely ignored by the user.
        """
        self._svm.random_state = self._random_state
        self.params_ = {"X_train": X}
        self._quantum_kernel.initialize_params(
            feature_dimension=X.shape[1], class_labels=np.unique(y).tolist()
        )
        # if self._quantum_kernel.data_embedding == qml.AmplitudeEmbedding:
        #     X = pad_and_normalize_data(X)
        self._svm.fit(X, y, sample_weight)
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Makes a prediction with the quantum support vector machine.

        Parameters
        ----------
        X : NDArray
            Feature matrix / 2D Array of shape (m, d) where m is the number of samples used
            for testing and d is the number of features.

        Returns
        -------
        NDArray
            A prediction vector / 1D Array of shape (m,) where m is the number of samples
            used for testing.
        """
        self.check_if_fitted()
        return self._svm.predict(X)

    def predict_proba(self, X: NDArray) -> NDArray:
        self.check_if_fitted()
        return self._svm.predict_proba(X)

    def score(
        self, X: NDArray, y: NDArray, sample_weight: NDArray | None = None
    ) -> float:
        self.check_if_fitted()
        return self._svm.score(X, y, sample_weight)


class QSVC(BaseQSVM):
    """The classification implementation of the quantum support vector machine,
    making it a quantum support vector classifier. We pass in the classical
    support vector classifier from scikit-learn, i.e. sklearn.svm.SVC.

    Parameters
    ----------
    BaseQSVM : qsvm.BaseQSVM
        The abstract class containing the fit, predict, and score methods of a
        quantum based classifier.
    """

    def __init__(
        self, quantum_kernel: QuantumKernel, random_state: int = 42, **svm_kwargs
    ):
        super().__init__(SVC, quantum_kernel, random_state, **svm_kwargs)
