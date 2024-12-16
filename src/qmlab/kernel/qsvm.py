import time
import numpy as np
from typing import Any, Dict
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from ..exceptions import NotFittedError, QMLabError


class BaseQSVM(BaseEstimator, ClassifierMixin):
    """An abstract class for a quantum support vector machine (QSVM).
    Written with the intention of making it
    extendable for regression or multiclass problems in the future.

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
        """Constructor of the QSVM.

        Parameters
        ----------
        svm : classical support vector machine (SVM).
        quantum_kernel : QuantumKernel
            A valid implementation of a quantum kernel, i.e.
            has to inherit from abstract class QuantumKernel.
        random_state : int, optional
            Fixing seed for reproducable results, by default 42.

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
                "Parameter `quantum_kernel` must be of type `QuantumKernel`."
            )
        self._quantum_kernel = quantum_kernel
        self._random_state = random_state
        self._svm = svm(
            kernel=self._quantum_kernel.evaluate, probability=True, **svm_kwargs
        )
        """This dict will save the feature matrix used for training of the quantum 
        kernel. This way we can check whether the QSVM is fitted or not."""
        self.classes_: None | NDArray = None
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
        """Fit the model to the training data. Ideally this will do three things:
        1. Save the feature matrix used for training so we can check if the model
        is fitted later on.
        2. Initialize a valid quantum kernel.
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
            The model itself fitted on the data.
            This output can be safely ignored by the user.
        """
        self.classes_ = np.unique(y)
        if self.classes_ is None:
            raise QMLabError("Did you provide a correct label vector?")
        self._svm.random_state = self._random_state
        self.params_ = {"X_train": X}
        self._quantum_kernel.initialize_params(
            feature_dimension=X.shape[1], class_labels=self.classes_.tolist()
        )
        start = time.time()
        self._svm.fit(X, y, sample_weight)
        self.training_time_ = time.time() - start
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Makes a prediction with the QSVM.

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
        """Compute probabilities of possible outcomes for all samples
        in X.

        Parameters
        ----------
        X : NDArray
            Feature matrix of shape (m, d), where m is the number of samples
            used for testing and d is the number of features.

        Returns
        -------
        NDArray
            of shape (m, c), where m is the number of samples used for testing
            and c is the number of class labels (here always c = 2).
            Contains the probabilites for each sample to lay in one of the
            classes. Columns respond to the classes in sorted order, i.e.
            here [-1, +1].
        """
        self.check_if_fitted()
        return self._svm.predict_proba(X)

    def score(
        self, X: NDArray, y: NDArray, sample_weight: NDArray | None = None
    ) -> float:
        r"""Uses the metric that can be specified when running a cross-validation
        to estimate the performance of the model on the dataset.

        Parameters
        ----------
        X : NDArray
            Feature matrix of shape (m, d), m are the number of samples and d the
            number of features.
        y : NDArray
            Row vector of shape (1, m) containing the corresponding labels.

        Returns
        -------
        float
            The result of the metric indicating how well the model performs
            on the task of regression or classification.
            By default the accuracy (ACC) will be used:

            \text{ACC}(h_{\bm{\theta}}(\boldsymbol{x}), \boldsymbol{y}) = \frac{1}{m} \sum_{j=1}^m
            I[h(x_j) = y_j], \qquad \text{where} \qquad
            h(\boldsymbol{x}), \boldsymbol{y} \in \mathbb{R}^{1 \times m}.

        """
        self.check_if_fitted()
        return self._svm.score(X, y, sample_weight)

    def decision_function(self, X: NDArray) -> NDArray:
        r"""Signed distance to the separating hyperplane.

        :math:
            h_{\bm{\theta}}(\bm{x})
            = \text{sign} \left( \left\langle \bm{w}, \bm{x} \right\rangle + b \right).

        Parameters
        ----------
        X : NDArray
            Feature matrix of shape (m, d), where m is the number of
            samples and d the number of features.

        Returns
        -------
        NDArray: of shape (m,) (the decision function for the complete dataset).

        """
        return self._svm.decision_function(X)


class QSVC(BaseQSVM):
    """The classification implementation of the quantum support vector machine,
    making it a quantum support vector classifier. We pass in the classical
    support vector classifier from scikit-learn, i.e. `sklearn.svm.SVC`.

    This is based on Algorithm 2 of the thesis.

    Parameters
    ----------
    BaseQSVM : qsvm.BaseQSVM
        The abstract class containing the `fit`-, `predict`-, `predict_proba`- and
        `score`-methods of a QSVM.
    """

    def __init__(
        self, quantum_kernel: QuantumKernel, random_state: int = 42, **svm_kwargs
    ):
        super().__init__(SVC, quantum_kernel, random_state, **svm_kwargs)
