import pytest
import numpy as np
from numpy.typing import NDArray
from pennylane import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding, AdamOptimizer
from qmlab.kernel import FidelityQuantumKernel
from qmlab.exceptions import InvalidEmbeddingError
from .qmlab_testcase import QMLabTest
from qmlab.exceptions import QMLabError


class TestFidelityQuantumKernel(QMLabTest):

    def setUp(self) -> None:
        super().setUp()
        self.X = np.asarray([[0, 0], [1, 1]])

    def test_wrong_embedding_as_str(self) -> None:
        with pytest.raises(InvalidEmbeddingError):
            FidelityQuantumKernel(data_embedding="rbf")

    def test_wrong_embedding_as_operation(self) -> None:
        with pytest.raises(InvalidEmbeddingError):
            FidelityQuantumKernel(
                data_embedding=AdamOptimizer
            )  # doesn't even make sense to give it an optimizer here but yeaaa. :D

    def test_embeddings_as_str_amplitude(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding="Amplitude")
        assert qkernel.data_embedding == AmplitudeEmbedding

    def test_embeddings_as_str_angle(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding="Angle")
        assert qkernel.data_embedding == AngleEmbedding

    def test_embeddings_as_str_iqp(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding="IQP")
        assert qkernel.data_embedding == IQPEmbedding

    def test_embedding_via_operation(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding=IQPEmbedding)
        assert qkernel.data_embedding == IQPEmbedding

    def test_evaluate_called_before_param_initialization(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding="IQP")
        with pytest.raises(QMLabError):
            qkernel.evaluate(self.X, self.X)

    def test_gram_matrix_is_psd(self) -> None:
        gram_matrix = self.compute_gram_matrix()
        assert np.all(np.linalg.eigvals(gram_matrix) > 0)

    def test_gram_matrix_has_ones_across_diagonal(self) -> None:
        gram_matrix = self.compute_gram_matrix()
        # rounding because of finite precision
        assert round(np.trace(gram_matrix)) == self.X.shape[1]

    def test_gram_matrix_is_symmetric(self) -> None:
        gram_matrix = self.compute_gram_matrix()
        np.testing.assert_allclose(
            actual=gram_matrix.T, desired=gram_matrix, rtol=1e-5, atol=1e-5
        )

    def test_fidelity_with_angle_embedding(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding="Angle")
        x1 = np.array([[0, np.pi / 2, np.pi]])
        x2 = np.array([[0, -np.pi, np.pi / 2]])
        qkernel.initialize_params(feature_dimension=x1.shape[1])
        # again we almost equal here because of the problem of finite precision
        np.testing.assert_array_almost_equal(qkernel.evaluate(x1, x2), np.array(1.0))

    def compute_gram_matrix(self) -> NDArray:
        """Helper method that takes the necessary steps
        to obtain the gram matrix for dummy data.

        Returns
        -------
        NDArray
            The gram matrix, Array of shape (m,m),
            where m is the number of examples of
            the dummy data.
        """
        qkernel = FidelityQuantumKernel(data_embedding="IQP")
        qkernel.initialize_params(feature_dimension=self.X.shape[1])
        gram_matrix = qkernel.evaluate(self.X, self.X)
        del qkernel
        return gram_matrix
