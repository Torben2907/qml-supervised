import pytest
import numpy as np
from numpy.typing import NDArray
from pennylane import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding, AdamOptimizer
from qmlab.kernel import FidelityQuantumKernel
from qmlab.exceptions import InvalidEmbeddingError
from .qmlab_testcase import QMLabTest


class TestFidelityQuantumKernel(QMLabTest):

    def setUp(self) -> None:
        super().setUp()
        self.features = np.asarray([[0, 0], [1, 1]])

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

    def test_gram_matrix_is_psd(self) -> None:
        gram_matrix = self.compute_gram_matrix()
        assert np.all(np.linalg.eigvals(gram_matrix) > 0)

    def test_gram_matrix_has_ones_across_diagonal(self) -> None:
        gram_matrix = self.compute_gram_matrix()
        # rounding because of finite precision
        assert round(np.trace(gram_matrix)) == self.features.shape[1]

    def test_gram_matrix_is_symmetric(self) -> None:
        gram_matrix = self.compute_gram_matrix()
        np.testing.assert_array_equal(gram_matrix.T, gram_matrix)

    def compute_gram_matrix(self) -> NDArray:
        qkernel = FidelityQuantumKernel(data_embedding="IQP")
        qkernel.initialize(feature_dimension=self.features.shape[1])
        gram_matrix = qkernel.evaluate(self.features, self.features)
        del qkernel
        return gram_matrix
