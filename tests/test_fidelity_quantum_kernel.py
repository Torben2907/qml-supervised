import pytest
import numpy as np
from pennylane import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding, AdamOptimizer
from qmlab.kernel import FidelityQuantumKernel
from qmlab.exceptions import InvalidEmbeddingError
from .qmlab_testcase import QMLabTest


class TestFidelityQuantumKernel(QMLabTest):

    def setUp(self):
        super().setUp()
        self.features = np.asarray([[0, 0], [1, 1]])

    def test_wrong_embedding_as_str(self):
        with pytest.raises(InvalidEmbeddingError):
            FidelityQuantumKernel(data_embedding="rbf")

    def test_wrong_embedding_as_operation(self):
        with pytest.raises(InvalidEmbeddingError):
            FidelityQuantumKernel(
                data_embedding=AdamOptimizer
            )  # doesn't even make sense to give it an optimizer here but yeaaa. :D

    def test_embeddings_as_str_amplitude(self):
        qkernel = FidelityQuantumKernel(data_embedding="Amplitude")
        assert qkernel.data_embedding == AmplitudeEmbedding

    def test_embeddings_as_str_angle(self):
        qkernel = FidelityQuantumKernel(data_embedding="Angle")
        assert qkernel.data_embedding == AngleEmbedding

    def test_embeddings_as_str_iqp(self):
        qkernel = FidelityQuantumKernel(data_embedding="IQP")
        assert qkernel.data_embedding == IQPEmbedding

    def test_embedding_via_operation(self):
        qkernel = FidelityQuantumKernel(data_embedding=IQPEmbedding)
        assert qkernel.data_embedding == IQPEmbedding

    # def test_gram_matrix_is_psd(self):
    #     qkernel = FidelityQuantumKernel(data_embedding="IQP")
    #     qkernel.initialize(feature_dimension=self.features.shape[1])
    #     gram_matrix = qkernel.evaluate(self.features, self.features)
    #     del qkernel
    #     assert np.all(np.linalg.eigvals(gram_matrix) > 0)

    # def test_gram_matrix_has_ones_across_diagonal(self):
    #     qkernel = FidelityQuantumKernel(data_embedding="IQP")
    #     qkernel.initialize(feature_dimension=self.features.shape[1])
    #     gram_matrix = qkernel.evaluate(self.features, self.features)
    #     del qkernel
    #     assert round(np.trace(gram_matrix)) == self.features.shape[1]

    # def test_gram_matrix_is_symmetric(self):
    #     gram_matrix = self.compute_gram_matrix()
    #     np.testing.assert_array_equal(gram_matrix.T, gram_matrix)
