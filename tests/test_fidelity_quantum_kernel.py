import pytest
from pennylane import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding, AdamOptimizer
from qmlab.kernel import FidelityQuantumKernel
from qmlab.exceptions import InvalidEmbeddingError
from .qmlab_testcase import QMLabTest


class TestFidelityQuantumKernel(QMLabTest):

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
