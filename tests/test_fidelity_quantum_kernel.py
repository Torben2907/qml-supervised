import pytest
from pennylane import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding, AdamOptimizer
from qmlab.kernel import FidelityQuantumKernel
from qmlab.exceptions import InvalidEmbeddingError
from .qmlab_testcase import QMLabTest


class TestFidelityQuantumKernel(QMLabTest):

    def test_wrong_embedding_as_str(self):
        with pytest.raises(InvalidEmbeddingError):
            FidelityQuantumKernel(embedding="rbf")

    def test_wrong_embedding_as_operation(self):
        with pytest.raises(InvalidEmbeddingError):
            FidelityQuantumKernel(
                embedding=AdamOptimizer
            )  # doesn't even make sense to give it an optimizer here but yeaaa. :D

    def test_embeddings_as_str_amplitude(self):
        qkernel = FidelityQuantumKernel(embedding="Amplitude")
        assert qkernel.embedding == AmplitudeEmbedding

    def test_embeddings_as_str_angle(self):
        qkernel = FidelityQuantumKernel(embedding="Angle")
        assert qkernel.embedding == AngleEmbedding

    def test_embeddings_as_str_iqp(self):
        qkernel = FidelityQuantumKernel(embedding="IQP")
        assert qkernel.embedding == IQPEmbedding

    def test_embedding_via_operation(self):
        qkernel = FidelityQuantumKernel(embedding=IQPEmbedding)
        assert qkernel.embedding == IQPEmbedding
