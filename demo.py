import jax.numpy as jnp
import numpy as np
import pennylane as qml
import jax
from qmlab.kernel.kernel_utils import vmap_batch

dev = qml.device("default.qubit", wires=range(2))


@qml.qnode(dev)
def circuit(concat_vec=None):
    qml.AmplitudeEmbedding(
        features=concat_vec[:2], wires=range(2), pad_with=0.0, normalize=True
    )
    qml.adjoint(
        qml.AmplitudeEmbedding(
            features=concat_vec[2:], wires=range(2), pad_with=0.0, normalize=True
        )
    )
    return qml.state()


f1 = np.array([[0, np.pi, np.pi / 2]])
f2 = np.array([[0, np.pi / 2, np.pi]])

Z = np.array(
    [np.concatenate((f1[i], f2[j])) for i in range(len(f1)) for j in range(len(f2))]
)

print(Z)


batched_circuit = vmap_batch(jax.vmap(circuit), start=0, max_batch_size=256)

state = batched_circuit(Z)
print(state)
