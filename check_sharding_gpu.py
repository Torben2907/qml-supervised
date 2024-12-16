"""This script tests the sharding 
    (dividing the computations onto multiple GPUs)
    of JAX for PennyLane QNodes.  

    Returns
    -------
    Any
        Grid displaying the sharded devices,
        Correctly computed gram matrix by multiple devices at 
        the same time.
"""

from typing import Callable
import jax
import matplotlib as mpl
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from qmlab.kernel.kernel_utils import vmap_batch
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh, PositionalSharding
from pennylane.measurements import ProbabilityMP

num_qubits = 2
max_batch_size = 100

device = qml.device("default.qubit", wires=num_qubits)

X = np.array([[0, 0], [1, 1], [2, 2]])


@qml.qnode(device, interface="jax")
def circuit(combined_input: jax.Array) -> ProbabilityMP:
    qml.AmplitudeEmbedding(
        features=combined_input[:num_qubits],
        wires=range(num_qubits),
        pad_with=0.5,
    )
    qml.adjoint(
        qml.AmplitudeEmbedding(
            features=combined_input[num_qubits:],
            wires=range(num_qubits),
            pad_with=0.5,
        )
    )
    return qml.probs()


combined_input = jnp.array(
    [np.concatenate((X[i], X[j])) for i in range(len(X)) for j in range(len(X))]
)
print(combined_input)

sharding = PositionalSharding(jax.devices())
num_devices = jax.local_device_count()
sharded_input = jax.device_put(combined_input, sharding.reshape(1, num_devices))

circuit_jitted = jax.jit(circuit)
batched_circuit = vmap_batch(
    jax.vmap(circuit_jitted, 0), start=0, max_batch_size=max_batch_size
)

print(combined_input.shape)


def visualize(batched_circuit: Callable, color_map: str = "Set3") -> None:
    jax.debug.visualize_array_sharding(
        batched_circuit, color_map=mpl.colormaps[color_map]
    )


visualize(sharded_input)
bc = batched_circuit(sharded_input)[:, 0]
gram_matrix = np.reshape(bc, (len(X), len(X)))
print(gram_matrix)
