import jax
import jax.numpy as jnp
import pennylane as qml
import matplotlib.pyplot as plt

num_qubits = 5

dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev, interface="jax", diff_method=None)
def qc(x: jax.Array):
    qml.IQPEmbedding(x, wires=range(num_qubits))
    return qml.state()


# print(qml.draw(qc)(jnp.zeros((1, 4))))

qml.draw_mpl(qml.transforms.decompose(qc, max_expansion=1), level=1, style="sketch")(
    jnp.zeros((1, num_qubits))
)
plt.show()
