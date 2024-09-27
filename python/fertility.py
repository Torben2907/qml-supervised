import pennylane as qml
import sys
from pennylane import numpy as pnp
import pandas as pd
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pnp.random.seed(0) # reproducability

# default pennylane device
dev = qml.device("default.qubit")

num_qubits = 2
num_layers = 6
weights_init = (1 / 100) * pnp.random.randn(
    num_layers, num_qubits, 3, requires_grad=True
)
bias_init = pnp.array(0.0, requires_grad=True)
opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

# DATASET
df = pd.read_csv("data/sobar_new.csv")
df = df.iloc[:, 1:]  # drop first column as this contains just a numeration of samples
X, y = df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy()
y = (2 * y) - 1  # [0, 1] -> [-1, +1]

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

padding = pnp.ones((len(X_reduced), X_reduced.shape[1])) * 0.1
X_pad = pnp.c_[X_reduced, padding]

normalization = pnp.sqrt(pnp.sum(X_pad**2, -1))
X_norm = (X_pad.T / normalization).T
print(f"First X sample normalized {X_norm[0]}")


def get_angles(x: pnp.ndarray) -> pnp.ndarray:
    beta0 = 2 * pnp.arcsin(
        pnp.sqrt(x[1] ** 2) / pnp.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12)
    )
    beta1 = 2 * pnp.arcsin(
        pnp.sqrt(x[3] ** 2) / pnp.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12)
    )
    beta2 = 2 * pnp.arcsin(pnp.linalg.norm(x[2:]) / pnp.linalg.norm(x))

    return pnp.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def state_preparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)


features = pnp.array([get_angles(x) for x in X_norm], requires_grad=False)
print(f"first encoded feature sample: {features[0]}")


plt.figure()
plt.scatter(X_reduced[:, 0][y == 1], X_reduced[:, 1][y == 1], c="b", marker="o", ec="k")
plt.scatter(
    X_reduced[:, 0][y == -1], X_reduced[:, 1][y == -1], c="r", marker="o", ec="k"
)
plt.title("Original data")
plt.show()

plt.figure()
dim1 = 0
dim2 = 1
plt.scatter(X_norm[:, dim1][y == 1], X_norm[:, dim2][y == 1], c="b", marker="o", ec="k")
plt.scatter(
    X_norm[:, dim1][y == -1], X_norm[:, dim2][y == -1], c="r", marker="o", ec="k"
)
plt.title(f"Padded and normalised data (dims {dim1} and {dim2})")
plt.show()

plt.figure()
dim1 = 0
dim2 = 3
plt.scatter(
    features[:, dim1][y == 1], features[:, dim2][y == 1], c="b", marker="o", ec="k"
)
plt.scatter(
    features[:, dim1][y == -1], features[:, dim2][y == -1], c="r", marker="o", ec="k"
)
plt.title(f"Feature vectors (dims {dim1} and {dim2})")
plt.show()


def layer(layer_weights):
    for wire in range(num_qubits):
        qml.Rot(*layer_weights[wire], wires=wire)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit(weights, feats):
    state_preparation(feats)

    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, feats):
    return circuit(weights, feats) + bias


def square_loss(label, predictions):
    return pnp.mean((label - qml.math.stack(predictions)) ** 2)


def cost(weights, bias, X, y):
    predictions = variational_classifier(weights, bias, X.T)
    return square_loss(y, predictions)


def accuracy(labels, predictions):
    acc = sum(
        abs(label - prediction) < 1e-5 for label, prediction in zip(labels, predictions)
    )
    acc = acc / len(labels)
    return acc


def training_loop(
    feats_train,
    y_train,
    feats_val,
    y_val,
    weights,
    bias,
    opt,
    batch_size=5,
    epochs=100,
    print_training=False
):
    num_train = len(y_train)
    for it in range(epochs):
        batch_index = pnp.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        y_train_batch = y_train[batch_index]
        weights, bias, _, _ = opt.step(
            cost, weights, bias, feats_train_batch, y_train_batch
        )

        predictions_train = pnp.sign(
            variational_classifier(weights, bias, feats_train.T)
        )
        predictions_val = pnp.sign(variational_classifier(weights, bias, feats_val.T))

        acc_train = accuracy(y_train, predictions_train)
        acc_val = accuracy(y_val, predictions_val)
        if print_training: 
            if (it + 1) % 2 == 0:
                _cost = cost(weights, bias, feats_train, y_train)
                print(
                    f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
                    f"Acc train: {acc_train:0.7f} | Acc val: {acc_val:0.7f}"
                )
    
    return weights, bias 

num_data = len(y)
num_train = int(0.75 * num_data)
index = pnp.random.permutation(range(num_data))
feats_train = features[index[:num_train]]
y_train = y[index[:num_train]]
feats_val = features[index[num_train:]]
y_val = y[index[num_train:]]

X_train = X_reduced[index[:num_train]]
X_val = X_reduced[index[num_train:]]

weights_final, bias_final = training_loop(
    feats_train,
    y_train,
    feats_val,
    y_val,
    weights=weights_init,
    bias=bias_init,
    opt=opt,
    batch_size=5,
    epochs=60,
)

plt.figure(dpi=300)
cm = plt.cm.RdBu
# cm = plt.colormaps["viridis"]

# make data for decision regions
xx, yy = pnp.meshgrid(pnp.linspace(-20, 15, 30), pnp.linspace(-12, 12, 30))
X_grid = pnp.array([pnp.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())])

# preprocess grid points like data inputs above
padding = 0.1 * pnp.ones((len(X_grid), 2))
X_grid = pnp.c_[X_grid, padding]  # pad each input
normalization = pnp.sqrt(pnp.sum(X_grid**2, -1))
X_grid = (X_grid.T / normalization).T  # normalize each input
features_grid = pnp.array([get_angles(x) for x in X_grid])  # angles are new features
predictions_grid = variational_classifier(weights_final, bias_final, features_grid.T)
Z = pnp.reshape(predictions_grid, xx.shape)

# plot decision regions
levels = pnp.arange(-1, 1.1, 0.1)
cnt = plt.contourf(xx, yy, Z, levels=levels, cmap=cm, alpha=0.8, extend="both")
plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
for color, label in zip(["b", "r"], [1, -1]):
    plot_x = X_train[:, 0][y_train == label]
    plot_y = X_train[:, 1][y_train == label]
    plt.scatter(plot_x, plot_y, c=color, marker="o", ec="k", label=f"class {label} train")
    plot_x = (X_val[:, 0][y_val == label],)
    plot_y = (X_val[:, 1][y_val == label],)
    plt.scatter(plot_x, plot_y, c=color, marker="^", ec="k", label=f"class {label} validation")

print(X_val)
# plt.legend()
plt.show()