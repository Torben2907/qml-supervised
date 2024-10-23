import sys

sys.path.append("./python")
from preprocessing import parse_biomed_data_to_ndarray, reduce_feature_dim
import pennylane as qml

X, y = parse_biomed_data_to_ndarray("cervical_new", return_X_y=True)
X = reduce_feature_dim(X)
num_samples, num_features = X.shape

dev = qml.device("default.qubit", wires=num_features)


@qml.qnode(dev)
def circuit(*, feature_vector):
    qml.AmplitudeEmbedding(
        features=feature_vector,
        wires=range(num_features),
        normalize=True,
        pad_with=0.0,
    )
    return qml.expval(qml.Z(0)), qml.state()


print(X[0])
res, state = circuit(feature_vector=X[0])
print(res)
print(state)
