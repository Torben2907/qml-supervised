import sys

sys.path.append("./python")
import pennylane as qml
import matplotlib.pyplot as plt
from models.qsvc import QSVC
from preprocessing import parse_biomed_data_to_ndarray, reduce_feature_dim
from sklearn.model_selection import train_test_split
from models.variational_classifier import VariationalClassifier


X, y = parse_biomed_data_to_ndarray("sobar_new")
X = reduce_feature_dim(X)
# doing this only for debugging rn
X = X[15:25, :]
y = y[15:25]
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# plot_2d_data_with_train_test_split(X_train, y_train, X_test, y_test)

# fit model
model = VariationalClassifier(
    num_reps=1,
    num_layers=10,
    lr=0.01,
    batch_size=20,
    quantum_device="default.qubit",
    interface="torch",
    diff_method="parameter-shift",
    random_seed=42,
    scaling=1,
    convergence_threshold=10,
    gpu_device="cpu",
    num_steps=100,
)
model.fit(X_train, y_train)
fig, ax = qml.draw_mpl(model.circuit)(params=model.params_, x=X_train[0])
# score the model
print(model.score(X_test, y_test))

quantum_svc = QSVC()
quantum_svc.fit(X_test)

fig.show()
plt.show()
