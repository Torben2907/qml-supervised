from qmlab.preprocessing import parse_biomed_data_to_ndarray
from sklearn.svm import SVC
from qmlab.kernel import AngleEmbeddedKernel
from qmlab.utils import run_cross_validation

X, y, feature_names = parse_biomed_data_to_ndarray("haberman_new", return_X_y=True)
svm = SVC(kernel="linear", probability=True, random_state=42)
qsvm = AngleEmbeddedKernel(jit=True)

classical_res = run_cross_validation(svm, X, y)
print(classical_res)


quantum_res = run_cross_validation(qsvm, X, y)
print(quantum_res)
