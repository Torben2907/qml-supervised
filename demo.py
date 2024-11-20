from qmlab.preprocessing import parse_biomed_data_to_ndarray, downsample_biomed_data
from qmlab.kernel.iqp_kernel import FidelityIQPKernel
from qmlab.kernel.angle_embedded_kernel import AngleEmbeddedKernel
from sklearn.svm import SVC
from qmlab.utils import run_cross_validation

X, y, feature_names = parse_biomed_data_to_ndarray("haberman_new", return_X_y=True)
# X, y = downsample_biomed_data(X, y)
svm = SVC(kernel="linear", probability=True, random_state=42)
qsvm = AngleEmbeddedKernel(jit=True)
res = run_cross_validation(svm, X, y)
print(res)
