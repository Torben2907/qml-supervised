from qmlab.preprocessing import parse_biomed_data_to_ndarray, downsample_biomed_data
from qmlab.kernel.iqp_kernel import FidelityIQPKernel
from qmlab.kernel.angle_embedded_kernel import AngleEmbeddingKernel
from qmlab.utils import run_shuffle_split

X, y, feature_names = parse_biomed_data_to_ndarray("haberman_new", return_X_y=True)
X, y = downsample_biomed_data(X, y)
qsvm = AngleEmbeddingKernel(jit=True)
res = run_shuffle_split(qsvm, X, y)
print(res)
