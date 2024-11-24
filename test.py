from sklearn.svm import SVC
from qmlab.preprocessing import parse_biomed_data_to_ndarray

X, y, _ = parse_biomed_data_to_ndarray("ctg_new", return_X_y=True)
svm = SVC(kernel="linear")
svm.fit(X, y)
print(svm.score(X, y))
