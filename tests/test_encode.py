import sys
import numpy as np

sys.path.append("./python")
from encode import parse_biomed_data_csv_to_numpy


def test_cervical_new_dataset():
    X, y = parse_biomed_data_csv_to_numpy("cervical_new", return_X_y=True)
    assert X.shape == (761, 7)
    assert y.shape == (761,)
    np.testing.assert_allclose(X[0], np.array([52, 5, 16, 4, 1, 37, 37]))
    np.testing.assert_equal(y[0], 1)
    np.testing.assert_allclose(X[-1], np.array([29, 2, 20, 1, 0, 0, 0]))
    np.testing.assert_allclose(y[-1], -1)
    assert X.dtype == np.float32
    assert y.dtype == np.int8
