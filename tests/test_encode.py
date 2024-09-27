import sys
import numpy as np

sys.path.append("./python")
from encode import parse_biomed_data_to_ndarray


def test_cervical_cancer_new_dataset():
    X, y, df = parsing_helper("cervical_new")
    assert X.shape == (761, 7)
    assert y.shape == (761,)
    assert df.shape == (761, 8)

def test_correct_parsing_of_start_and_end_of_file():
    X, y, df = parsing_helper("cervical_new")
    np.testing.assert_allclose(X[0], np.array([52, 5, 16, 4, 1, 37, 37]))
    np.testing.assert_equal(y[0], 1)
    np.testing.assert_allclose(X[-1], np.array([29, 2, 20, 1, 0, 0, 0]))
    np.testing.assert_allclose(y[-1], -1)


def test_datatypes():
    X, y, df = parsing_helper("cervical_new")
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert df.dtype == np.float32


def test_cardiotography_dataset():
    X, y, df = parsing_helper("ctg_new")
    assert X.shape == (1831, 22)
    assert y.shape == (1831,)
    assert df.shape == (1831, 23)

    assert np.count_nonzero(y == -1) == 1655
    assert np.count_nonzero(y == +1) == 176


def parsing_helper(dataset_name: str):
    """helper for getting both the (X, y)
       and the data - arrays. For more details
       see the documentation of
       parse_biomed_data_to_ndarray.

    Args:
        dataset_name (str): Name of dataset. DO NOT use .csv at the end!
    """
    X, y = parse_biomed_data_to_ndarray(dataset_name, return_X_y=True)
    df = parse_biomed_data_to_ndarray(dataset_name, return_X_y=False)
    return X, y, df
