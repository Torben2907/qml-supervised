import sys
import numpy as np
import pytest

sys.path.append("./python")
from preprocessing import parse_biomed_data_to_ndarray, scale_data_to_range


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


data_with_associated_shape = [
    {"name": "wdbc_new", "shape": (569, 30), "pos": 212, "neg": 357},
    {"name": "fertility_new", "shape": (100, 9), "pos": 12, "neg": 88},
    {"name": "haberman_new", "shape": (306, 3), "pos": 81, "neg": 225},
]


@pytest.mark.parametrize("data", data_with_associated_shape)
def test_shape_datasets(data):
    name = data["name"]
    shape = data["shape"]
    pos = data["pos"]
    neg = data["neg"]
    X, y, df = parsing_helper(name)
    assert X.shape == shape
    assert y.shape == (shape[0],)
    assert df.shape == (shape[0], shape[1] + 1)

    assert np.count_nonzero(y == -1) == neg
    assert np.count_nonzero(y == +1) == pos


@pytest.mark.parametrize(
    "range",
    [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-5.0, 5.0), (-2 * np.pi, np.pi)],
)
def test_scale_data_to_range(range):
    X, _ = parse_biomed_data_to_ndarray("ctg_new")
    X_scaled = scale_data_to_range(X, range)
    assert np.any((X_scaled < range[0]) | (X_scaled > range[1]))


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
