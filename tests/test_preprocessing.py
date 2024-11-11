import numpy as np
import pytest

from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    scale_data_to_specified_range,
    pad_and_normalize_data,
)


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


# Using the information about the datasets from
# https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00283-6
data_with_associated_attrs = [
    {"name": "wdbc_new", "shape": (569, 30), "pos": 212, "neg": 357},
    {"name": "fertility_new", "shape": (100, 9), "pos": 12, "neg": 88},
    {"name": "haberman_new", "shape": (306, 3), "pos": 81, "neg": 225},
    {"name": "cervical_new", "shape": (761, 7), "pos": 17, "neg": 744},
    {"name": "hcv_new", "shape": (546, 12), "pos": 20, "neg": 526},
    {"name": "nafld_new", "shape": (74, 9), "pos": 22, "neg": 52},
    {"name": "heroin_new", "shape": (942, 11), "pos": 97, "neg": 845},
    {"name": "ctg_new", "shape": (1831, 22), "pos": 176, "neg": 1655},
    {"name": "sobar_new", "shape": (72, 19), "pos": 21, "neg": 51},
]


@pytest.mark.parametrize("data", data_with_associated_attrs)
def test_shape_datasets(data):
    name = data["name"]
    shape = data["shape"]
    pos = data["pos"]
    neg = data["neg"]
    X, y, df = parsing_helper(name)
    assert X.shape == shape
    assert y.shape == (shape[0],)
    assert df.shape == (shape[0], shape[1] + 1)
    assert (pos + neg) == shape[0]

    assert np.count_nonzero(y == -1) == neg
    assert np.count_nonzero(y == +1) == pos


# we specify a variety of ranges to test the scaling
@pytest.mark.parametrize(
    "range",
    [
        (-np.pi, np.pi),
        (-np.pi / 2, np.pi / 2),
        (-5.0, 5.0),
        (-2 * np.pi, np.pi),
        (0.0, np.pi / 2),
    ],
)
@pytest.mark.parametrize("data", data_with_associated_attrs)
def test_scale_data_to_range(range, data):
    X, _ = parse_biomed_data_to_ndarray(data["name"])
    X_scaled = scale_data_to_specified_range(X, range)
    assert np.any((X_scaled <= range[0]) | (X_scaled >= range[1]))


def test_pad_and_normalize_data_with_zeros():
    x = np.array([[1 / 2, 1 / 2, 1 / 2]])
    x_norm = pad_and_normalize_data(x, pad_with=0.0)
    desired = (1 / np.sqrt(3)) * np.array([[1, 1, 1, 0]])
    np.testing.assert_allclose(x_norm, desired, rtol=1e-7, atol=1e-9)


def test_pad_and_normalize_data_with_ones():
    x = np.array([[1 / 2, 1 / 2, 1 / 2]])
    x_norm = pad_and_normalize_data(x, pad_with=1.0)
    desired = 4 / (2 * np.sqrt(13)) * np.array([[1.0, 1.0, 1.0, 1 / 2]])
    np.testing.assert_allclose(x_norm, desired, rtol=1e-7, atol=1e-9)


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
