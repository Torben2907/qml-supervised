from typing import Tuple, List
import numpy as np
import pytest
import scipy
import scipy.special

from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    scale_to_specified_range,
    pad_and_normalize_data,
    subsample_features,
)


def test_incorrect_file_name():
    with pytest.raises(FileNotFoundError):
        parse_biomed_data_to_ndarray(
            "my_data_that_will_never_exist_cause_who_would_name_it_like_this_fr"
        )


def test_return_X_y():
    res = parse_biomed_data_to_ndarray("haberman_new", return_X_y=True)
    assert len(res) == 3


def test_return_dataframe():
    res = parse_biomed_data_to_ndarray("haberman_new", return_X_y=False)
    assert len(res) == 2


def test_correct_parsing_of_start_and_end_of_file():
    X, y, _ = parse_biomed_data_to_ndarray("cervical_new")
    np.testing.assert_allclose(X[0], np.array([52, 5, 16, 4, 1, 37, 37]))
    np.testing.assert_equal(y[0], 1)
    np.testing.assert_allclose(X[-1], np.array([29, 2, 20, 1, 0, 0, 0]))
    np.testing.assert_allclose(y[-1], -1)


def test_correct_feature_names():
    _, _, feature_names = parse_biomed_data_to_ndarray("haberman_new")
    assert feature_names == ["V2", "V3", "V4"]


def test_datatypes_1():
    X, y, _ = parse_biomed_data_to_ndarray("cervical_new")
    assert X.dtype == np.float32
    assert y.dtype == np.int8


def test_datatypes_2():
    df, _ = parse_biomed_data_to_ndarray("cervical_new", return_X_y=False)
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
def test_shape_datasets_X_y(data):
    name = data["name"]
    shape = data["shape"]
    pos = data["pos"]
    neg = data["neg"]
    X, y, _ = parse_biomed_data_to_ndarray(name)
    assert X.shape == shape
    assert y.shape == (shape[0],)
    assert (pos + neg) == shape[0]

    assert np.count_nonzero(y == -1) == neg
    assert np.count_nonzero(y == +1) == pos


@pytest.mark.parametrize("data", data_with_associated_attrs)
def test_shape_datasets_dataframe(data):
    name = data["name"]
    shape = data["shape"]
    pos = data["pos"]
    neg = data["neg"]
    df, _ = parse_biomed_data_to_ndarray(name, return_X_y=False)
    assert df.shape == (shape[0], shape[1] + 1)
    assert (pos + neg) == shape[0]


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
    (X, _, _) = parse_biomed_data_to_ndarray(data["name"])
    X_scaled = scale_to_specified_range(X, range)
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


def test_subsample_features_no_wrap_around():
    """no rotation simply means that total number of features // number of features to subsample!"""
    X, _, feature_names = parse_biomed_data_to_ndarray("haberman_new")
    results = subsample_features(
        X, feature_names, num_features_to_subsample=1, all_possible_combinations=False
    )
    assert len(results) == 3
    first_feature_vector, first_feature_name = results[0]
    # subsample_features didn't change num of examples:
    assert len(first_feature_vector) == 306
    assert first_feature_vector[0] == 34
    assert first_feature_vector[-1] == 77
    assert first_feature_name == ["V2"]


def test_subsample_features_wrap_around():
    X, _, feature_names = parse_biomed_data_to_ndarray("haberman_new")
    results = subsample_features(
        X, feature_names, num_features_to_subsample=2, all_possible_combinations=False
    )
    assert len(results) == 2


def test_subsample_features_all_possible_combinations():
    X, _, feature_names = parse_biomed_data_to_ndarray("ctg_new")
    results = subsample_features(
        X, feature_names, num_features_to_subsample=2, all_possible_combinations=True
    )
    assert len(results) == scipy.special.binom(X.shape[1], 2)
