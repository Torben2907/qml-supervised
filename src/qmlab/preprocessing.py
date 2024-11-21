import os
from typing import List, Literal, Tuple, overload
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler
import itertools as it
from sklearn.utils import resample

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/")


def parse_biomed_data_to_df(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR + dataset_name + ".csv")
    df = df.iloc[:, 1:]
    df["V1"] = (2 * df["V1"]) - 1
    label_column = df.pop("V1")
    df["V1"] = label_column
    return df


@overload
def parse_biomed_data_to_ndarray(
    dataset_name: str, return_X_y: Literal[True]
) -> Tuple[np.ndarray, np.ndarray, List[str]]: ...


@overload
def parse_biomed_data_to_ndarray(
    dataset_name: str, return_X_y: Literal[False]
) -> Tuple[np.ndarray, List[str]]: ...


def parse_biomed_data_to_ndarray(
    dataset_name: str, return_X_y: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]] | Tuple[np.ndarray, List[str]]:
    """Function to read in the biomedical datasets as .csv-files
       and output as `numpy.ndarrays`.

       Consistent with the the thesis the notation
        - :math:`m` for the number of examples in the dataset,
        - :math:`d` for the number of features in the dataset,
        - :math:`k` for the number of classes in the dataset
        is used.

    Args:
        dataset_name (str): Name of the dataset. DO NOT put `.csv` at the end.
        return_X_y (bool, optional): When `True` a tuple of np.ndarrays gets
        returned where X is the feature matrix of shape :math:`(m, d)` and
        y is the (row) vector of labels of shape :math:`(m,)`.
        When `False` a single np.ndarray gets returned of shape
        :math:`(m, d+1)` where the one extra dimension is coming from the concatenation
        of X and y (IMPORTANT: In this case y is the first column).
        Defaults to True.
        It will ALWAYS return a list of strings which are the features of the
        data as the third or second return type.

    Returns:
        tuple[np.ndarray, np.ndarray, List[str]] | np.ndarray:
        Two arrays of shapes :math:`(m, d)` and :math:`(d,)`
        (in the following refered to as :math:`X` and :math:`y`)
        or a single array of shape :math:`(m, d+1)` (in the following refered to as df).
        X is the feature matrix of shape (m, d).
        y is the label vector of shape (d,) with labels in {-1, +1}.
        feature_names a list containing the names of the features as srings.
        df is the concatenation of X and y.T (such that y is the first column).
        See also `return_X_y` for more information.
    """
    try:
        df = pd.read_csv(DATA_DIR + dataset_name + ".csv")
    except FileNotFoundError as fnf:
        raise FileNotFoundError(
            f"Dataset {dataset_name} not found! Did you spell it correctly?\n"
            "Remember do not add .csv at the end! See docstring for more."
        ) from fnf
    except Exception as e:
        raise Exception(f"An error occured while parsing the dataset: {e}") from e
    df = df.iloc[
        :, 1:
    ]  # drop first column (assuming it contains numbering of examples)
    feature_names = list(df.columns)
    feature_names.remove("V1")
    if return_X_y:
        # "V1" is the column of labels
        X: np.ndarray = df.iloc[:, df.columns != "V1"].to_numpy(dtype=np.float32)
        y: np.ndarray = (
            df.iloc[:, df.columns == "V1"]
            .to_numpy(dtype=np.int8)
            .reshape(
                X.shape[0],
            )
        )  # y as row vector (n,)
        y = (2 * y) - 1
        return (X, y, feature_names)
    else:
        return (df.to_numpy(dtype=np.float32), feature_names)


def downsample_biomed_data(
    X: np.ndarray, y: np.ndarray, replace: bool = True, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    X_pos, X_neg = X[y == +1], X[y == -1]
    y_pos, y_neg = y[y == +1], y[y == -1]

    X_neg_downsample, y_neg_downsample = resample(
        X_neg, y_neg, replace=replace, n_samples=len(X_pos), random_state=random_state
    )

    X_down = np.vstack((X_pos, X_neg_downsample))
    y_down = np.hstack((y_pos, y_neg_downsample))

    return X_down, y_down


def upsample_biomed_data(
    X: np.ndarray, y: np.ndarray, replace: bool = True, random_state: int = 42
):
    X_pos, X_neg = X[y == +1], X[y == -1]
    y_pos, y_neg = y[y == +1], y[y == -1]

    X_pos_upsampled, y_pos_upsampled = resample(
        X_pos, y_pos, replace=replace, n_samples=len(X_neg), random_state=random_state
    )

    X_up = np.vstack((X_pos_upsampled, X_neg))
    y_up = np.hstack((y_pos_upsampled, y_neg))

    return X_up, y_up


def subsample_features(
    X: np.ndarray,
    feature_names: List[str],
    num_features_to_subsample: int,
    all_possible_combinations: bool = False,
) -> List[Tuple[np.ndarray, List[str]]]:
    feature_dimension = X.shape[1]
    if feature_dimension != len(feature_names):
        raise ValueError(
            "The length of `feature_names` must match the number of columns in `X`."
        )
    if num_features_to_subsample > feature_dimension:
        raise ValueError(
            "`num_features_to_subsample` cannot be greater than the total number of features."
        )
    subsampled_results: List[Tuple[np.ndarray, List[str]]] = []

    if all_possible_combinations is True:
        all_combs = list(
            it.combinations(range(feature_dimension), num_features_to_subsample)
        )
        for combination in all_combs:
            subsampled_X = X[:, combination]
            subsampled_feature_names = [feature_names[i] for i in combination]
            subsampled_results.append((subsampled_X, subsampled_feature_names))
    else:
        for start_idx in range(0, feature_dimension, num_features_to_subsample):
            end_idx = start_idx + num_features_to_subsample
            subsampled_indices = list(range(start_idx, min(end_idx, feature_dimension)))
            # If end_idx exceeds the number of features, wrap around to the start
            if end_idx > feature_dimension:
                subsampled_indices += list(range(end_idx - feature_dimension))

            subsampled_X = X[:, subsampled_indices]
            subsampled_feature_names = [feature_names[i] for i in subsampled_indices]

            subsampled_results.append((subsampled_X, subsampled_feature_names))

    return subsampled_results


def reduce_feature_dim(
    X: np.ndarray, output_dimension: int = 2, method: str = "PCA"
) -> np.ndarray:
    """Reduces the dimension of the input feature matrix X which is
        assummed to have shape (m, d), where
        - m is the number of examples,
        - d is the number of features.

    Args:
        X (np.ndarray): feature matrix of shape `(m, d)`
        num_features (int, optional): the number of features of the reduced feature
        matrix. Defaults to 2.
        method (str, optional): method of dimensionality reduction. Defaults to "PCA".
        Supported methods are "PCA" and "kPCA" (short for kernel PCA using rbf kernel).

    Raises:
        ValueError: if no supported method for dimensionality reduction is provided.

    Returns:
        np.ndarray: the reduced feature matrix of shape `(m, num_features)`.
    """
    assert X.ndim == 2, "X must be a 2D-array, i.e. matrix"
    if method == "PCA":
        pca = PCA(n_components=output_dimension)
        X_reduced = pca.fit_transform(X)
    elif method == "kPCA":
        kpca = KernelPCA(n_components=output_dimension)
        X_reduced = kpca.fit_transform(X)
    else:
        raise ValueError("provide either PCA or kPCA as reduction method")
    return X_reduced


def scale_to_specified_interval(
    X: np.ndarray,
    interval: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    scaling: float = 1.0,
) -> np.ndarray:
    """Scales all values of the feature matrix X to the interval specified
    in `interval`.

    Args:
        X (np.ndarray): 2D-feature matrix of dimension m x d.
        interval (tuple, optional): scaling interval of floating point values. Defaults to (-np.pi / 2, np.pi / 2).
        scaling (float, optional): Extra scaling of all values of X. Defaults to 1.0 (i.e. no extra scaling).

    Returns:
        np.ndarray: Scaled feature matrix of size :math:`m x d`.
    """
    assert X.ndim == 2, "X must be a 2D-feature array"
    assert len(interval) == 2, "interval must be a tuple of size 2"
    for vals in interval:
        assert isinstance(vals, float), "vals in interval must be of type float!"
    scaler = MinMaxScaler(interval)
    scaler.fit(X)
    X = scaler.transform(X)
    return X * scaling


def pad_and_normalize_data(X: np.ndarray, pad_with: float = 0.0) -> np.ndarray:
    r"""Padding and normalization for Amplitude Embedding.

    Remember that padding is necessary because we're mapping
    d features to :math:`\lceil \log_2(n) \rceil` qubits and in the case
    that :math:`2^d > n` we need to pad the dimension of our feature vector
    to match the dimension of the output state vector.

    Args:
        X (np.ndarray): feature matrix of shape (m, d) to be padded and normalized.
        pad_with (float, optional): Value to pad the missing entries with. Defaults to 0.0.

    Returns:
        np.ndarray: padded and normalized feature matrix. Now ready to be used for
        Amplitude Embeddings.
    """
    assert X.ndim == 2, "X must be a 2D-feature array"

    num_features = X.shape[1]
    num_qubits = int(np.ceil(np.log2(num_features)))
    max_num_features = 2**num_qubits
    padding_amount = max_num_features - num_features
    padding = np.empty(shape=(len(X), padding_amount))
    padding.fill(pad_with)
    padding /= max_num_features

    X_pad = np.c_[X, padding]
    X_norm = np.divide(X_pad, np.linalg.norm(X_pad, axis=1)[:, None])
    return X_norm
