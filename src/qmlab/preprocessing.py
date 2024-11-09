import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler


def parse_biomed_data_to_ndarray(
    dataset_name: str, return_X_y=True
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Function to read in the biomedical datasets as .csv-files
       and output as numpy.ndarrays.

       Consistent with the the thesis we use the notation:
        - m for the number of examples in the dataset,
        - d for the number of features in the dataset,
        - k for the number of classes in the dataset.

    Args:
        dataset_name (str): Name of the dataset. DO NOT put `.csv` at the end.
        return_X_y (bool, optional): When `True` a tuple of np.ndarrays gets
        returned where X is the feature matrix of shape :math:`(m, d)` and
        y is the (row) vector of labels of shape :math:`(m,)`.
        When `False` a single np.ndarray gets returned of shape
        :math:`(m, d+1)` where the one extra dimension is coming from the concatenation
        of X and y (IMPORTANT: In this case y is the first column).
        Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray] | np.ndarray:
        Two arrays of shapes :math:`(m, d)` and :math:`(d,)`
        (in the following refered to as :math:`X` and :math:`y`)
        or a single array of shape :math:`(m, d+1)` (in the following refered to as df).
        X is the feature matrix of shape (m, d).
        y is the label vector of shape (d,) with labels in {-1, +1}.
        df is the concatenation of X and y.T (such that y is the first column).
        See also `return_X_y` for more information.
    """
    df = pd.read_csv("data/" + dataset_name + ".csv")
    df = df.iloc[:, 1:]  # drop first column (contains only numbering of examples)
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
        return (X, y)
    else:
        return df.to_numpy(dtype=np.float32)


def reduce_feature_dim(
    X: np.ndarray, num_features: int = 2, method: str = "PCA"
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
        pca = PCA(n_components=num_features)
        X_reduced = pca.fit_transform(X)
    elif method == "kPCA":
        kpca = KernelPCA(n_components=num_features)
        X_reduced = kpca.fit_transform(X)
    else:
        raise ValueError("provide either PCA or kPCA as reduction method")
    return X_reduced


def scale_data_to_specified_range(
    X: np.ndarray,
    range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    scaling: float = 1.0,
) -> np.ndarray:
    """Scales all values of the feature matrix X to the interval specified
    in `range`.

    Args:
        X (np.ndarray): 2D-feature matrix of dimension m x d.
        range (tuple, optional): scaling interval of floating point values. Defaults to (-np.pi / 2, np.pi / 2).
        scaling (float, optional): Extra scaling of all values of X. Defaults to 1.0 (i.e. no extra scaling).

    Returns:
        np.ndarray: Scaled feature matrix of size :math:`m x d`.
    """
    assert X.ndim == 2, "X must be a 2D-feature array"
    assert len(range) == 2, "range must be a tuple of size 2"
    for vals in range:
        assert isinstance(vals, float), "vals in range must be of type float"
    scaler = MinMaxScaler(range)
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
