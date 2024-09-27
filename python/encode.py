import pandas as pd
import numpy as np


def parse_biomed_data_to_ndarray(
    dataset_name: str, return_X_y=True
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Function to read in biomedical datasets as .csv-files
       and output as numpy.ndarrays.

       Consistent with the notation in the thesis I use the
       convention:
        - m for the number of examples in the dataset,
        - n for the number of features in the dataset,
        - k for the number of classes in the dataset.

    Args:
        dataset_name (str): Name of the dataset. DO NOT put `.csv` at the end.
        return_X_y (bool, optional): When `True` a tuple of np.ndarrays gets
        returned where X is the feature matrix of shape (m, n) and
        y is the (row) vector of labels of shape (m,).
        When `False` a single np.ndarray gets returned of shape
        (m, n+1) where the one extra dimension is coming from the concatenation
        of X and y (IMPORTANT: In this case y is the first column).
        Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray] | np.ndarray:
        Two arrays of shapes (m,n) and (n,) (in the following refered to as X and y)
        or one single array of shape (m, n+1) (in the following refered to as df).
        X is the feature matrix of shape (m, n).
        y is the label vector of shape (n,) with labels in {-1, +1}.
        df is the concatenation of X and y.T (such that y is the first column).
        See also `return_X_y` for more information.
    """
    df = pd.read_csv("data/" + dataset_name + ".csv")
    df = df.iloc[:, 1:]  # drop the first column (contains only numbering of examples)
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
    