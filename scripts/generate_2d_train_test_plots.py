"""script to create 2-dimensional plots of all datasets with train and 
    test split. I'm keeping this for completeness but I didn't use those further,
    since scaling down the dimensions so vastly using PCA wasn't a good approach.
"""

import os
import yaml
from qmlab.preprocessing import parse_biomed_data_to_ndarray, reduce_feature_dim
from qmlab.plotting import plot_2d_data_with_train_test_split
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "../data_names.yaml")
        ) as file:
            data_names = yaml.safe_load(file)
    except FileNotFoundError as fnf:
        print(f"FileNotFoundError: {fnf}")

    import matplotlib.pyplot as plt

    for name in data_names:
        plt.clf()  # Clear the current figure
        X, y, _ = parse_biomed_data_to_ndarray(name, return_X_y=True)
        X = reduce_feature_dim(X, method="PCA")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        plot_2d_data_with_train_test_split(
            X_train, y_train, X_test, y_test, data_name=name, save_plot=True
        )
