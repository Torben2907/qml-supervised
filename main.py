import sys

sys.path.append("./python")

from encode import parse_biomed_data_to_ndarray, reduce_feature_dim
from plots.helper_plots import plot_2d_data, plot_2d_data_with_train_test_split
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = parse_biomed_data_to_ndarray("sobar_new")
    X = reduce_feature_dim(X, method="PCA")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    plot_2d_data_with_train_test_split(
        X_train, y_train, X_test, y_test, data_name="sobar_new", save_plot=True
    )
