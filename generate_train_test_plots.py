"""script to create 2-dimensional plots of all datasets with train and 
    test split
"""

from qmlab.preprocessing import parse_biomed_data_to_ndarray, reduce_feature_dim
from qmlab.plots.plot_data import plot_2d_data_with_train_test_split
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data_names = [
        "cervical_new",
        "sobar_new",
        "ctg_new",
        "fertility_new",
        "hcv_new",
        "nafld_new",
        "heroin_new",
        "wdbc_new",
    ]
    for name in data_names:
        X, y = parse_biomed_data_to_ndarray(name)
        X = reduce_feature_dim(X, method="PCA")
        # might change the train-test-split later
        # for now 80% for training 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        plot_2d_data_with_train_test_split(
            X_train, y_train, X_test, y_test, data_name=name, save_plot=True
        )
