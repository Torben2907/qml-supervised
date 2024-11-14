import os
from typing import List
import yaml
import pandas as pd
from sklearn import metrics
from qmlab.preprocessing import parse_biomed_data_to_ndarray, reduce_feature_dim
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

dir_out = os.path.join(os.path.dirname(__file__), "../res/")
random_state = 12345
feature_dimension = 2
train_size = 0.7
scoring = "balanced_accuracy"

hyper_parameter_settings = {
    "SVM-RBF": {
        "gamma": [
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
            5.0,
            10.0,
        ],
        "C": [1, 2, 4, 6, 8, 10, 100],
    },
    "SVM-Poly": {
        "degree": [1, 2, 3, 5, 10, 20, 50, 100],
        "C": [1, 2, 4, 6, 8, 10, 100],
    },
}

file_path = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(file_path) as file:
    data_names: list[str] = yaml.safe_load(file)

if __name__ == "__main__":
    for data in data_names:
        for model in hyper_parameter_settings.keys():
            X, y = parse_biomed_data_to_ndarray(data)

            # reducing to the two most important features
            X = reduce_feature_dim(X, output_dimension=feature_dimension)

            num_samples, num_features = X.shape

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=random_state
            )

            grid_search = GridSearchCV(
                estimator=SVC(random_state=random_state),
                param_grid=hyper_parameter_settings[model],
                scoring=scoring,
                n_jobs=16,
                cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state),
            )

            grid_search.fit(X_train, y_train)

            results_with_best_params: dict[str, List] = {
                "best_hypers": [grid_search.best_params_],
                "train_acc": [],
                "test_acc": [],
            }

            train_pred = grid_search.predict(X_train)
            train_acc = metrics.balanced_accuracy_score(y_train, train_pred)

            results_with_best_params["train_acc"].append(train_acc)

            test_pred = grid_search.predict(X_test)
            test_acc = metrics.balanced_accuracy_score(y_test, test_pred)

            results_with_best_params["test_acc"].append(test_acc)

            results_filename = " ".join([model + "_" + data.upper() + "_GridSearchCV"])
            path_out = os.path.join(
                dir_out, results_filename + "_best_hypers_results.csv"
            )

            df = pd.DataFrame.from_dict(results_with_best_params)
            df.to_csv(path_out)
