import os
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay, auc
from ..kernel.qsvm import QSVC
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 8,
        # "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def run_cv(
    clf: SVC | QSVC,
    X: NDArray,
    y: NDArray,
    num_splits: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    scv = StratifiedKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    accuracies = []
    # f1_scores = []
    precisions = []
    auc_scores = []
    mccs = []
    for train_idx, test_idx in scv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = metrics.accuracy_score(y_test, y_pred)
        # f1 = metrics.f1_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        accuracies.append(accuracy)
        # f1_scores.append(f1)
        precisions.append(precision)
        auc_scores.append(auc)
        mccs.append(mcc)

    results = {
        "accuracy": np.mean(accuracies).item(),
        # "f1": np.mean(f1_scores).item(),
        "precision": np.mean(precisions).item(),
        "auc": np.mean(auc_scores).item(),
        "mcc": np.mean(mccs).item(),
    }

    return results


def run_cv_roc_analysis(
    clf: SVC | QSVC,
    X: NDArray,
    feature_names: List[str],
    y: NDArray,
    num_splits: int = 10,
    random_state: int = 42,
    output_dir: str = "roc_plots",
    kernel_name: str = "kernel",
    dataset_name: str = "dataset",
) -> List:

    scv = StratifiedKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(3, 3))  # Smaller figure size
    fig.set_size_inches(
        w=2.95333, h=1.5
    )  # Half of the original size for LaTeX rendering
    plot_files = []

    for fold, (train, test) in enumerate(scv.split(X, y)):
        clf.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            clf,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == num_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="#006ab3",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="#b5cbd6",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="FPR",
        ylabel="TPR",
        title=f"{kernel_name}",
    )
    ax.legend(loc="lower right")

    # Save the overall plot
    overall_plot_path = os.path.join(
        output_dir,
        generate_plot_filename(dataset_name, kernel_name, feature_names),
    )
    ax.get_legend().remove()  # Remove legend
    plt.savefig(overall_plot_path, bbox_inches="tight", transparent=True)
    plot_files.append(overall_plot_path)

    plt.close(fig)
    return plot_files


def generate_plot_filename(
    dataset_name: str,
    kernel_name: str,
    feature_names: List[str],
    fold: int | None = None,
) -> str:
    """
    Generate a concise filename for saving plots.

    Args:
        dataset_name: Name of the dataset.
        kernel_name: Name of the kernel.
        feature_names: List of feature names used in the fold.
        fold: Fold number (optional).

    Returns:
        A string representing the filename.
    """
    max_features_to_display = 3
    # Create a short identifier for features (e.g., first 3 feature names)
    feature_summary = "_".join(feature_names[:max_features_to_display])
    if len(feature_names) > max_features_to_display:
        feature_summary += f"_and_{len(feature_names) - max_features_to_display}_more"

    # Construct the filename
    filename = f"{dataset_name}_{kernel_name}_{feature_summary}"
    if fold is not None:
        filename += f"_fold{fold}"
    filename += ".pgf"

    # Ensure filename is safe for the filesystem
    return filename.replace(" ", "_").replace(",", "").replace("/", "_")
