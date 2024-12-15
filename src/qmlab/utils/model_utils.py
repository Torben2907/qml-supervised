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
from scipy.stats import norm

matplotlib.use("pgf")
import matplotlib.pyplot as plt

# configuration for export to TeX.
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 6,
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
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float | List[float]]]:
    scv = StratifiedKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    accuracies = []
    aucs = []
    mccs = []

    for train_idx, test_idx in scv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = metrics.accuracy_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)

        accuracies.append(accuracy)
        aucs.append(auc)
        mccs.append(mcc)

    accuracy_ci = compute_confidence_interval(accuracies, confidence)
    auc_ci = compute_confidence_interval(aucs, confidence)
    mcc_ci = compute_confidence_interval(mccs, confidence)

    results = {
        "accuracy": {"mean": np.mean(accuracies).tolist(), "CI": accuracy_ci},
        "auc": {"mean": np.mean(aucs).tolist(), "CI": auc_ci},
        "mcc": {"mean": np.mean(mccs).tolist(), "CI": mcc_ci},
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
    rotation_for_angle: str = "X",
) -> List:

    scv = StratifiedKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(3, 3))  # Smaller figure size
    fig.set_size_inches(w=2.5, h=1.5)  # for LaTeX rendering
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
        color="#006ab3",  # hhublue
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
        color="#b5cbd6",  # hhured
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    if kernel_name == "Angle":
        ax.set(
            xlabel="FPR",
            ylabel="TPR",
            title=f"{kernel_name}($\sigma = {rotation_for_angle}$)",
        )
    else:
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
    Generate the filename for saving plots.

    Args:
        dataset_name: Name of the dataset.
        kernel_name: Name of the kernel.
        feature_names: List of feature names used in the fold.
        fold: Fold number (optional).

    Returns:
        Filename as string.
    """
    max_features_to_display = 3
    feature_summary = "_".join(feature_names[:max_features_to_display])
    if len(feature_names) > max_features_to_display:
        feature_summary += f"_and_{len(feature_names) - max_features_to_display}_more"

    filename = f"{dataset_name}_{kernel_name}_{feature_summary}"
    if fold is not None:
        filename += f"_fold{fold}"
    filename += ".pgf"

    return filename.replace(" ", "_").replace(",", "").replace("/", "_")


def compute_confidence_interval(
    data: list[float], confidence: float = 0.95
) -> List[float]:
    """
    Compute the confidence interval for a given data set.
    """
    k = len(data)  # k-folds
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(k)  # Standard error
    z_score = norm.ppf((1 + confidence) / 2)  # Z-score for the confidence level
    margin = z_score * std_err
    CI = np.array([mean - margin, mean + margin])
    return CI.tolist()
