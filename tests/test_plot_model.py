import matplotlib.lines
import matplotlib.pyplot
import pytest
import matplotlib
import numpy as np
from types import SimpleNamespace
from qmlab.plotting import plot_decision_boundaries

random_state = 12345


@pytest.fixture
def mock_matplotlib(mocker):
    fig = mocker.Mock(spec=matplotlib.pyplot.Figure)
    ax = mocker.Mock(spec=matplotlib.pyplot.Axes)
    line2d = mocker.Mock(name="step", spec=matplotlib.lines.Line2D)
    ax.plot.return_value = (line2d,)

    mpl = mocker.patch("matplotlib.pyplot", autospec=True)
    clf = mocker.patch("sklearn.svm.SVC", autospec=True)
    mocker.patch("matplotlib.pyplot.subplots", return_value=(fig, ax))

    return SimpleNamespace(fig=fig, ax=ax, mpl=mpl, clf=clf)


def test_plot_decision_boundaries(mock_matplotlib):
    mpl = mock_matplotlib.mpl
    clf = mock_matplotlib.clf

    X = np.array(
        [
            [[0.3, 1.1], [-0.2, -1.0]],
            [[1.7, -1.9], [0.6, 0.4]],
        ]
    )
    X_train, X_test = X
    y_train = np.array([-1, 1])
    y_test = np.array([1, -1])

    clf.fit(X_train, y_train)
    plot_decision_boundaries(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert len(mpl.mock_calls) == 10
