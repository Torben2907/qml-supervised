import matplotlib.lines
import matplotlib.pyplot
import pytest
import matplotlib
from types import SimpleNamespace


@pytest.fixture
def mock_matplotlib(mocker):
    fig = mocker.Mock(spec=matplotlib.pyplot.Figure)
    ax = mocker.Mock(spec=matplotlib.pyplot.Axes)
    line2d = mocker.Mock(name="step", spec=matplotlib.lines.Line2D)
    ax.plot.return_value = (line2d,)

    mpl = mocker.patch("matplotlib.pyplot", autospec=True)
    mocker.patch("matplotlib.pyplot.subplots", return_value=(fig, ax))

    return SimpleNamespace(fig=fig, ax=ax, mpl=mpl)


def test_plot_2d_data(mock_matplotlib):
    pass
