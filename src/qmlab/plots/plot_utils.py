def set_figure_params():
    """Set output figure parameters"""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "lines.linewidth": 2,
            "axes.titlesize": 24,
            "lines.markersize": 10,
        }
    )
