import os
import math
import yaml
import matplotlib
import matplotlib.pyplot as plt

from qmlab.plotting import set_plot_style, labels_pie_chart
from qmlab.preprocessing import parse_biomed_data_to_ndarray

set_plot_style("ggplot")

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 8,
        "font.sans-serif": ["DejaVu Sans"],
        "text.latex.preamble": r"\usepackage{sfmath} \renewcommand{\familydefault}{\sfdefault}",
        "pgf.rcfonts": False,
    }
)

out_dir = os.path.join(os.path.dirname(__file__), "../figures/")
path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)

ncols = 5
nrows = math.ceil(len(datasets) / ncols)
assert nrows == 2  # this should be true cause we got 9 datasets in total :)
fig, axes = plt.subplots(nrows, ncols, figsize=(len(datasets), 1))
axes = axes.flatten()

# this scaling is adjusted to the width of the .tex layout!
fig.set_size_inches(w=5.90666, h=3.0)

for ax, name in zip(axes, datasets):
    X, y, _ = parse_biomed_data_to_ndarray(name, return_X_y=True)
    name = name.removesuffix("_new").upper()  # make it fancyyyyy
    # colors are hhublue and turquoise!
    labels_pie_chart(
        y,
        title=name,
        ax=ax,
        colors=["#006ab3", "#57bab1"],
    )

for i, ax in enumerate(axes):
    if i >= len(datasets):
        ax.remove()

fig.patch.set_alpha(0)

os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, "imbalance_pie_chart.pgf")
plt.savefig(
    output_path,
    transparent=True,
)
plt.show()
plt.close(fig)
print(f"Pie chart figure successfully saved to {output_path}! \n :-O")
