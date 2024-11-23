import os
import yaml
from qmlab.preprocessing import parse_biomed_data_to_ndarray
from qmlab.plotting import set_plot_style
import matplotlib.pyplot as plt
import numpy as np

set_plot_style("fivethirtyeight")

datasets_with_associated_attrs = [
    {"name": "wdbc_new", "shape": (569, 30), "pos": 212, "neg": 357},
    {"name": "fertility_new", "shape": (100, 9), "pos": 12, "neg": 88},
    {"name": "haberman_new", "shape": (306, 3), "pos": 81, "neg": 225},
    {"name": "cervical_new", "shape": (761, 7), "pos": 17, "neg": 744},
    {"name": "hcv_new", "shape": (546, 12), "pos": 20, "neg": 526},
    {"name": "nafld_new", "shape": (74, 9), "pos": 22, "neg": 52},
    {"name": "heroin_new", "shape": (942, 11), "pos": 97, "neg": 845},
    {"name": "ctg_new", "shape": (1831, 22), "pos": 176, "neg": 1655},
    {"name": "sobar_new", "shape": (72, 19), "pos": 21, "neg": 51},
]

path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    data_names: list[str] = yaml.safe_load(file)
data_names = [name.removesuffix("_new").upper() for name in data_names]

colors = [
    "#006ab3",
    "#003964",
    "#b5cbd6",
    "#57bab1",
    "#8cb110",
    "#ee7f00",
    "#be0a26",
    "yellow",
    "pink",
]

plt.xticks(rotation=45)

fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(True)
ax.set_ylabel("Distribution of POS Labels")
ax.set_title("Boxplot of Simulated POS Label Distributions")

# Simulating data for each dataset based on pos (centered around pos with some variability)
np.random.seed(42)  # For reproducibility
simulated_data = [
    np.random.normal(data["pos"], 5, 50) for data in datasets_with_associated_attrs
]
# Create boxplots using the simulated data
bplot = ax.boxplot(simulated_data, patch_artist=True, label=data_names)

# Fill the boxplots with colors and set outline colors
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)

# Add mean markers
means = [np.mean(dataset) for dataset in simulated_data]
for i, mean in enumerate(
    means, start=1
):  # `start=1` for 1-based indexing of boxplot positions
    ax.scatter(i, mean, color="black", marker="o", label="Mean" if i == 1 else "")

ax.legend(loc="upper right")

plt.show()
