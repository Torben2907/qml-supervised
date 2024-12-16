"""Creates figure of a perfect model that has a sensitivity of 1 and 
specificity of 0.
"""

import matplotlib
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


fig = plt.figure(figsize=(2, 2))
fig.set_size_inches(w=2.5, h=1.5)  # Half of the original size for LaTeX rendering
plt.plot([0, 0, 1], [0, 1, 1], color="#006ab3", linewidth=4)
plt.fill_between([0, 1], [0, 1], color="#006ab3", alpha=0.2)
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=2)

# Add labels and title
plt.text(
    0.5,
    0.8,
    "AUC = 1.0",
    fontsize=8,
    ha="center",
    va="center",
    color="#003964",
    alpha=0.7,
)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(visible=False)
plt.gca().set_facecolor("lightblue")

# Limits and aspect ratio
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal", adjustable="box")

plt.savefig("perfect_roc_figure.pgf", bbox_inches="tight")
