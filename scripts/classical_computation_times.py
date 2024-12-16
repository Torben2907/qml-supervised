import matplotlib.pyplot as plt
import pandas as pd


data = {
    "Datasets": [
        "CCRF",
        "CTG",
        "FERTILITY",
        "HABERMAN",
        "HCV",
        "NAFLD",
        "HEROIN",
        "CCBR",
        "WDBC",
    ],
    "rbf": [
        0.010203838348388672,
        0.0168612003326416 + 0.03145432472229004,
        0.00602412223815918,
        0.011577129364013672,
        0.0059740543365478516 + 0.011947154998779297,
        0.0062901973724365234,
        0.013252735137939453,
        0.006140947341918945 + 0.011764049530029297,
        0.019979000091552734 + 0.040840864181518555 + 0.05727696418762207,
    ],
    "sigmoid": [
        0.005269050598144531,
        0.027242183685302734 + 0.04432225227355957,
        0.004926919937133789,
        0.01019430160522461,
        0.005531787872314453 + 0.011021852493286133,
        0.0052797794342041016,
        0.01196908950805664,
        0.005420207977294922 + 0.01072382926940918,
        0.038866281509399414 + 0.08527421951293945 + 0.11661314964294434,
    ],
    "poly": [
        0.0060291290283203125,
        0.012665033340454102 + 0.024574995040893555,
        0.005429983139038086,
        0.010603189468383789,
        0.005772590637207031 + 0.011439800262451172,
        0.005732059478759766,
        0.011059045791625977,
        0.005357027053833008 + 0.01071310043334961,
        0.015084981918334961 + 0.03433799743652344 + 0.047470808029174805,
    ],
}

df = pd.DataFrame(data)

# Plot
fig, ax = plt.subplots(figsize=(6, 10))

# rbf kernel
ax.plot(
    df["Datasets"],
    df["rbf"],
    marker="H",
    color="#b5cbd6",  # hhuiceblue
    label="rbf",
    linestyle="-",
    alpha=0.8,
)

# sigmoid kernel
ax.plot(
    df["Datasets"],
    df["sigmoid"],
    marker="h",
    color="#8cb110",  # hhugreen
    label="sigmoid",
    linestyle="-",
    alpha=0.8,
)

# poly kernel
ax.plot(
    df["Datasets"],
    df["poly"],
    marker="^",
    color="#57bab1",  # hhucyan
    label="poly",
    linestyle="-",
    alpha=0.8,
)

ax.set_xlabel("Datasets", fontsize=10)
ax.set_ylabel("Time in seconds (s)", fontsize=10)
ax.legend(title="Embedding", fontsize=8)
ax.grid(True)
fig.set_size_inches(w=4.5, h=3.5)  # for LaTeX rendering
plt.xticks(rotation=45, ha="right")
plt.yscale("log")  # If log scale is needed like in the reference image
plt.tight_layout()

# Show plot
plt.savefig("cpu-computation-times.pdf", bbox_inches="tight", transparent=True)
print("Figure saved successfully.")
