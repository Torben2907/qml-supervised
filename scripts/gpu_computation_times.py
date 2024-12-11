import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Dataset": [
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
    "Angle": [
        37.799760580062866,
        360.0253961086273,
        15.048522472381592,
        10.944638967514038,
        70.33695721626282,
        19.034689903259277,
        65.25011873245239,
        43.516340255737305,
        109.43064141273499,
    ],
    "IQP": [
        94.83670711517334,
        2182.2871265411377,
        106.37863564491272,
        18.779486656188965,
        528.1848683357239,
        127.14523816108704,
        427.3411147594452,
        376.60979294776917,
        816.8104865550995,
    ],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Angle
ax.plot(
    df["Dataset"],
    df["Angle"],
    marker="H",
    color="#be0a26",  # hhurejd
    label="Angle",
    linestyle="-",
    alpha=0.8,
)

# Plot IQP
ax.plot(
    df["Dataset"],
    df["IQP"],
    marker="h",
    color="#006ab3",  # hhublue
    label="IQP",
    linestyle="-",
    alpha=0.7,
)

# Formatting
ax.set_xlabel("Dataset", fontsize=10)
ax.set_ylabel("Time in seconds (s)", fontsize=10)
ax.legend(title="Embedding", fontsize=8, loc="lower right")
ax.grid(True)
fig.set_size_inches(w=4.5, h=3.5)  # for LaTeX rendering
plt.xticks(rotation=45, ha="right")
plt.yscale("log")  # If log scale is needed like in the reference image
plt.tight_layout()

# Show plot
plt.savefig("gpu-computation-times.pdf", bbox_inches="tight", transparent=True)
print("Figure saved successfully.")
