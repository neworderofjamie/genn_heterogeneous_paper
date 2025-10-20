import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plot_settings
import seaborn as sns

from pandas import read_csv

df = read_csv("hip_neuron_vector.csv", delimiter=",")

devices = df["Device"].unique()

fig, axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))

# Loop through devices
actors = []
labels = []
for d in devices:
    device_df = df[df["Device"] == d]
    float_df = device_df[device_df["Data type"] == "float"]
    half_df = device_df[device_df["Data type"] == "half"]

    actor = axis.plot(float_df["Num neurons"], float_df["Total time"])[0]
    axis.plot(half_df["Num neurons"], half_df["Total time"],
              color=actor.get_color(), linestyle="--")
    actors.append(actor)
    labels.append(d)

axis.set_xlabel("Number of neurons")
axis.set_ylabel("Simulation time [s]")
axis.xaxis.grid(False)
axis.ticklabel_format(useOffset=False, style="plain") 
sns.despine(ax=axis)

assert len(actors) == 2
fig.legend([actors[0], mlines.Line2D([],[], color="black"), actors[1], mlines.Line2D([],[], linestyle="--", color="black")],
           [labels[0], "Single-precision", labels[1], "Half-precision vectorized"], 
           loc="lower center", ncol=2, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/vector_perf.pdf", dpi=600)

plt.show()