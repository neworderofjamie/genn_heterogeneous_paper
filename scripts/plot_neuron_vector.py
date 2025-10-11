import matplotlib.pyplot as plt
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
sns.despine(ax=axis)

fig.legend(actors, labels, loc="lower center", 
           ncol=len(devices), frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.175, 1.0, 1.0])
plt.show()