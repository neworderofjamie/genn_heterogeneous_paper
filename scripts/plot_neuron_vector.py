import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plot_settings
import seaborn as sns

from pandas import read_csv

neuron_df = read_csv("hip_neuron_vector.csv", delimiter=",")
spike_prop_df = read_csv("hip_dense_vector.csv", delimiter=",")

devices = neuron_df["Device"].unique()
#assert devices == spike_prop_df["Device"].unique()

if plot_settings.presentation:
    neuron_fig, neuron_axis = plt.subplots()
    spike_prop_fig, spike_prop_axis = plt.subplots()
    axes = [neuron_axis, spike_prop_axis]
    figs = [neuron_fig, spike_prop_fig]
else:
    fig, axes = plt.subplots(1, 2, figsize=(plot_settings.double_column_width, 2.0))
    figs = [fig]
    
# Loop through devices
actors = []
labels = []
for d in devices:
    print(d)
    
    # Get neuron data
    neuron_device_df = neuron_df[neuron_df["Device"] == d]
    neuron_float_df = neuron_device_df[neuron_device_df["Data type"] == "float"]
    neuron_half_df = neuron_device_df[neuron_device_df["Data type"] == "half"]
    print(f"\tMax neuron speedup: {np.amax(neuron_float_df['Neuron time'].to_numpy() / neuron_half_df['Neuron time'].to_numpy())}")
    
    # Plot lines
    actor = axes[0].plot(neuron_float_df["Num neurons"], neuron_float_df["Neuron time"], marker=".")[0]
    axes[0].plot(neuron_half_df["Num neurons"], neuron_half_df["Neuron time"],
                 marker=".", color=actor.get_color(), linestyle="--")
    
    # Get synapse data
    spike_prop_device_df = spike_prop_df[spike_prop_df["Device"] == d]
    spike_prop_float_df = spike_prop_device_df[spike_prop_device_df["Data type"] == "float"]
    spike_prop_half_df = spike_prop_device_df[spike_prop_device_df["Data type"] == "half"]
    print(f"\tMax spike propagation speedup: {np.amax(spike_prop_float_df['Presynaptic time'].to_numpy() / spike_prop_half_df['Presynaptic time'].to_numpy())}")
    # Plot lines
    axes[1].plot(spike_prop_float_df["Num neurons"], spike_prop_float_df["Presynaptic time"],
                 marker=".", color=actor.get_color())
    axes[1].plot(spike_prop_half_df["Num neurons"], spike_prop_half_df["Presynaptic time"],
                 marker=".", color=actor.get_color(), linestyle="--")
    
    
    
    actors.append(actor)
    labels.append(d)

axes[0].set_title("A", x=-0.06666)
axes[1].set_title("B", x=-0.06666)
axes[1].set_xticks(np.linspace(0, 2000000, 6))
for a in axes:
    a.set_xlabel("Number of neurons")
    a.set_ylabel("Kernel time [s]")
    a.xaxis.grid(False)
    a.ticklabel_format(useOffset=False, style="plain") 
    sns.despine(ax=a)

assert len(actors) == 2
for f in figs:
    f.legend([actors[0], actors[1], mlines.Line2D([],[], color="black"), mlines.Line2D([],[], linestyle="--", color="black")],
             [labels[0], labels[1], "Standard kernel", "Vectorised kernel"], 
             loc="lower center", ncol=4, frameon=False)

    f.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/vector_perf.pdf", dpi=600)

plt.show()
