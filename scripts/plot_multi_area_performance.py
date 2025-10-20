import matplotlib.pyplot as plt
import numpy as np
from os import path
import plot_settings
import seaborn as sns

from glob import glob
from json import load

def plot_stacked_bar(axis, sort_index, ref_labels,  
                     ref_heights, labels, heights):
    # Numpify heights
    order = np.argsort(heights[sort_index])
    heights = [np.asarray(h)[order] for h in heights]
    
    # Re-order labels
    labels = [labels[o] for o in order]

    num_bars = len(prepare_times_s)
    bar_x = np.arange(num_bars)
    bottom = np.zeros(num_bars)
    
    actors = []
    for h in heights:
        actors.append(axis.bar(bar_x, h, bottom=bottom))
        bottom += h
    
    ref_bar_x = np.arange(num_bars, num_bars + len(ref_heights))
    actors.append(axis.bar(ref_bar_x, ref_heights))
    
    axis.set_xticks(np.concatenate((bar_x, ref_bar_x)))
    axis.set_xticklabels(labels + ref_labels)
    return actors

# Loop through parameter files
baseline_time = 100500.0
labels = []
prepare_times_s = []
init_times_s = []
neuron_update_times_s = []
presynaptic_update_times_s = []
overhead_times_s = []

for f in glob(path.join("multiarea_logs", "custom_params*")):
    # Extract hash
    hash = path.split(f)[1].split("_")[-1]
    print(hash)

    # Read params JSON
    with open(f, "r") as f:
        params = load(f)

    sim_time_ms = params["sim_params"]["t_sim"]
    half_weights = params["sim_params"].get("half_precision_weights", False)
    half_neurons = params["sim_params"].get("half_precision_neurons", False)
    procedural_connectivity = params["sim_params"].get("procedural_connectivity", True)
    normalize_v = params["network_params"]["neuron_params"].get("normalize_voltage", False)

    # Build labels
    if procedural_connectivity:
        labels.append("Procedural")
    elif half_weights and not half_neurons and not normalize_v:
        labels.append("Half-precision\nweights")
    elif half_weights and half_neurons and normalize_v:
        labels.append("Half-precision")
    else:
        assert False

    # Read log JSON
    with open(path.join("multiarea_logs", f"{hash}_logfile"), "r") as f:
        log = load(f)
    
    # Figure out how much to scale sim times by
    sim_scale = sim_time_ms / baseline_time
    
    prepare_times_s.append(log["time_prepare"] 
                           + log["time_network_local"]
                           + log["time_network_global"]
                           + log["time_genn_build"]
                           + log["time_genn_load"])
    init_times_s.append((log["time_genn_init"]
                         + log["time_genn_init_sparse"]) / 1000.0)
    neuron_update_times_s.append(log["time_genn_neuron_update"]
                                 / (1000.0 * sim_scale))
    presynaptic_update_times_s.append(log["time_genn_presynaptic_update"]
                                      / (1000.0 * sim_scale))
    
    sim_s = log["time_simulate"] / sim_scale
    overhead_times_s.append(sim_s - neuron_update_times_s[-1]
                            - presynaptic_update_times_s[-1])

# Numpify

fig, axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))

# Plot stacked bar
nest_gpu = 15.3 * (baseline_time / 1000.0)
nest_cpu = 47.9 * (baseline_time / 1000.0)
actors = plot_stacked_bar(axis, 3, ["NEST GPU\nCluster", "NEST\nCluster"], [nest_gpu, nest_cpu],
                          labels,
                          [prepare_times_s, init_times_s, neuron_update_times_s,
                           presynaptic_update_times_s, overhead_times_s])



axis.set_ylabel("Time [s]")
sns.despine(ax=axis, left=True, bottom=True)
axis.xaxis.grid(False)

fig.legend(actors, ["Preparation", "Initialisation", "Neuron update",
                    "Presynaptic update", "Overhead", "NEST simulation"],
           loc="lower center", ncol=3, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.175, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/multiarea_perf.pdf", dpi=600)
    
plt.show()
