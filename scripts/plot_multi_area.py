from glob import glob
from itertools import repeat
from matplotlib import gridspec as gs
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import json
import numpy as np
import seaborn as sns
from copy import copy
from os import path
from scipy.stats import entropy
from six import iteritems
from sys import argv
import plot_settings

def create_pop_data_array(populations, simulators, values):
    # Create suitable numpy array
    data = np.empty(len(populations), dtype=[("pop", "U10"), ("sim", "U10"), ("value", float)])
    
    # Populate it and return
    data["pop"][:] = populations
    data["sim"][:] = simulators
    data["value"][:] = np.concatenate(values)
    return data

def remove_junk(axis):
    sns.despine(ax=axis, left=True, bottom=True)
    axis.xaxis.grid(False)

def load_pop_data(stat_file_stem, simulator_prefix, data_path):
    # Create default dict for data
    populations = []
    values = []
    
    # Get list of files containing data for this
    data_files = list(glob(path.join(data_path, "%s_%s_*.npy" % (simulator_prefix, stat_file_stem))))
    for d in data_files:
        # Extract pop name
        pop_name = path.splitext(d)[0].split("_")[-1]
        
        # Load data
        data = np.load(d)
        
        # Add to arrays
        populations.extend(repeat(pop_name, len(data)))
        values.append(data)
    
    # Create and populate numpy array of data
    return create_pop_data_array(populations, simulator_prefix, values)

def plot_area(name, axis, archive):
    # Find files containing spikes for this area
    area_spikes = list(reversed(sorted(f for f in archive.files if f.startswith(f"{name}_"))))

    # Extract names of sub-populations from filenames
    pop_names = [path.basename(s).split("_")[1].split(".")[0] for s in area_spikes]
    assert all(a[-1] == "I" for a in pop_names[::2])
    assert all(a[-1] == "E" for a in pop_names[1::2])

    # Loop through area spike files and population names
    start_id = 0
    layer_counts = np.zeros(len(pop_names) // 2, dtype=int)
    excitatory_actor = None
    inhibitory_actor = None
    for i, (s, n)  in enumerate(zip(area_spikes, pop_names)):
        data = archive[s]
        
        # Approx
        num = int(np.amax(data[1]))

        # Add num to layer count
        layer_counts[i // 2] += num

        num_spikes = len(data[0])
        indices = np.random.choice(num_spikes, int(round(num_spikes * 0.03)))

        # Plot spikes
        is_inhibitory = n[-1] == "I"
        actor = axis.scatter(data[0][indices] / 1000.0, data[1][indices] + start_id, s=2,
                             rasterized=True, edgecolors="none", 
                             color="firebrick" if is_inhibitory else "navy")

        # Store actors
        if is_inhibitory:
            inhibitory_actor = actor
        else:
            excitatory_actor = actor

        # Update offset
        start_id += num

    # Label layers
    axis.set_yticks(np.cumsum(layer_counts) - (layer_counts / 2))
    axis.set_yticklabels(["L" + n[:-1] for n in pop_names[::2]])
    remove_junk(axis)
    axis.yaxis.grid(False)
    
    axis.set_xlim((3.0, 3.5))
    axis.set_ylim((0.0, np.sum(layer_counts)))
    axis.set_xlabel("Time [s]")
    
    return copy(excitatory_actor), copy(inhibitory_actor)

def plot_violin(stat_file_stem, genn_data_path, nest_data_path, axis, vertical, label, lim):
    # Combine GeNN and NEST rates
    data = np.hstack((load_pop_data(stat_file_stem, "nest", nest_data_path), 
                      load_pop_data(stat_file_stem, "average_pop", genn_data_path)))
    # Calculate order
    order = np.sort(np.unique(data["pop"]))

    # Plot split violin plot
    sns.violinplot(x=data["pop"] if vertical else data["value"], 
                   y=data["value"] if vertical else data["pop"], 
                   hue=data["sim"], split=True, inner="quartile", 
                   linewidth=0.75, cut=0.0, ax=axis, order=order, 
                   density_norm="width", legend=False)

    # Remove junk
    axis.minorticks_on()
    remove_junk(axis)
    #axis.yaxis.grid(True, "both")
    axis.yaxis.grid(False)

    # Configure axes
    if vertical:
        axis.set_ylabel(label)
        axis.set_ylim(lim)
        plt.setp(axis.get_xticklabels(), ha="center", rotation=90)
    else:
        axis.set_xlabel(label)
        axis.set_xlim(lim)

def plot_kl_divergence(data_path, stat, axis):
    populations = ["23E", "23I", "4E", "4I", "5E", "5I", "6E", "6I"]

    # Loop through permutations
    kl_div = []
    for i, perm in enumerate(["nest_seed_1", "nest_seed_2", "nest_seed_3",
                              "seed_1_seed_2", "seed_1_seed_3", "seed_2_seed_3"]):
        # Loop through populations
        kls = []
        for pop in populations:
            with open(path.join(data_path, f"{perm}_{stat}_{pop}.npy"), "rb") as f:
                bin_x = np.load(f)
                ground_truth_hist = np.load(f)
                comp_hist = np.load(f)

            # Normalize histograms
            bin_width = bin_x[1] - bin_x[0]
            ground_truth_hist = np.divide(ground_truth_hist, np.sum(ground_truth_hist) / bin_width, dtype="float")
            comp_hist = np.divide(comp_hist, np.sum(comp_hist) / bin_width, dtype="float")
            
            # Mask out bins with no data
            mask = (comp_hist > 1.0E-15)
            kls.append(entropy(ground_truth_hist[mask], comp_hist[mask]))

        kl_div.append(np.asarray(kls))
    
    kl_div = np.vstack(kl_div)
    kl_mean = [np.mean(kl_div[:3,:], axis=0), np.mean(kl_div[3:,:], axis=0)]
    kl_std = [np.std(kl_div[:3,:], axis=0), np.std(kl_div[3:,:], axis=0)]
    
    # Position bars
    kl_bar_width = 0.8
    kl_bar_pad = 0.2
    kl_bar_x = np.arange(0.0, len(populations) * (kl_bar_width + kl_bar_pad), kl_bar_width + kl_bar_pad)

    # Draw rate KL-divergence bars
    errorbar_kwargs = {"linestyle": "None", "marker": "o", "markersize": 1.0, "zorder": 10,
                       "capsize": 5.0, "elinewidth": 0.75, "capthick": 0.75, "clip_on": False}
    pal = sns.color_palette()
    permutation_actors = []
    for i, (m, s) in enumerate(zip(kl_mean, kl_std)):
        permutation_actors.append(axis.errorbar((kl_bar_x * 2.0) + (i * kl_bar_width), m, 
                                                yerr=s, color=pal[2 + i], **errorbar_kwargs)[2])

    axis.set_xticks((kl_bar_x * 2.0) + (0.5 * kl_bar_width))
    axis.set_xticklabels(populations)
    remove_junk(axis)
    axis.yaxis.grid(False)
    axis.set_ylabel("$D_{KL}$")
    return permutation_actors

# Create plot
fig = plt.figure(frameon=False, figsize=(plot_settings.double_column_width, 4.0))

# Create outer gridspec with three columns
gsp = gs.GridSpec(1, 3)

# Create two sub-gridspecs to divide these columns into gridspecs for raster and violin plots with an axis for each regime
raster_gsp = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gsp[0:2])
violin_gsp = gs.GridSpecFromSubplotSpec(7, 1, subplot_spec=gsp[2], hspace=0.5)

# Create axes within outer gridspec
v1_1_9_axis = plt.Subplot(fig, raster_gsp[0])
v2_1_9_axis = plt.Subplot(fig, raster_gsp[1])
fef_1_9_axis = plt.Subplot(fig, raster_gsp[2])

# Create axes within violin plot gridspec
rate_1_9_violin_axis = plt.Subplot(fig, violin_gsp[0])
corr_coeff_1_9_violin_axis = plt.Subplot(fig, violin_gsp[1])
irregularity_1_9_violin_axis = plt.Subplot(fig, violin_gsp[2])

rate_1_9_kl_axis = plt.Subplot(fig, violin_gsp[4])
corr_coeff_1_9_kl_axis = plt.Subplot(fig, violin_gsp[5])
irregularity_1_9_kl_axis = plt.Subplot(fig, violin_gsp[6])

# Add axes
fig.add_subplot(v1_1_9_axis)
fig.add_subplot(v2_1_9_axis)
fig.add_subplot(fef_1_9_axis)
fig.add_subplot(rate_1_9_violin_axis)
fig.add_subplot(corr_coeff_1_9_violin_axis)
fig.add_subplot(irregularity_1_9_violin_axis)
fig.add_subplot(rate_1_9_kl_axis)
fig.add_subplot(corr_coeff_1_9_kl_axis)
fig.add_subplot(irregularity_1_9_kl_axis)

# Plot example GeNN raster plots
recordings = np.load(path.join("chi_1_9", "genn_recordings.npz"))
excitatory_actor, inhibitory_actor = plot_area("V1", v1_1_9_axis, recordings)
plot_area("V2", v2_1_9_axis, recordings)
plot_area("FEF", fef_1_9_axis, recordings)

genn_data_path = path.join("chi_1_9", "genn_half_weight")
nest_data_path = "chi_1_9"

# Combine GeNN and NEST rates and plot split violin plot
plot_violin("rates", genn_data_path, nest_data_path, rate_1_9_violin_axis, 
            True, "Rate\n[spikes/s]", (-10.0, 150.0))
            
# Combine GeNN and NEST correlation coefficients and plot split violin plot
plot_violin("corr_coeff", genn_data_path, nest_data_path, corr_coeff_1_9_violin_axis, 
            True, "Correlation\ncoefficient", (-0.1, 0.6))

# Combine GeNN and NEST irregularity and plot split violin plot
plot_violin("irregularity", genn_data_path, nest_data_path, irregularity_1_9_violin_axis, 
            True, "Irregularity", (-0.5, 2.5))

# Plot KL divergences
kl_actors = plot_kl_divergence(nest_data_path, "rates", rate_1_9_kl_axis)
plot_kl_divergence(nest_data_path, "corr_coeff", corr_coeff_1_9_kl_axis)
plot_kl_divergence(nest_data_path, "irregularity", irregularity_1_9_kl_axis)

# Label axes
v1_1_9_axis.set_title("A: V1", loc="left")
v2_1_9_axis.set_title("B: V2", loc="left")
fef_1_9_axis.set_title("C: FEF", loc="left")

rate_1_9_violin_axis.set_title("D", loc="left")
corr_coeff_1_9_violin_axis.set_title("E", loc="left")
irregularity_1_9_violin_axis.set_title("F", loc="left")
rate_1_9_kl_axis.set_title("G", loc="left")
corr_coeff_1_9_kl_axis.set_title("H", loc="left")
irregularity_1_9_kl_axis.set_title("I", loc="left")

# Configure axis ticks
rate_1_9_violin_axis.yaxis.set_minor_locator(MultipleLocator(100.0))
corr_coeff_1_9_violin_axis.yaxis.set_minor_locator(MultipleLocator(0.25))
irregularity_1_9_violin_axis.yaxis.set_minor_locator(MultipleLocator(1.0))

plt.setp(rate_1_9_violin_axis.get_xticklabels(), visible=False)
plt.setp(corr_coeff_1_9_violin_axis.get_xticklabels(), visible=False)
plt.setp(rate_1_9_kl_axis.get_xticklabels(), visible=False)
plt.setp(corr_coeff_1_9_kl_axis.get_xticklabels(), visible=False)

# Show figure legend with devices beneath figure
pal = sns.color_palette()
fig.legend([Rectangle((0, 0), 1, 1, fc=pal[0]), Rectangle((0, 0), 1, 1, fc=pal[1])],
           ["NEST", "GeNN"], ncol=2, frameon=False, bbox_to_anchor=(0.85, 0.525), loc="center")
fig.legend(kl_actors, ["GeNN vs NEST", "GeNN vs GeNN"],
           ncol=2, frameon=False, bbox_to_anchor=(0.85, 0.0), loc="lower center")

# Increase size of markers in spike actors
excitatory_actor.set_sizes([10])
inhibitory_actor.set_sizes([10])

# Show second figure legend with inhibitory and excitatory spikes
fig.legend([excitatory_actor, inhibitory_actor],
           ["Excitatory", "Inhibitory"], ncol=2, frameon=False, bbox_to_anchor=(0.333, 0.0), loc="lower center")


fig.align_ylabels([rate_1_9_violin_axis, corr_coeff_1_9_violin_axis, irregularity_1_9_violin_axis])
fig.tight_layout(pad=0, w_pad=2.0, rect= [0.0, 0.05, 1.0, 1.0])

if not plot_settings.presentation:
    fig.savefig("../figures/multi_area.pdf", dpi=600)
    
# Show plot
plt.show()
