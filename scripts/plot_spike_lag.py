import numpy as np
import os
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns

def load_spikes(data_type: str, time: float):
    return (np.load(os.path.join("compare_neurons", f"spike_id_{data_type}_0.1_{time}.npy")),
            np.load(os.path.join("compare_neurons", f"spike_time_{data_type}_0.1_{time}.npy")))

def plot_lag(axis, time: float, half_colour, half_rescale_colour, num_neurons: int = 10):
    # Load data
    spike_id_float, spike_time_float = load_spikes("float", time)
    spike_id_half, spike_time_half = load_spikes("half", time)
    spike_id_half_rescale, spike_time_half_rescale = load_spikes("half_rescale", time)

    # Loop through neurons
    half_lag = []
    half_rescale_lag = []
    for i in range(num_neurons):
        # Extract spike ids for this neuron
        neuron_spike_time_float = spike_time_float[spike_id_float == i] / 1000.0
        neuron_spike_time_half = spike_time_half[spike_id_half == i] / 1000.0
        neuron_spike_time_half_rescale = spike_time_half_rescale[spike_id_half_rescale == i] / 1000.0

        # Make length the same
        num_half = min(len(neuron_spike_time_float),
                       len(neuron_spike_time_half))
        num_half_rescale = min(len(neuron_spike_time_float),
                               len(neuron_spike_time_half_rescale))

        # Calculate sum of absolute lags
        half_lag.append(np.sum(np.abs(neuron_spike_time_half[:num_half]
                                      - neuron_spike_time_float[:num_half])))
        half_rescale_lag.append(np.sum(np.abs(neuron_spike_time_half_rescale[:num_half_rescale] 
                                              - neuron_spike_time_float[:num_half_rescale])))
    
    lag = np.column_stack((half_lag, half_rescale_lag))
    bplot = axis.boxplot(lag, showfliers=False, patch_artist=True)
    bplot["boxes"][0].set_facecolor(half_colour)
    bplot["boxes"][1].set_facecolor(half_rescale_colour)
    bplot["medians"][0].set_color("black")
    bplot["medians"][1].set_color("black")
    axis.xaxis.grid(False)
    sns.despine(ax=axis)

def plot_voltage(axis, time: float, neuron: int = 0):
    v_float = np.load(os.path.join("compare_neurons", f"v_float_0.1_{time}.npy"))
    v_half = np.load(os.path.join("compare_neurons", f"v_half_0.1_{time}.npy"))
    v_half_rescale = np.load(os.path.join("compare_neurons", f"v_half_rescale_0.1_{time}.npy"))

    assert v_float.shape == v_half.shape
    assert v_float.shape == v_half_rescale.shape

    t = np.arange(0, v_float.shape[0] * 0.1, 0.1)
    float_actor = axis.plot(t, v_float[:,neuron], linestyle="--", color="gray")[0]
    half_actor = axis.plot(t, v_half[:,neuron], alpha=0.5)[0]
    half_rescale_actor = axis.plot(t, (np.float64(v_half_rescale[:,neuron]) * 15.0) - 65.0, 
                                   alpha=0.5)[0]
    axis.xaxis.grid(False)
    sns.despine(ax=axis)
    return float_actor, half_actor, half_rescale_actor

def plot_rmse(axis, time: float, half_colour, half_rescale_colour, num_neurons: int = 10):
    v_float = np.load(os.path.join("compare_neurons", f"v_float_0.1_{time}.npy"))
    v_half = np.load(os.path.join("compare_neurons", f"v_half_0.1_{time}.npy"))
    v_half_rescale = np.load(os.path.join("compare_neurons", f"v_half_rescale_0.1_{time}.npy"))

    assert v_float.shape == v_half.shape
    assert v_float.shape == v_half_rescale.shape
    
    # Loop through neurons
    half_rmse = []
    half_rescale_rmse = []
    for i in range(num_neurons):
        v_half_rescale_scale = (np.float64(v_half_rescale[:,i]) * 15.0) - 65.0
        half_rmse.append(np.sqrt(np.sum(np.square(v_half[:,i] - v_float[:,i])) / v_float.shape[0]))
        half_rescale_rmse.append(np.sqrt(np.sum(np.square(v_half_rescale_scale - v_float[:,i])) / v_float.shape[0]))

    rmse = np.column_stack((half_rmse, half_rescale_rmse))
    bplot = axis.boxplot(rmse, showfliers=False, patch_artist=True)
    bplot["boxes"][0].set_facecolor(half_colour)
    bplot["boxes"][1].set_facecolor(half_rescale_colour)
    bplot["medians"][0].set_color("black")
    bplot["medians"][1].set_color("black")
    axis.xaxis.grid(False)
    sns.despine(ax=axis)

def get_ulps(start, stop):
    c = [np.float16(start)]
    while c[-1] < stop:
        c.append(np.nextafter(c[-1], stop))
    return c

def plot_divergence(spike_axis, initial_axis, time: float, neuron: int, 
                    half_colour, half_rescale_colour):
    v_float = np.load(os.path.join("compare_neurons", f"v_float_0.1_{time}.npy"))
    v_half = np.load(os.path.join("compare_neurons", f"v_half_0.1_{time}.npy"))
    v_half_rescale = np.load(os.path.join("compare_neurons", f"v_half_rescale_0.1_{time}.npy"))

    assert v_float.shape == v_half.shape
    assert v_float.shape == v_half_rescale.shape
    
    t = np.arange(0, v_float.shape[0] * 0.1, 0.1)
    for i, axis in enumerate([spike_axis, initial_axis]):
        axis.plot(t, v_float[:,neuron], linestyle="--", color="gray")
        axis.plot(t, v_half[:,neuron], alpha=0.5, color=half_colour, 
                  marker="None" if i == 0 else ".")
        axis.plot(t, (np.float64(v_half_rescale[:,neuron]) * 15.0) - 65.0, 
                  alpha=0.5, color=half_rescale_colour)
        axis.xaxis.grid(False)
        axis.yaxis.grid(False)
        sns.despine(ax=axis)
    
    # Plot fp16 values in interval
    for c in get_ulps(-65.0, -64.5):
        initial_axis.axhline(c, linestyle=":", alpha=0.2, color="gray")
   
    # Zoom in on different bits
    spike_axis.set_xlim((13.25, 13.85))
    spike_axis.set_ylim((-50.25, -50.0))
    
    initial_axis.set_xlim((0.0, 0.4))
    initial_axis.set_ylim((-65.0, -64.6))
    
    
fig = plt.figure(frameon=False, figsize=(plot_settings.double_column_width, 2.0))

# Create outer gridspec with four columns
gsp = gs.GridSpec(1, 4, width_ratios=[0.25, 0.25, 0.5 / 3, 1.0 / 3])


# Create two sub-gridspecs to divide these columns into gridspecs for voltage traces and box plots
boxplot_gsp = gs.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsp[3], hspace=0.4)
inset_gsp = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[2], hspace=0.4)

# Create axes within voltage gridspec
low_rate_v_axis = plt.Subplot(fig, gsp[0])
high_rate_v_axis = plt.Subplot(fig, gsp[1])

# Create axes for insets
start_v_axis = plt.Subplot(fig, inset_gsp[0,0])
spike_v_axis = plt.Subplot(fig, inset_gsp[1,0])

# Create axes within violin plot gridspec
low_acc_lag_axis = plt.Subplot(fig, boxplot_gsp[0,0])
low_rmse_lag_axis = plt.Subplot(fig, boxplot_gsp[1,0])
high_acc_lag_axis = plt.Subplot(fig, boxplot_gsp[0,1])
high_rmse_lag_axis = plt.Subplot(fig, boxplot_gsp[1,1])

# Add axes
fig.add_subplot(low_rate_v_axis)
fig.add_subplot(high_rate_v_axis, sharey=low_rate_v_axis)
fig.add_subplot(spike_v_axis)
fig.add_subplot(start_v_axis)

fig.add_subplot(low_acc_lag_axis)
fig.add_subplot(low_rmse_lag_axis, sharex=low_acc_lag_axis)
fig.add_subplot(high_acc_lag_axis, sharey=low_acc_lag_axis)
fig.add_subplot(high_rmse_lag_axis, sharex=high_acc_lag_axis, sharey=low_acc_lag_axis)

float_actor, half_actor, half_rescale_actor = plot_voltage(low_rate_v_axis, 16000.0)
plot_voltage(high_rate_v_axis, 4000.0)
low_rate_v_axis.set_xlim((4800.0, 5000.0))
high_rate_v_axis.set_xlim((1800.0, 2000.0))

plot_divergence(spike_v_axis, start_v_axis, 4000.0, 0, 
                half_actor.get_color(), half_rescale_actor.get_color())

plot_lag(low_acc_lag_axis, 16000.0, half_actor.get_color(), half_rescale_actor.get_color())
plot_lag(high_acc_lag_axis, 4000.0, half_actor.get_color(), half_rescale_actor.get_color())
plot_rmse(low_rmse_lag_axis, 16000.0, half_actor.get_color(), half_rescale_actor.get_color())
plot_rmse(high_rmse_lag_axis, 4000.0, half_actor.get_color(), half_rescale_actor.get_color())

low_rate_v_axis.set_title("A", x=-0.1333)
high_rate_v_axis.set_title("B", x=-0.1333)
start_v_axis.set_title("C", x=-0.2)
spike_v_axis.set_title("D", x=-0.2)
low_acc_lag_axis.set_title("E", x=-0.2)
low_rmse_lag_axis.set_title("G", x=-0.2)
high_acc_lag_axis.set_title("F", x=-0.2)
high_rmse_lag_axis.set_title("H", x=-0.2)

# Label axes
low_acc_lag_axis.set_ylabel("Lag [s]")
low_rmse_lag_axis.set_ylabel("RMSE [mV]")
low_rate_v_axis.set_ylabel("Membrane voltage [mV]")

# Hide ticks
plt.setp(low_acc_lag_axis.get_xticklabels(), visible=False)
plt.setp(high_acc_lag_axis.get_xticklabels(), visible=False)
plt.setp(low_rmse_lag_axis.get_xticklabels(), visible=False)
plt.setp(high_rmse_lag_axis.get_xticklabels(), visible=False)

plt.setp(high_rate_v_axis.get_yticklabels(), visible=False)
plt.setp(high_acc_lag_axis.get_yticklabels(), visible=False)
plt.setp(high_rmse_lag_axis.get_yticklabels(), visible=False)

fig.align_ylabels([low_acc_lag_axis, low_rmse_lag_axis])
fig.legend([float_actor, half_actor, half_rescale_actor], ["Single-precision", "Half-precision", "Rescaled half-precision"], 
           loc="lower center", ncol=3, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/single_neuron.pdf", dpi=600)

plt.show()
