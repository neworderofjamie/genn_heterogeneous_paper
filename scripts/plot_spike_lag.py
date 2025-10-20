import numpy as np
import os
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns

def load_spikes(data_type: str, time: float):
    return (np.load(os.path.join("compare_neurons", f"spike_id_{data_type}_0.1_{time}.npy")),
            np.load(os.path.join("compare_neurons", f"spike_time_{data_type}_0.1_{time}.npy")))

def plot_lag(axis, time: float, num_neurons: int = 10):
    # Load data
    spike_id_float, spike_time_float = load_spikes("float", time)
    spike_id_half, spike_time_half = load_spikes("half", time)
    spike_id_half_rescale, spike_time_half_rescale = load_spikes("half_rescale", time)

    # Loop through neurons
    half_lag = []
    half_rescale_lag = []
    for i in range(num_neurons):
        # Extract spike ids for this neuron
        neuron_spike_time_float = spike_time_float[spike_id_float == i]
        neuron_spike_time_half = spike_time_half[spike_id_half == i]
        neuron_spike_time_half_rescale = spike_time_half_rescale[spike_id_half_rescale == i]

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
    axis.boxplot(lag, showfliers=False)
    axis.set_ylabel("Lag [ms]")
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
    half_rescale_actor = axis.plot(t, (v_half_rescale[:,neuron] * 15.0) - 65.0, 
                                   alpha=0.5)[0]
    axis.set_ylabel("Membrane voltage [mV]")

    axis.xaxis.grid(False)
    sns.despine(ax=axis)
    return float_actor, half_actor, half_rescale_actor

def plot_rmse(axis, time: float, num_neurons: int = 10):
    v_float = np.load(os.path.join("compare_neurons", f"v_float_0.1_{time}.npy"))
    v_half = np.load(os.path.join("compare_neurons", f"v_half_0.1_{time}.npy"))
    v_half_rescale = np.load(os.path.join("compare_neurons", f"v_half_rescale_0.1_{time}.npy"))

    assert v_float.shape == v_half.shape
    assert v_float.shape == v_half_rescale.shape
    
    # Loop through neurons
    half_rmse = []
    half_rescale_rmse = []
    for i in range(num_neurons):
        v_half_rescale_scale = (v_half_rescale[:,i] * 15.0) - 65.0

        half_rmse.append(np.sqrt(np.sum(np.square(v_half[:,i] - v_float[:,i])) / v_float.shape[0]))
        half_rescale_rmse.append(np.sqrt(np.sum(np.square(v_half_rescale_scale - v_float[:,i])) / v_float.shape[0]))

    rmse = np.column_stack((half_rmse, half_rescale_rmse))
    axis.boxplot(rmse, showfliers=False)
    axis.set_ylabel("RMSE [mV]")
    axis.xaxis.grid(False)
    sns.despine(ax=axis)

fig = plt.figure(frameon=False, figsize=(plot_settings.column_width, 3.0))

# Create outer gridspec with three columns
gsp = gs.GridSpec(1, 2)

# Create two sub-gridspecs to divide these columns into gridspecs for voltage traces and box plots
voltage_gsp = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[0], hspace=0.3)
boxplot_gsp = gs.GridSpecFromSubplotSpec(4, 1, subplot_spec=gsp[1], hspace=0.3)

# Create axes within voltage gridspec
low_rate_v_axis = plt.Subplot(fig, voltage_gsp[0])
high_rate_v_axis = plt.Subplot(fig, voltage_gsp[1])

# Create axes within violin plot gridspec
low_acc_lag_axis = plt.Subplot(fig, boxplot_gsp[0])
low_rmse_lag_axis = plt.Subplot(fig, boxplot_gsp[1])
high_acc_lag_axis = plt.Subplot(fig, boxplot_gsp[2])
high_rmse_lag_axis = plt.Subplot(fig, boxplot_gsp[3])

# Add axes
fig.add_subplot(low_rate_v_axis)
fig.add_subplot(high_rate_v_axis)
fig.add_subplot(low_acc_lag_axis)
fig.add_subplot(low_rmse_lag_axis)
fig.add_subplot(high_acc_lag_axis)
fig.add_subplot(high_rmse_lag_axis)

float_actor, half_actor, half_rescale_actor = plot_voltage(low_rate_v_axis, 16000.0)
plot_voltage(high_rate_v_axis, 4000.0)
low_rate_v_axis.set_xlim((4800.0, 5000.0))
high_rate_v_axis.set_xlim((1800.0, 2000.0))

plot_lag(low_acc_lag_axis, 16000.0)
plot_lag(high_acc_lag_axis, 4000.0)
plot_rmse(low_rmse_lag_axis, 16000.0)
plot_rmse(high_rmse_lag_axis, 4000.0)

low_rate_v_axis.set_title("A", loc="left")
high_rate_v_axis.set_title("B", loc="left")
low_acc_lag_axis.set_title("C", loc="left")
low_rmse_lag_axis.set_title("D", loc="left")
high_acc_lag_axis.set_title("E", loc="left")
high_rmse_lag_axis.set_title("F", loc="left")

low_acc_lag_axis.set_xticks([])
low_rmse_lag_axis.set_xticks([])
high_acc_lag_axis.set_xticks([])
high_rmse_lag_axis.set_xticklabels(["Half-precision", "Half-precision\nrescaled"])

fig.align_ylabels([low_acc_lag_axis, low_rmse_lag_axis, high_acc_lag_axis, high_rmse_lag_axis])
fig.legend([float_actor, half_actor, half_rescale_actor], ["Float", "Half-precision", "Rescaled half-precision"], 
           loc="lower center", ncol=3, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.1, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/single_neuron.pdf", dpi=600)

plt.show()