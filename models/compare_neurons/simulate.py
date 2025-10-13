import numpy as np 
import os

from pandas import DataFrame
from pygenn import GeNNModel

from glob import glob
from itertools import product
from pygenn import (create_current_source_model, create_neuron_model,
                    init_var)

current_source = create_current_source_model(
    "current_source",
    injection_code="""
    current += Init * (scalar)numSpikes[(id * numTimesteps) + (int)round(t / dt)];
    injectCurrent(current);
    current *= ExpDecay;
    """,
    params=["weight", "tauSyn", ("numTimesteps", "int")],
    derived_params=[("ExpDecay", lambda pars, dt: np.exp(-dt / pars["tauSyn"])),
                    ("Init", lambda pars, dt: pars["weight"] * (1.0 - np.exp(-dt / pars["tauSyn"])) * (pars["tauSyn"] / dt))],
    vars=[("current", "scalar")],
    extra_global_params=[("numSpikes", "int*")])

lif_half = create_neuron_model(
    "lif_half",
    sim_code="""
        if (RefracTime <= 0.0) {
          scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;
          V = alpha - (ExpTC * (alpha - V));
        }
        else {
          RefracTime -= dt;
        }
        """,
    threshold_condition_code="RefracTime <= 0.0 && V >= Vthresh",
    reset_code="""
        V = Vreset;
        RefracTime = TauRefrac;
        """,
    params=["C", "TauM", "Vrest", "Vreset","Vthresh","Ioffset","TauRefrac"],

    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["TauM"])),
                    ("Rmembrane", lambda pars, dt: pars["TauM"] / pars["C"])],
    vars=[("V", "scalar", "half"), ("RefracTime", "scalar", "half")])

# Find data for this timestep
data = glob(f"poisson_*_*_*_*.npy")
assert len(data) > 0

# Split up filenames
splits = list(zip(*(os.path.splitext(d)[0].split("_")[1:] for d in data)))

# Stick in dataframe
df = DataFrame({"rate": [float(s) for s in splits[0]],
                "time": [float(s) for s in splits[1]],
                "dt": [float(s) for s in splits[2]],
                "repeat": [int(s) for s in splits[3]],
                "filename": data})

lif_init = {"V": -58.0, "RefracTime": 0.0}
lif_params = {"C": 0.25, "TauM": 10.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh" : -50.0,
              "Ioffset": 0.0, "TauRefrac": 2.0}
cs_init = {"current": 0.0}

# Loop through dt and time i.e. things that require seperate simulations
for (dt, time), df_group in df.groupby(["dt", "time"]):
    print(dt, time)
    model = GeNNModel("float", "compare_neurons")
    model.dt = dt
    num_timesteps = int(round(time / dt))

    num_neurons = len(df_group)

    # Create neuron populations
    float_neuron_pop = model.add_neuron_population("FloatNeuron", num_neurons, "LIF", lif_params, lif_init)
    half_neuron_pop = model.add_neuron_population("HalfNeuron", num_neurons, lif_half, lif_params, lif_init)
    float_neuron_pop.spike_recording_enabled = True
    half_neuron_pop.spike_recording_enabled = True

    # Add current sources to deliver poisson input
    cs_params = {"weight": 87.8 / 1000.0, "tauSyn": 0.5, "numTimesteps": num_timesteps}
    float_cs = model.add_current_source("FloatCS", current_source, float_neuron_pop, cs_params, cs_init)
    half_cs = model.add_current_source("HalfCS", current_source, half_neuron_pop, cs_params, cs_init)

    # Load poisson data and stack together
    poisson_data = np.vstack([np.load(f) for f in df_group["filename"]])
    assert(poisson_data.shape == (num_neurons, num_timesteps))
    half_cs.extra_global_params["numSpikes"].set_init_values(poisson_data.flatten())
    float_cs.extra_global_params["numSpikes"].set_init_values(poisson_data.flatten())

    model.build()
    model.load(num_recording_timesteps=num_timesteps)

    float_v = []
    half_v = []
    for t in range(num_timesteps):
        model.step_time()
        float_neuron_pop.vars["V"].pull_from_device()
        half_neuron_pop.vars["V"].pull_from_device()
        float_v.append(float_neuron_pop.vars["V"].values)
        half_v.append(half_neuron_pop.vars["V"].values)

    # Stack voltages and save
    float_v = np.vstack(float_v)
    half_v = np.vstack(half_v)
    np.save(f"v_float_{dt}_{time}.npy", float_v)
    np.save(f"v_half_{dt}_{time}.npy", half_v)

    # Read spikes and save
    model.pull_recording_buffers_from_device()
    float_spike_times, float_spike_ids = float_neuron_pop.spike_recording_data[0]
    half_spike_times, half_spike_ids = half_neuron_pop.spike_recording_data[0]
    np.save(f"spike_time_float_{dt}_{time}.npy", float_spike_times)
    np.save(f"spike_id_float_{dt}_{time}.npy", float_spike_ids)
    np.save(f"spike_time_half_{dt}_{time}.npy", half_spike_times)
    np.save(f"spike_id_half_{dt}_{time}.npy", half_spike_ids)