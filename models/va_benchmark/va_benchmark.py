import csv
import numpy as np

from pygenn import (GeNNModel, VarAccess, VarLocation, 
                    create_neuron_model, create_weight_update_model,
                    init_var, init_postsynaptic, init_weight_update,
                    init_sparse_connectivity)

from time import perf_counter

def build_dense(num_pre: int, num_post: int, weight: float, prob):
    weights = np.zeros(num_pre * num_post, dtype=np.float32)
    
    weights[np.random.rand(num_pre * num_post) < prob] = weight
    
    return weights

def simulate_genn(num_neurons, num_timesteps=1000, dense=False, half=False,
                  probability_connection = 0.1, record=False):
    # Create models
    storage_type = "half" if half else "scalar"
    lif_model = create_neuron_model(
        "lif",
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
        vars=[("V", "scalar", storage_type),
              ("RefracTime", "scalar", storage_type)])

    static_pulse_model = create_weight_update_model(
        "static_pulse",
        vars=[("g", "scalar", storage_type, VarAccess.READ_ONLY)],
        pre_spike_syn_code="addToPost(g);")
    
    excitatory_inhibitory_ratio = 4
    num_excitatory = int(round((num_neurons * excitatory_inhibitory_ratio) / (1.0 + excitatory_inhibitory_ratio)))
    num_inhibitory = num_neurons - num_excitatory
    scale = (4000.0 / num_neurons) * (0.02 / probability_connection)
    inh_weight = -51.0E-3 * scale
    exc_weight = 4.0E-3 * scale

    print(f"\t{num_excitatory} excitatory neurons, {num_inhibitory} inhibitory neurons")

    print(f"\tWeight inhibitory: {inh_weight}, excitatory: {exc_weight}")
    print(f"\t{'Dense' if dense else 'sparse'} connectivity, {'Half' if half else 'Single'}-precision")
    
    model = GeNNModel("float", "va_benchmark")
    model.dt = 1.0
    model.timing_enabled = True
    model.default_narrow_sparse_ind_enabled = True
    fixed_prob = {"prob": probability_connection}
    
    lif_params = {"C": 10.0, "TauM": 20.0, "Vrest": 1.1, "Vreset": 0.0,
                  "Vthresh": 1.0, "Ioffset": 0.0, "TauRefrac": 5.0}

    lif_init = {"V": init_var("Uniform", {"min": 0.0, "max": 1.0}),
                "RefracTime": 0.0}

    excitatory_pop = model.add_neuron_population("E", num_excitatory, lif_model, lif_params, lif_init)
    inhibitory_pop = model.add_neuron_population("I", num_inhibitory, lif_model, lif_params, lif_init)

    excitatory_pop.spike_recording_enabled = record
    inhibitory_pop.spike_recording_enabled = record

    excitatory_postsynaptic_init = init_postsynaptic("ExpCurr", {"tau": 5.0})
    inhibitory_postsynaptic_init = init_postsynaptic("ExpCurr", {"tau": 10.0})

    if dense:
        model.add_synapse_population("EE", "DENSE",
            excitatory_pop, excitatory_pop,
            init_weight_update(static_pulse_model, {}, {"g": build_dense(num_excitatory,
                                                                         num_excitatory,
                                                                         exc_weight,
                                                                         probability_connection)}),
            excitatory_postsynaptic_init)

        model.add_synapse_population("EI", "DENSE",
            excitatory_pop, inhibitory_pop,
            init_weight_update(static_pulse_model, {}, {"g": build_dense(num_excitatory,
                                                                         num_inhibitory,
                                                                         exc_weight,
                                                                         probability_connection)}),
            excitatory_postsynaptic_init)

        model.add_synapse_population("II", "DENSE",
            inhibitory_pop, inhibitory_pop,
            init_weight_update(static_pulse_model, {}, {"g": build_dense(num_inhibitory,
                                                                         num_inhibitory,
                                                                         inh_weight,
                                                                         probability_connection)}),
            inhibitory_postsynaptic_init)

        model.add_synapse_population("IE", "DENSE",
            inhibitory_pop, excitatory_pop,
            init_weight_update(static_pulse_model, {}, {"g": build_dense(num_inhibitory,
                                                                         num_excitatory,
                                                                         inh_weight,
                                                                         probability_connection)}),
            inhibitory_postsynaptic_init)
    else:
        excitatory_weight_init = init_weight_update("StaticPulseConstantWeight", {"g": exc_weight})
        inhibitory_weight_init = init_weight_update("StaticPulseConstantWeight", {"g": inh_weight})
    
        model.add_synapse_population("EE", "SPARSE",
            excitatory_pop, excitatory_pop,
            excitatory_weight_init, excitatory_postsynaptic_init,
            init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

        model.add_synapse_population("EI", "SPARSE",
            excitatory_pop, inhibitory_pop,
            excitatory_weight_init, excitatory_postsynaptic_init,
            init_sparse_connectivity("FixedProbability", fixed_prob))

        model.add_synapse_population("II", "SPARSE",
            inhibitory_pop, inhibitory_pop,
            inhibitory_weight_init, inhibitory_postsynaptic_init,
            init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

        model.add_synapse_population("IE", "SPARSE",
            inhibitory_pop, excitatory_pop,
            inhibitory_weight_init, inhibitory_postsynaptic_init,
            init_sparse_connectivity("FixedProbability", fixed_prob))

    print("\tBuilding Model")
    model.build()
    print("\tLoading Model")
    model.load(num_recording_timesteps=num_timesteps)

    sim_start_time = perf_counter()
    while model.timestep < num_timesteps:
        model.step_time()
    sim_end_time = perf_counter()
    
    if record:
        # Download recording data
        model.pull_recording_buffers_from_device()

        # Get recording data
        e_spike_times, e_spike_ids = excitatory_pop.spike_recording_data[0]
        i_spike_times, i_spike_ids = inhibitory_pop.spike_recording_data[0]
        
        print(f"\tAverage excitatory spike rate: {len(e_spike_times) / num_excitatory} Hz")
        print(f"\tAverage inhibitory spike rate: {len(i_spike_times) / num_inhibitory} Hz")

        return (e_spike_times, e_spike_ids, i_spike_times, i_spike_ids, sim_end_time - sim_start_time,
                model.neuron_update_time, model.presynaptic_update_time)
    else:
        return (sim_end_time - sim_start_time, model.neuron_update_time, model.presynaptic_update_time)

with open(f"va_benchmark_perf.csv", "w") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    
    csv_writer.writerow(["Num neurons", "Dense connectivity", "Half precision", "Probability of connection", 
                         "Simulation time [s]", "Neuron update time [s]", "Presynaptic update time [s]"])
           
    # Build configurations
    max_sparse = 200000
    max_dense = 80000
    configs = [(n, True) for n in range(20000, max_dense + 1, 20000)]
    configs += [(n, False) for n in range(20000, max_sparse + 1, 20000)]
    
    # Loop through configurations
    for num_neurons, dense in configs:
        # Run simulation
        print(f"{num_neurons} neurons")
        
        # Loop through dense and sparse
        for i, half in enumerate([True, False]):
            data = simulate_genn(num_neurons=num_neurons, num_timesteps=1000,
                                 dense=dense, half=half, probability_connection=0.1,
                                 record=False)
            
            # Write CSV
            csv_writer.writerow([num_neurons, dense, half, 0.1] + list(data))
