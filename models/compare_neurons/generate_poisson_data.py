import numpy as np

# Rate [spikes/s],  Time [ms], DT [ms]
simulations = [(8000.0, 16000.0, 1.0),
               (8000.0, 16000.0, 0.1),
               (10000.0, 4000.0, 1.0),
               (10000.0, 4000.0, 0.1)]

for rate, time, dt in simulations:
    num_timesteps = int(round(time / dt))
    dt_rate = rate * dt / 1000.0
    
    for i in range(10):
        data = np.random.poisson(dt_rate, num_timesteps)
        np.save(f"poisson_{rate}_{time}_{dt}_{i}.npy", data)