from pandas import read_csv
import matplotlib.pyplot as plt

mi300 = read_csv("va_benchmark_perf_mi300a.csv", delimiter=",")

# Split dense and sparse
mi300_dense = mi300[mi300["Dense connectivity"] == True]
mi300_dense_half = mi300_dense[mi300_dense["Half precision"] == True]
mi300_dense_single = mi300_dense[mi300_dense["Half precision"] == False]
mi300_sparse = mi300[mi300["Dense connectivity"] == False]
mi300_sparse_half = mi300_sparse[mi300_sparse["Half precision"] == True]
mi300_sparse_single = mi300_sparse[mi300_sparse["Half precision"] == False]


fig, axes = plt.subplots(1,2)
axes[0].plot(mi300_dense_single["Num neurons"], mi300_dense_single["Neuron update time [s]"], label="Single precision")
axes[0].plot(mi300_dense_half["Num neurons"], mi300_dense_half["Neuron update time [s]"], label="Half precision")
axes[0].legend()
axes[1].plot(mi300_dense_single["Num neurons"], mi300_dense_single["Presynaptic update time [s]"], label="Dense, single precision")
axes[1].plot(mi300_dense_half["Num neurons"], mi300_dense_half["Presynaptic update time [s]"], label="Dense, half precision")
axes[1].plot(mi300_sparse_single["Num neurons"], mi300_sparse_single["Presynaptic update time [s]"], label="Sparse, single precision")
axes[1].plot(mi300_sparse_half["Num neurons"], mi300_sparse_half["Presynaptic update time [s]"], label="Sparse, half precision")
axes[1].legend()
plt.show()
