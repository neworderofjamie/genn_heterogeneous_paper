import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns

fig, axis = plt.subplots()

fp64_vector_actor = axis.bar([0, 1], [81.7, 33.5], 0.2)
fp64_matrix_actor = axis.bar([0.2, 1.2], [163.4, 66.9], 0.2)
fp16_matrix_actor = axis.bar([0.4, 1.4], [2614.9, 1979], 0.2)
axis.set_xticks([0.2, 1.2], ["AMD MI300X", "NVIDIA Grace Hopper"])


axis.set_ylabel("Peak TFLOPs")
axis.xaxis.grid(False)
sns.despine(ax=axis)

fig.legend([fp64_vector_actor, fp64_matrix_actor, fp16_matrix_actor],
           ["FP64 (vector)", "FP64 (matrix)", "FP16 (matrix)"], 
           loc="lower center", ncol=3, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.05, 1.0, 1.0])

plt.show()