import numpy as np
import matplotlib.pyplot as plt

pc1var_values = np.array([[0.7721, 0.5290, 0.5826, 0.4893, 0.5143, 0.4893, 0.3508],
                          [0.8074, 0.5838, 0.3760, 0.4188, 0.2883, 0.3341, 0.2719],
                          [0.7861, 0.4677, 0.3867, 0.4192, 0.3168, 0.2570, 0.2703],
                          [0.8602, 0.5529, 0.4620, 0.3987, 0.4485, 0.2715, 0.3023],
                          [0.7823, 0.5344, 0.3786, 0.3960, 0.4758, 0.3499, 0.2631],
                          [0.8368, 0.5812, 0.4319, 0.2976, 0.3094, 0.3019, 0.3208]])

mean = np.mean(pc1var_values, axis=0)
std = np.std(pc1var_values, axis=0)
Zscore = 1.96 # Z value for 95% confidence interval
conf_int = Zscore * std / np.sqrt(pc1var_values.shape[0])

upper_bound = mean + conf_int
lower_bound = mean - conf_int

fig = plt.figure(figsize=(7, 4))
plt.plot(range(1,8), mean, 'k', lw=3)
plt.fill_between(range(1,8), upper_bound, lower_bound, color='k', alpha=.5)
plt.xlabel('Dimensions represented')
plt.ylabel('Variance explained by PC1')
plt.tight_layout()
plt.show()
