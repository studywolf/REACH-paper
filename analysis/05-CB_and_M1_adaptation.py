import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn

folder = 'data/8_reaches/CB_and_M1_adapt_data/*.npz'

plt.figure(figsize=(5, 5))
files = glob.glob(folder)

for f in files:
    path = np.load(f)['array1']
    indices = range(2500, path.shape[0] - 2500, 100)
    if path.shape[0] > 0:
        plt.plot(path[indices,0],
                    path[indices,1], 'k.', lw=2, mew=2)

plt.title('CB and M1 adaptation while reaching')
plt.tight_layout()
plt.show()
