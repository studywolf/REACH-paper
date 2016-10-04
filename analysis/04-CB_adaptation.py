import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn

folders = ['data/8_reaches/CB_adapt_data/pre_adaptation/*.npz',
           'data/8_reaches/CB_adapt_data/post_adaptation/*.npz']
titles = ['Pre-adaptation', 'Post-adaptation']

plt.figure(figsize=(4, 8))
for ii, folder in enumerate(folders):
    files = glob.glob(folder)
    plt.subplot(2, 1, ii+1)

    for f in files:
        path = np.load(f)['array1']
        indices = range(2500, path.shape[0] - 2500, 100)
        if path.shape[0] > 0:
            plt.plot(path[indices,0],
                     path[indices,1], 'k.', lw=2, mew=2)

    plt.title(titles[ii])
    plt.tight_layout()

plt.show()
