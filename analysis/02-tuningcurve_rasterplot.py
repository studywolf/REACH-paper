import matplotlib.pyplot as plt
import numpy as np

from utils.tuningcurve import plot as plot_tuningcurve
from utils.process_spikes import read_spikes, bin_data

# parameters for plotting this data
folder = "data/8_reaches/spiking_data"
num_reaches = 8
num_trials = 5
plot_raster = True

spikes = read_spikes(folder=folder,
                     filename="M1_spikes",
                     n_neurons=2000,
                     n_trials=num_reaches*num_trials)
spikes_binned = bin_data(spikes, bin_size=10)

for neuron in range(spikes.shape[2]):
    print('neuron: %i' % neuron)

    plt.figure(figsize=(9, 5), facecolor='white')
    for jj in range(num_reaches):
        ax = plt.subplot(4, 2, jj + 1)
        ax.minorticks_on()
        plt.xticks(range(250, 2050, 500),
                   ['-500', '0', '500', '1000'])
        ax.tick_params('both', length=5, width=2,
                       which='major', direction='out')
        ax.tick_params('both', length=3, width=1,
                       which='minor', direction='out')
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        plt.xlim(1, 2050)
        plt.ylim(0, 6)

        # index through runs under '%i%i' % (reach_direction, trial) format
        for kk in range(num_trials):
            index = jj + kk*8
            s = spikes_binned[index][:, neuron]
            np.array(s).tofile(
                '%s/processed_data/spikes_reach%i_trial%i' %
                (folder, jj, kk))

            if plot_raster is True:
                # don't plot the zeros
                ind = np.where(s >= 1)[0]
                if ind.shape[0] > 0:
                    plt.vlines(ind, 1 + kk + .5, 1 + kk + 1., lw=1)
                plt.plot(np.ones(9)*750, np.arange(9), 'k', lw=2)

    plt.tight_layout()
    plot_tuningcurve(folder=folder)
    plt.show()
