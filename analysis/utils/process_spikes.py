'''
This class processes the neural activity output from a Nengo model,
where the information has been saved to file using the numpy.savez or
numpy.savez_compressed method. It is assumed that the simulation was
run with 1ms timestep.

The data is read in and binned into bin_size ms windows to get firing
rates, then is averaged across num_average_together trials. After being
averaged the data is the smoothed with a 20ms Gaussian filter and saved
back to file using the numpy.savez_compressed method.
'''

import gc
import numpy as np


def read_spikes(folder, filename, n_neurons,
                n_trials, start_num=0):

    trajectories = []
    min_length = 1e20

    # read in all the files
    gc.disable()
    for ii in range(start_num, n_trials):
        name = '%s/%s_trial%.4i.npz' % (folder, filename, ii)
        print('reading in %s...' % name)
        spikes = np.load(name)['array1']
        spikes = spikes.reshape(-1, n_neurons)
        if spikes.shape[0] < min_length:
            min_length = spikes.shape[0]
            print('clipping all data to length %i' % min_length)
        trajectories.append(spikes)
    gc.enable()

    return np.array(trajectories[:][:min_length])


def bin_data(data, bin_size):
    n_bins = int(data.shape[1] / bin_size)
    data_binned = np.zeros((data.shape[0], n_bins, data.shape[2]))

    for ii in range(data.shape[0]):
        for jj in range(n_bins):
            data_binned[ii, jj] = np.sum(
                data[ii][jj*bin_size:(jj+1)*bin_size], axis=0)
    return data_binned


def average_trials(data, n_avg_together):
    n_avg_trials = int(data.shape[0] / n_avg_together)
    data_avg = np.zeros((n_avg_trials, data.shape[1], data.shape[2]))

    for ii in range(n_avg_trials):
        avg = np.zeros((data.shape[1], data.shape[2]))
        for jj in range(n_avg_together):
            avg += data[ii * n_avg_together + jj]
        avg /= n_avg_together
        data_avg[ii] = np.copy(avg)
    return data_avg


def apply_Gaussian_filter(data):
    gauss = np.exp(-np.linspace(-1, 1, data.shape[1])**2 / (2*.02))
    gauss /= np.sum(gauss)

    data_smoothed = np.zeros(data.shape)
    for ii in range(data.shape[0]):
        smoothed = np.zeros((data.shape[1], data.shape[2]))
        for jj in range(data.shape[2]):
            smoothed[:, jj] = np.convolve(data[ii, :, jj],
                                          gauss, mode='same')
        data_smoothed[ii] = smoothed
    return data_smoothed


def process_correlation_activity(folder, filename, n_neurons, n_trials,
                                 bin_size=10, n_reaches=8, n_avg_together=1):

    for start_num in range(n_reaches):
        # read in trajectories' neural data
        min_length = 1e20
        trajectories = []
        for ii in range(start_num, n_trials, n_reaches):
            name = '%s/%s_trial%.4i.npz' % (folder, filename, ii)
            print('reading in %s...' % name)
            spikes = np.load(name)['array1']
            spikes = spikes.reshape(-1, n_neurons)
            if spikes.shape[0] < min_length:
                min_length = spikes.shape[0]
            trajectories.append(spikes)

        # get the rest of our parameters for analysis
        n_timesteps = min_length
        n_avg_trials = int(len(trajectories) / n_avg_together)

        neuron_data = []
        for ii in range(len(trajectories)):
            neuron_data.append(trajectories[ii][:n_timesteps])
        neuron_data = np.asarray(neuron_data)

        # bin the spikes
        firing_rates = bin_data(neuron_data, bin_size=10)
        # average the trials
        fr_avgs = average_trials(firing_rates, n_avg_together=1)
        # filter the data
        filtered = apply_Gaussian_filter(fr_avgs)

        for ii in range(filtered.shape[0]):
            print('writing %i to file' % int(ii*8+start_num))
            np.savez_compressed(
                '%s/processed_data/%s_processed%.4i' %
                (folder, filename, ii*8+start_num),
                array1=filtered[ii])
