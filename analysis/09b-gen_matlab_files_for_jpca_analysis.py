import numpy as np

bin_size = 10
num_average_together = 1
num_neurons = 1000
start_num = 0
num_trials = 50

folder = 'data/attractor'
filename = 'attractor'

# read in trajectories' neural data
min_length = 1e20
for dimension in range(8):
    trajectories = []

    for jj in range(start_num, num_trials):
        name = ('%s/%s_%i-dimensional_trial%.3i.dat.npz' %
                (folder, filename, dimension, jj))
        print('reading in %s...' % name)
        spikes = np.load(name)['array1']
        spikes = spikes.reshape(-1, num_neurons)
        if spikes.shape[0] < min_length:
            min_length = spikes.shape[0]
            print('clipping all data to length %i' % min_length)
        trajectories.append(spikes)
    num_averaged_trials = int(len(trajectories) / num_average_together)
    num_timesteps = min_length
    num_bins = int(num_timesteps / bin_size)

    neuron_data = np.array(trajectories[:][:num_timesteps])

    print('binning spike data')
    # convert the spike trains to firing rates, binned in 10ms windows
    firing_rates = np.zeros((len(trajectories), num_bins, num_neurons))
    for ii in range(len(trajectories)):
        for jj in range(num_neurons):
            for kk in range(num_bins):
                firing_rates[ii, kk, jj] = np.sum(
                    neuron_data[ii][kk*bin_size:(kk+1)*bin_size, jj])

    print('averaging %i trials together' % num_average_together)
    fr_avgs = np.zeros((num_averaged_trials, num_bins, num_neurons))
    for ii in range(num_averaged_trials):
        fr_avg = np.zeros((num_bins, num_neurons))
        for jj in range(num_average_together):
            fr_avg += firing_rates[ii*num_average_together + jj]
        fr_avg /= num_average_together
        fr_avgs[ii] = fr_avg.copy()

    print('applying 20**2ms Gaussian filter')
    gauss = np.exp(-np.linspace(-1, 1, num_bins)**2 / (2*.02))
    gauss /= np.sum(gauss)
    fr_dict = {}
    for ii in range(num_averaged_trials):
        fr_smoothed = np.zeros((num_bins, num_neurons))
        for jj in range(num_neurons):
            fr_smoothed[:, jj] = np.convolve(fr_avgs[ii, :, jj],
                                             gauss, mode='same')
        fr_dict.update({'trial%i' % ii: fr_smoothed})

    print('writing full set to matlab file')
    import scipy.io
    scipy.io.savemat(
        'data/attractor/processed_data/attractor_%i_firing_rates' %
        dimension, fr_dict)
