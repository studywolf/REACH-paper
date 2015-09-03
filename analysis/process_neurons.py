import numpy as np

bin_size = 10
num_average_together = 1
num_neurons = 1000
start_num = 0
num_trials = 50

folder = '' 
filename = 'attractor'
# folder = 'integrator/'
# filename = 'integrator_'
# folder = 'ramp/'
# filename = 'ramp_'
# folder = 'spline/'
# filename = 'spline_'

# read in trajectories' neural data 
min_length = 1e20

for dimension in range(8):
    trajectories = []

    for jj in range(start_num, num_trials):
        name = '%s%s_%i-dimensional_trial%.3i.dat.npz'%(folder, filename, dimension, jj)
        print 'reading in %s...'%name
        spikes = np.load(name)['array1']
        spikes = spikes.reshape(-1, num_neurons)
        if spikes.shape[0] < min_length:
            min_length = spikes.shape[0]
            print min_length
        trajectories.append(spikes)

    print len(trajectories)

    num_averaged_trials = len(trajectories) / num_average_together

    print spikes

    # get the rest of our parameters for analysis
    num_timesteps = min_length
    print num_timesteps
    num_bins = num_timesteps / bin_size
    print num_bins

    neuron_data = np.array(trajectories[:][:num_timesteps])

    print 'binning spike data'

    # now convert the spike trains to firing rates, binned in 10ms windows
    firing_rates  = np.zeros((len(trajectories), num_bins, num_neurons))
    print firing_rates.shape
    # fr_dict = {}
    for ii in range(len(trajectories)):
        print ii
        # firing_rates  = np.zeros((num_bins, num_neurons))
        for jj in range(num_neurons):
            for kk in range(num_bins):
                firing_rates[ii, kk,jj] = np.sum(neuron_data[ii][kk*bin_size:(kk+1)*bin_size, jj])
                # firing_rates[kk,jj] = np.sum(neuron_data[ii, kk*bin_size:(kk+1)*bin_size, jj])
        # fr_dict.update({'trial%i'%ii:firing_rates.copy()})

    print 'averaging %i trials together'%num_average_together

    # now average num_average_together runs together
    # fr_dict = {}
    fr_avgs = np.zeros((num_averaged_trials, num_bins, num_neurons))
    for ii in range(num_averaged_trials):
        fr_avg = np.zeros((num_bins, num_neurons))
        for jj in range(num_average_together):
            fr_avg += firing_rates[ii*num_average_together + jj]
        fr_avg /= num_average_together
        fr_avgs[ii] = fr_avg.copy()
        # fr_dict.update({'trial%i'%ii:fr_avg})

    print 'applying 20**2ms Gaussian filter'
    gauss = np.exp(-np.linspace(-1,1,num_bins)**2 / (2*.02))
    gauss /= np.sum(gauss)
    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(-1,1,100), gauss)
    # plt.show()
    fr_dict = {}
    for ii in range(num_averaged_trials):
        fr_smoothed = np.zeros((num_bins, num_neurons))
        for jj in range(num_neurons):

            fr_smoothed[:, jj] = np.convolve(fr_avgs[ii,:,jj], gauss, mode='same')

            # import matplotlib.pyplot as plt
            # plt.plot(fr_avgs[ii,:,jj])
            # plt.plot(fr_smoothed)
            # plt.show()

        # print 'writing ', ii+start_num, ' to file'
        # np.savez_compressed('%s/post_processing_%s_trial%i'%(folder, filename, ii+start_num), array1=fr_smoothed)
        # print 'written'

        fr_dict.update({'trial%i'%ii:fr_smoothed})

    print 'writing to matlab file'
    import scipy.io
    scipy.io.savemat('attractor_%i_firing_rates'%dimension, fr_dict)
