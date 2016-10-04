import matplotlib.pyplot as plt
import numpy as np
import seaborn
import sys

folder = 'data/correlations-CB'
filename = 'CB_spikes'
n_trials  = 10  # can be up to 100
n_neurons = 10000

# to plot correlations with different movement parameters call
# script with an argument corresponding to data type you want to plot
correlate_with = ['ee_path',
                  'ee_velocity',
                  'distance_from_target',
                  'joint_angles',
                  'joint_velocities',
                  'joint_accelerations',
                  'control_signal'][int(sys.argv[1])]
xlabel = ['End-effector position (ft)',
           'End-effector velocity (ft / s)',
           'Distance from target (ft)',
           'Joint angles (degrees)',
           'Joint velocities (degrees / s)',
           'Joint accelerations (degrees / s^2)',
           'Control signal (Nm)'][int(sys.argv[1])]
print('Correlating neural activity with %s' % correlate_with)

activity_set = []
data_set = []
neural_folder = '%s/neural_activity' % folder
move_param_folder = '%s/movement_parameters' % folder
# choose randomly 50 trials to correlate across
trials = np.random.permutation(range(n_trials))[:50]
for ii in trials:
    print('reading in trial%i ' % ii)

    activity = np.load('%s/%s_processed%.4i.npz' %
                       (neural_folder, filename, ii))['array1']

    if correlate_with in ('distance_from_target', 'end-effector position'):
        data = np.load('%s/%s_trial%.4i.npz' %
                       (move_param_folder, correlate_with, ii))['array1']
    else:
        if correlate_with == 'joint_accelerations':
            data = np.load('%s/joint_velocities_trial%.4i.npz' %
                           (move_param_folder, ii))['array1']
            data = np.diff(data, axis=0) / .001
        elif correlate_with == 'ee_velocity':
            data = np.load('%s/ee_path_trial%.4i.npz' %
                           (move_param_folder, ii))['array1']
            data = np.diff(data, axis=0) / .001
        else:
            data = np.load('%s/%s_trial%.4i.npz' %
                           (move_param_folder, correlate_with, ii))['array1']

        if correlate_with in ('joint_velocities', 'joint_accelerations',
                              'ee_velocity', 'control_signal'):
            gauss = np.exp(-np.linspace(-1, 1, 1000)**2 / (2*.02))
            gauss /= np.sum(gauss)
            data[:, 0] = np.convolve(data[:, 0], gauss, mode='same')
            data[:, 1] = np.convolve(data[:, 1], gauss, mode='same')

    # account for the 10 ms spike binning
    data = data[range(0, data.shape[0]-10, 10)]
    # chop data down to size because all activity
    # was clipped during the processing
    data = data[:activity.shape[0]]

    # double check they're the same size
    activity = activity[:data.shape[0]]

    # now chop off the start and tail
    buff_start = 300
    buff_end = 250
    activity = activity[buff_start:-buff_end]
    data = data[buff_start:-buff_end]

    activity_set.append(activity)
    data_set.append(data)

maxval = 0.0
for jj in range(n_neurons):

    data_plot_set = []
    activity_plot_set = []
    for ii in range(1, len(activity_set)):
        # find high coeff neuron in first trial

        index_xy = 1
        if abs(np.max(activity_set[ii][:, jj]) -
               np.min(activity_set[ii][:, jj])) > .15:

            coeff = abs(np.corrcoef(data_set[ii][:, index_xy],
                        activity_set[ii][:, jj])[0, 1])
            if coeff > maxval:
                maxval = coeff
            if coeff > .87 and not np.isinf(coeff):
                print(coeff)
                index = np.array(range(np.random.randint(30),
                                 data_set[ii].shape[0], 40))
                if index.shape[0] > 0:
                    data_plot_set.append(data_set[ii][index, index_xy])
                    activity_plot_set.append(activity_set[ii][index, jj] / .01)

    if len(data_plot_set) > 0:
        plt.figure(figsize=(4, 4))
        # fit a line to the data
        for ii in range(len(data_plot_set)):
            plt.plot(data_plot_set[ii], activity_plot_set[ii], 'k*', mew=3)

        data_stack = data_plot_set[0]
        activity_stack = activity_plot_set[0]
        for ii in range(1, len(data_plot_set)):
            data_stack = np.hstack([data_stack, data_plot_set[ii]])
            activity_stack = np.hstack([activity_stack, activity_plot_set[ii]])

        x = np.linspace(-10000, 10000, 2)
        linevals = np.polyfit(data_stack, activity_stack, 1)
        plt.plot(x, x*linevals[0] + linevals[1], 'r--', lw=2)
        plt.xlim([min(data_stack), max(data_stack)])
        plt.ylim([min(activity_stack), max(activity_stack)])
        plt.title('Neural correlation plot')
        plt.xlabel(xlabel)
        plt.ylabel('Firing rate (Hz)')
        plt.tight_layout()
        plt.show()

print('Strongest correlation coefficient: %f' % maxval)
