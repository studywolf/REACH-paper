import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import leastsq


def plot(folder):
    means = []
    sems = []
    for ii in range(8):
        spikes = []
        for jj in range(5):
            a = np.fromfile(
                '%s/processed_data/spikes_reach%i_trial%i' %
                (folder, ii, jj))
            # get the number of spikes, divide by 2 to get 1 second average
            spikes.append(len(np.where(a > 0)[0]) / 2.)
        means.append(np.mean(np.array(spikes)))
        sems.append(scipy.stats.sem(np.array(spikes)))

    means = np.roll(np.array(means), 2)

    guess_mean = np.mean(means)
    guess_std = 3*np.std(means)/(2**0.5)
    guess_phase = 0

    t = np.arange(8)
    # Define the function to optimize, in this case, want to minimize
    # the difference between the actual data and guessed parameters
    optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - means
    est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std,
                                                           guess_phase,
                                                           guess_mean])[0]

    data_fit = (est_std *
                np.sin(np.arange(0, np.pi*2, .1) + est_phase + .75) +
                est_mean - 2)
    # generate the tuning curve from Georgopoulos paper
    x = np.linspace(0, 350, len(data_fit))
    georgo = (32.37 + 7.281 *
              np.sin(np.radians(x)) - 21.343 * np.cos(np.radians(x)))
    print('correlation coefficient: ', np.corrcoef(data_fit, georgo))

    plt.figure(figsize=(7, 5), facecolor='white')
    ax = plt.subplot(1, 1, 1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(np.arange(0, 360, 45))

    # plot spike data and error bars
    plt.errorbar(np.arange(0, 350, 45), means, yerr=sems, fmt='ko', lw=1)
    plt.plot(x, data_fit, 'k')
    plt.plot(x, georgo)

    plt.yticks(np.arange(0, 65, 20))
    plt.xlim([-45, 360])
    plt.ylim([0, 60])
    plt.ylabel('Impulses / sec')
    plt.xlabel('Direction of movement')
