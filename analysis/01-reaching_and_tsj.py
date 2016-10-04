import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn


def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window,
                                                           half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


directory = 'data/8_reaches/trajectory_data/ee_path*'
files = sorted(glob.glob(directory))

tsj = []
plot_max = 40
plot_count = 0
plt.figure(figsize=(5, 5))
for f in files:
    path = np.load(f)['array1']
    if path.shape[0] > 0:
        # only print 5 reaches in each direction to keep things
        # from getting too crowded on the plot
        if plot_count < plot_max:
            plt.plot(path[:, 0], path[:, 1], 'k.', lw=1, mew=1)
            plot_count += 1

    # calculate trajectory squared jerk
    vel = np.sqrt(np.sum((np.diff(path, axis=0) / .001)**2, axis=1))
    tsj.append(np.sum(savitzky_golay(vel, 251, 4)**2))
plt.title('Normal reaching')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()

plt.figure(figsize=(4, 4))
# normalize by average TSJ
tsj = np.asarray(tsj)
tsj /= np.mean(tsj)
std = np.std(tsj)
# plot first half against second half
half = int(tsj.shape[0] / 2)
plt.plot(tsj[:half],
         tsj[half:half*2],
         '*', mew=1)
plt.plot([1+1.96*std, 1+1.96*std], [-10, 10], 'b--', lw=2)
plt.plot([1-1.96*std, 1-1.96*std], [-10, 10], 'b--', lw=2)
plt.plot([-10, 10], [1+1.96*std, 1+1.96*std], 'b--', lw=2)
plt.plot([-10, 10], [1-1.96*std, 1-1.96*std], 'b--', lw=2)
plt.xlim([.8, 1.3])
plt.ylim([.8, 1.3])
plt.title('Total squared jerk')
plt.ylabel('Total jerk (movements 1 - %i)' % (tsj.shape[0]/2))
plt.xlabel('Total jerk (movements %i - %i)' % (tsj.shape[0]/2,
                                               tsj.shape[0]/2*2))
plt.tight_layout()
plt.show()
