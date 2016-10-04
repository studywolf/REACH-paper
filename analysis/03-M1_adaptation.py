import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn

folder = 'data/8_reaches/M1_adapt_data/'
files1_name = 'distance*'
files2_name = 'ee_path*'
files3_name = 'target*'
files1 = sorted(glob.glob(folder + files1_name))
files2 = sorted(glob.glob(folder + files2_name))
files3 = sorted(glob.glob(folder + files3_name))

rms = []
full_path_ee = np.zeros((1,2))
full_path_target = np.zeros((1,2))
count = 0

gc.disable()
for f1, f2, f3 in zip(files1, files2, files3):
    path_dist = np.load(f1)['array1']
    path_ee = np.load(f2)['array1']
    path_target = np.load(f3)['array1']
    interval = path_dist.shape[0]
    if path_dist.shape[0] > 0  and count > 0:# and count < 30:
        rms.append(np.sum(np.sqrt(path_dist[:,0]**2 + path_dist[:,1]**2))*1e-3)
        full_path_ee = np.vstack([full_path_ee, path_ee])
        full_path_target = np.vstack([full_path_target, path_target])
    count += 1
gc.enable()

interval *= (count - 2)

plt.figure(figsize=(8,3))
plt.plot(rms, lw=3)
plt.title('Root mean squared error per trace')
plt.xlabel('Ellipse trace number')
plt.xlim([0, len(rms)])
plt.ylabel('RMS')
plt.tight_layout()

x = full_path_ee[range(0,interval,4000),0]
y = full_path_ee[range(0,interval,4000),1]

fig = plt.figure(figsize=(8,3.5))
time = np.linspace(0, interval * .0001, interval-5)

# plot the x axis wrt time
ax2 = fig.add_subplot(2, 2, 2)
step = 500
plottime = 500000
ax2.plot(time[2:plottime:step], full_path_ee[2:plottime:step,0], 'k', lw=2)
ax2.plot(time[2:plottime:step], full_path_target[2:plottime:step,0], 'r--', lw=2)
plt.ylim([-1.5, 1.5])
plt.xlabel('time (s)')
plt.ylabel('x (m)')

# plot the x axis wrt time
ax1 = fig.add_subplot(2, 2, 4)
ax1.plot(time[2:plottime:step], full_path_ee[2:plottime:step,1], 'k', lw=2)
ax1.plot(time[2:plottime:step], full_path_target[2:plottime:step,1], 'r--', lw=2)
plt.ylim([1, 3])
plt.xlabel('time (s)')
plt.ylabel('y (m)')

# plot x against y, with color getting darker over time
ax3 = fig.add_subplot(1, 2, 1)
offset = int(.5*len(x))
npoints = len(x)
cm = plt.get_cmap('binary')  # black and white
ax3.set_color_cycle([cm(1.*i/(npoints-1)+.25)
                     for i in range(npoints-1)])
ax3.set_aspect(1)
for i in range(1,npoints):
    ax3.plot(x[i:i+2],y[i:i+2], lw=2)

ax3.plot(full_path_target[1:interval:600,0], full_path_target[1:interval:600,1], 'r--', lw=2)
plt.xlim([-1.5, 1.5])
plt.ylim([1, 3])
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.tight_layout()
plt.show()
