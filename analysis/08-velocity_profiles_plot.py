import matplotlib.pyplot as plt
import numpy as np
import seaborn

# generate and convolve velocity profile with
# Gaussian to simulate filtering effect of muscles
from scipy.special import erf
def pdf(x):
    return np.exp(-x**2/10)
def cdf(x):
    return (1 + erf(x/np.sqrt(2))) / 2
def skew(x,e=0,w=1,a=0):
    t = (x-e) / w
    return 2 / w * pdf(t) * cdf(a*t)
e = 2.0 # location
w = 2.0 # scale
x = np.linspace(0,20,5000)
skew_gauss = skew(x,e,w,2)

# get the velocity profiles
vps = []
max_length = 0
for ii in range(8):
    t = np.fromfile('data/velocity_profile/trajectory_trial%.4i.npz'%ii)
    t = t.reshape(-1, 2)

    vp_x = np.diff(t[:,0])
    vp_y = np.diff(t[:,1])
    vp = np.sqrt(vp_x**2 + vp_y**2)
    vp = np.convolve(vp, skew_gauss)
    if vp.shape[0] > max_length:
        max_length = vp.shape[0]
    vps.append(vp)

# get the mean VP and variance
full_set = np.zeros((8, max_length))
for ii in range(8):
    full_set[ii, :vps[ii].shape[0]] = vps[ii]

# normalize
full_set /= np.max(full_set)
mean_vp = np.mean(full_set, axis=0)

# get variance
var = np.var(full_set)
x = np.linspace(0, 1, len(mean_vp))
plt.title('mean and variance')
plt.fill_between(x, mean_vp - var, mean_vp + var, facecolor='k', alpha=.5)
plt.plot(x, mean_vp, 'k', lw=2)
plt.xlabel('Normalized time')
plt.ylabel('Normalized velocity')

plt.tight_layout()
plt.show()
