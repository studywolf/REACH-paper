import numpy as np
import scipy.special

import nengo

def generate_scaling_functions(means, scales, filename=None):
    """ Generates functions to scale the input to neural ensembles.
    You can either pass in the means and scales or have them calculated
    from a npz file storing the input to the ensemble.
    """

    if filename is not None:
        data = np.load(filename)['arr_0']
        means = np.mean(data, axis=0)
        scales = np.max(data, axis=0) - np.min(data, axis=0)

    def scale_down(t, x):
        index = np.where(scales != 0)
        x[index] = (x[index] - means[index]) / scales[index]
        index = np.where(scales == 0)
        x[index] = x[index] - means[index]
        return x
    scale_up = lambda x: x * scales + means

    return scale_down, scale_up


def plot_3D_encoders(filename):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    a = np.load(filename)['arr_0']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], 'r', marker='o')
    ax.set_zlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])


    # draw the unit sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    plt.show()


class AreaIntercepts(nengo.dists.Distribution):
    """ Generate an optimally distributed set of intercepts in
    high-dimensional space.
    """
    dimensions = nengo.params.NumberParam('dimensions')
    base = nengo.dists.DistributionParam('base')

    def __init__(self, dimensions, base=nengo.dists.Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def __repr(self):
        return ("AreaIntercepts(dimensions=%r, base=%r)" %
                (self.dimensions, self.base))

    def transform(self, x):
        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(1 - scipy.special.betaincinv(
            (self.dimensions + 1) / 2.0, 0.5, x + 1))

    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for i in range(len(s)):
            s[i] = self.transform(s[i])
        return s


class Triangular(nengo.dists.Distribution):
    left = nengo.params.NumberParam('dimensions')
    right = nengo.params.NumberParam('dimensions')
    mode = nengo.params.NumberParam('dimensions')

    def __init__(self, left, mode, right):
        super(Triangular, self).__init__()
        self.left = left
        self.right = right
        self.mode = mode

    def __repr__(self):
        return ("Triangular(left=%r, mode=%r, right=%r)" %
                (self.left, self.mode, self.right))

    def sample(self, n, d=None, rng=np.random):
        if d is None:
            return rng.triangular(self.left, self.mode, self.right, size=n)
        else:
            return rng.triangular(self.left, self.mode, self.right, size=(n, d))
