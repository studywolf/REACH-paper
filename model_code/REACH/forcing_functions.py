import numpy as np
from scipy import interpolate

import nengo

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def generate(y_des, dt=.001, n_samples=1000,
             alpha=10.0, beta=10.0/4.0, rhythmic=False,
             plotting=False, generalized=False):

    # scale our trajectory and find the center point
    center = np.sum(y_des, axis=1) / y_des.shape[1]
    # center trajectory around (0, 0)
    # y_des -= center[:, None]
    if rhythmic is True:
        goal = np.zeros(center.shape)
        return_goal = center
    else:
        start = y_des[:, 0]
        goal = y_des[:, -1]

        # if we're generalizing things, rotate the trajectory
        # so that start and end points lie on x-axis
        if generalized == True:
            # move start to (0, 0)
            y_des -= start[:, None]
            start = y_des[:, 0]
            goal = y_des[:, -1]
            # move end to x-axis
            theta = np.arctan2(goal[1], goal[0])
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            y_des = np.dot(R.T, y_des)
            # normalize movement length to 1
            y_des /= y_des[0, -1]
            # get new end point on x-axis
            goal = y_des[:, -1]
    print('start: ', start)
    print('goal: ', goal)

    # interpolate our desired trajectory to smooth out the sampling
    path = np.zeros((y_des.shape[0], n_samples))
    x = np.linspace(-1, 1, y_des.shape[1])
    for d in range(y_des.shape[0]):
        path_gen = interpolate.interp1d(x, y_des[d])
        for ii, t in enumerate(np.linspace(-1, 1, n_samples)):
            path[d, ii] = path_gen(t)
    y_des = path

    # calculate velocity of y_des
    dy_des = np.diff(y_des) / dt / n_samples
    # add zero to the beginning of every row
    dy_des = np.hstack((np.zeros((y_des.shape[0], 1)), dy_des))

    # calculate acceleration of y_des
    ddy_des = np.diff(dy_des) / dt / n_samples
    # add zero to the beginning of every row
    ddy_des = np.hstack((np.zeros((y_des.shape[0], 1)), ddy_des))

    forces = []
    pa_forces = []
    forcing_functions = []
    x = np.linspace(-np.pi, np.pi, n_samples)
    for ii in range(y_des.shape[0]):
        # find the force required to move along this trajectory
        # by subtracting out the effects of the point attractor
        pa_forces.append(alpha * (beta * (goal[ii] - y_des[ii]) - dy_des[ii]))

        forces.append(ddy_des[ii] - pa_forces[ii])

        # if generalized is True and rhythmic is False:
        #     forces[ii] /= (goal[ii] - start[ii])

        # create the lookup table set for training oscillator output
        forcing_functions.append(
            nengo.utils.connection.target_function(
                np.array([np.cos(x), np.sin(x)]).T,
                forces[ii]))

    if rhythmic is False:
        goal = [start, goal]
    else:
        goal = return_goal
    return forces, forcing_functions, goal
