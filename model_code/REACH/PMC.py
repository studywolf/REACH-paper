import numpy as np
from scipy import interpolate

import nengo

import importlib
from . import forcing_functions; importlib.reload(forcing_functions)


def generate(y_des, speed=1, alpha=1000.0, direct_mode=False):
    """ Pre-motor cortex model, implements a dynamical movement
    primitive that generates the provided trajectory.

    input: None
    output: [trajectory[0], trajectory[1]]
    """
    beta = alpha / 4.0

    # generate the forcing function
    forces, _, goals = forcing_functions.generate(
        y_des=y_des, rhythmic=False, alpha=alpha, beta=beta)

    # create alpha synapse, which has point attractor dynamics
    tau = np.sqrt(1.0 / (alpha*beta))
    alpha_synapse = nengo.Alpha(tau)

    net = nengo.Network('PMC')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:
        net.output = nengo.Node(size_in=2)

        # create a start / stop movement signal
        time_func = lambda t: min(max((t * speed) % 4.5 - 2.5, -1), 1)

        def goal_func(t):
            t = time_func(t)
            if t <= -1:
                return goals[0]
            return goals[1]
        net.goal = nengo.Node(output=goal_func, label='goal')

        # -------------------- Ramp ---------------------------------

        ramp_node = nengo.Node(output=time_func, label='ramp')
        net.ramp = nengo.Ensemble(
            n_neurons=500, dimensions=1, label='ramp ens')
        nengo.Connection(ramp_node, net.ramp)

        # ------------------- Forcing Functions ---------------------

        def relay_func(t, x):
            t = time_func(t)
            if t <= -1:
                return [0, 0]
            return x
        # the relay prevents forces from being sent when resetting
        relay = nengo.Node(output=relay_func, size_in=2, label='relay gate')

        domain = np.linspace(-1, 1, len(forces[0]))
        x_func = interpolate.interp1d(domain, forces[0])
        y_func = interpolate.interp1d(domain, forces[1])
        nengo.Connection(net.ramp, relay[0], transform=1.0/alpha/beta,
                         function=x_func, synapse=alpha_synapse)
        nengo.Connection(net.ramp, relay[1], transform=1.0/alpha/beta,
                         function=y_func, synapse=alpha_synapse)
        nengo.Connection(relay, net.output)

        nengo.Connection(net.goal[0], net.output[0], synapse=alpha_synapse)
        nengo.Connection(net.goal[1], net.output[1], synapse=alpha_synapse)

    return net
