import numpy as np

import nengo
from utils import generate_scaling_functions

def generate(arm, direct_mode=False, means=None, scales=None):
    """ Placeholder for a model of the primary sensory cortex. This
    model is simply a relay with small populations that output their input.

    input: [q, dq, hand_xy]
    output: [q, dq, hand_xy]
    """
    dim = arm.DOF*2 + 2  # represents [q, dq, hand_xy]

    # NOTE: This function will scale the input so that each dimensions is
    # in the range of -1 to 1. Since we know the operating space of the arm
    # we can set these specifically. This is a hack so that we don't need
    # 100k neurons to be able to simulate accurately generated movement,
    # you can think of it as choosing a relevant part of motor cortex to run.
    # Without this scaling, it would work still, but it would require
    # significantly more neurons to achieve the same level of performance.
    means = np.zeros(dim) if means is None else means
    scales = np.ones(dim) if scales is None else scales
    scale_down, scale_up = generate_scaling_functions(
        np.asarray(means), np.asarray(scales))


    net = nengo.Network('S1')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:
        # create / connect up S1 --------------------------------------------------
        net.S1 = nengo.networks.EnsembleArray(n_neurons=500, n_ensembles=dim)

        # expecting input in form [q, x_des]
        net.input = nengo.Node(output=scale_down, size_in=dim)
        net.output = nengo.Node(lambda t, x: scale_up(x), size_in=dim)

        # don't account for synapses twice
        nengo.Connection(net.input, net.S1.input, synapse=None)
        nengo.Connection(net.S1.output, net.output, synapse=None)

    return net
