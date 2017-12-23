import numpy as np

import nengo
from utils import generate_scaling_functions, AreaIntercepts, Triangular

def generate(arm, kv=1,
             learning_rate=None,
             direct_mode=False,
             learned_weights=None,
             means=None, scales=None):
    """ Cerebellum model. Compensates for the inertia in the system.
    If dynamics compensation is set True, then it also generates a
    nonlinear adaptive control signal, using an efferent copy of the
    outgoing motor control signal as a training signal.
    """
    dim = arm.DOF * 2

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

    net = nengo.Network('CB')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:
        # create / connect up CB ----------------------------------------------
        net.CB = nengo.Ensemble(
            n_neurons=5000, dimensions=dim,
            radius=np.sqrt(dim),
            intercepts=AreaIntercepts(
                dimensions=dim, base=nengo.dists.Uniform(-1, 0)))
        # expecting input in form [q, dq, u]
        net.input = nengo.Node(output=scale_down, size_in=dim+arm.DOF+2)
        cb_input = nengo.Node(size_in=dim, label='CB input')
        nengo.Connection(net.input[:dim], cb_input)
        # output is [-Mdq, u_adapt]
        net.output = nengo.Node(size_in=arm.DOF*2)

        def CB_func(x):
            """ calculate the dynamic component of the OSC signal """
            x = scale_up(x)
            q = x[:arm.DOF]
            dq = x[arm.DOF:arm.DOF*2]

            # calculate inertia matrix
            M = arm.M(q=q)
            return -np.dot(M, kv * dq)
        # connect system feedback, don't account for synapses twice
        nengo.Connection(net.input[:dim], net.CB)
        nengo.Connection(net.CB, net.output[:arm.DOF],
                         function=CB_func, synapse=None)
        # dynamics adaptation
        if learning_rate is not None:
            net.CB_adapt = nengo.Ensemble(
                n_neurons=1000, dimensions=arm.DOF*2,
                radius=np.sqrt(arm.DOF*2),
                # enforce spiking neurons
                neuron_type=nengo.LIF(),
                intercepts=AreaIntercepts(
                    dimensions=arm.DOF, base=nengo.dists.Uniform(-.5, .2)))

            net.learn_encoders = nengo.Connection(
                net.input[:arm.DOF*2], net.CB_adapt,)

            # if no saved weights were passed in start from zero
            weights = (learned_weights if learned_weights is not None
                       else np.zeros((arm.DOF, net.CB_adapt.n_neurons)))
            net.learn_conn = nengo.Connection(
                # connect directly to arm so that adaptive signal
                # is not included in the training signal
                net.CB_adapt.neurons, net.output[arm.DOF:],
                # set up to initially have 0 output from adaptive connection
                transform=weights,
                learning_rule_type=nengo.PES(
                    learning_rate=learning_rate),
                synapse=None)

            # hook up training signal
            # NOTE: using a large synapse (i.e. low pass filter time constant)
            # here, because sudden changes in the training signal (as caused
            # when the target changes suddenly) will disrupt the learned
            # adaptive throughout state space
            nengo.Connection(net.input[dim:dim+2], net.learn_conn.learning_rule,
                             transform=-1, synapse=.01) #-1 for error (not reward)

    return net
