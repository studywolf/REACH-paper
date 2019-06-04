import numpy as np

import nengo
import nengolib

from utils import generate_scaling_functions, AreaIntercepts, Triangular

def generate(arm, kp=1,
             operational_space=True,
             inertia_compensation=True,
             direct_mode=False,
             means=None, scales=None):
    """ Primary motor cortex model. Computes the transform from
    task space to joint space. If inertia compensation is true, then
    it also includes dynamics compensation in task space, otherwise
    it's purely a kinematic transform.

    input: [q, x_des]
    output: [u_kinematics]
    """
    dim = arm.DOF + 2

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


    net = nengo.Network('M1')
    if direct_mode:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:
        # create / connect up M1 --------------------------------------------------
        net.M1 = nengo.Ensemble(
            n_neurons=30000,
            dimensions=dim,
            radius=np.sqrt(dim),
            encoders=nengolib.stats.ScatteredHypersphere(surface=True),
            # intercepts=AreaIntercepts(
            #     dimensions=dim, base=nengo.dists.Uniform(-1, .1)),
            intercepts=AreaIntercepts(dimensions=dim,
                                      base=Triangular(-.6, -.2, .1))
            )

        # expecting input in form [q, x_des]
        net.input = nengo.Node(output=scale_down, size_in=dim)
        net.output = nengo.Node(size_in=arm.DOF)

        def M1_func(x, operational_space, inertia_compensation):
            """ calculate the kinematic component of the OSC signal """
            x = scale_up(x)
            q = x[:arm.DOF]
            x_des = kp * x[arm.DOF:]

            # calculate hand (dx, dy)
            if operational_space:
                J = arm.J(q=q)

                if inertia_compensation:
                    # account for mass and transform to joint torques
                    Mx = arm.Mx(q=q)
                    x_des = np.dot(Mx, x_des)
                u = np.dot(J.T, x_des)
            else:
                u = x_des

                if inertia_compensation:
                    # account for mass
                    M = arm.M(q=q)
                    u = np.dot(M, x_des)

            return u

        # send in system feedback and target information
        # don't account for synapses twice
        nengo.Connection(net.input, net.M1, synapse=None)
        nengo.Connection(
            net.M1, net.output,
            function=lambda x: M1_func(
                x, operational_space, inertia_compensation),
            synapse=None)

    return net
