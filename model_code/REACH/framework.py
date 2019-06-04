import nengo


def generate(net=None,  # define PMC, M1, CB, S1 inside net
             probes_on=False):  # set True to record data
    """ Connect up the PMC, M1, CB, and S1 sub-networks up for
    the REACH model.
    """

    config = nengo.Config(nengo.Connection, nengo.Ensemble)
    with net, config:

        dim = net.dim  # the number of DOF of the arm
        net.probes_on = probes_on

        relay = nengo.Node(size_in=dim, label='relay')
        # connect the control signal summed from M1 and CB to the arm
        nengo.Connection(relay, net.arm_node[:dim])
        # send in (x, y) of target for plotting
        nengo.Connection(net.xy, net.arm_node[dim:])

        if getattr(net, "M1", False):
            # project control signal into output relay
            nengo.Connection(net.M1.output, relay)
            # send in x_des signal to M1
            nengo.Connection(net.error[:dim], net.M1.input[dim:])

        if getattr(net, "S1", False):
            nengo.Connection(net.arm_node, net.S1.input)
            if getattr(net, "M1", False):
                # connect up sensory feedback
                nengo.Connection(net.S1.output[:dim], net.M1.input[:dim])

        if getattr(net, "CB", False):
            # connect up sensory feedback
            nengo.Connection(net.arm_node[:dim*2], net.CB.input[:dim*2])

            # send in training signal
            nengo.Connection(relay, net.CB.input[dim*2:dim*3])

            # project dynamics compensation term into relay
            # to be included in the training signal for u_adapt
            nengo.Connection(net.CB.output[:dim], relay)

            # send u_adapt output directly to arm
            nengo.Connection(net.CB.output[dim:], net.arm_node[:dim])

        if probes_on:
            net.probe_S1 = nengo.Probe(net.S1.output)
            net.probe_M1 = nengo.Probe(net.M1.output)
            net.probe_CB = nengo.Probe(net.CB.output)

    return net
