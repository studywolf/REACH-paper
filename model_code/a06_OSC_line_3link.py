"""
Copyright (C) 2017 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

-------------------------------------------------------------------------------
'
A basic implementation of the REACH model (DeWolf et al, 2016) using
operational space control (OSC) to reach between two points in a line. The
model has a spiking implementation of M1, CB, and a simple S1 that just relays
joint angle and velocity information.

The M1, CB, and PMC sub-networks are defined in the M1.py and CB.py files in
the REACH folder, and should be read for further details on their
implementation.

In this model the PMC trajectory generation subsystem is replaced with a Nengo
Node that outputs a target position for the hand, and alternates every 2
seconds.

The framework.py file is called to connect the model components defined here
together. This architecture file was defined separately to reduce overlap
between examples.

To run this model, open in Nengo GUI.

-------------------------------------------------------------------------------
"""

import importlib
import numpy as np

# from abr_control.arms.twojoint import Config, ArmSim
from abr_control.arms.threejoint import Config, ArmSim
import nengo

from REACH import M1; importlib.reload(M1)
from REACH import CB; importlib.reload(CB)
from REACH import S1; importlib.reload(S1)
from REACH import framework; importlib.reload(framework)
from arms import arm_wrapper; importlib.reload(arm_wrapper)


def generate():
    kp = 50
    kv = np.sqrt(kp)

    start_target = np.array([0, 1.5])
    end_target = [0, 2.25]

    # arm_sim = arm.Arm2Link(dt=1e-3)
    config = Config()
    arm_sim = arm_wrapper.ArmWrapper(config, ArmSim(config), reflect=[1, -1])
    # set the initial position of the arm
    arm_sim.reset(q=arm_sim.inv_kinematics(start_target))

    net = nengo.Network()#seed=0)
    # net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    net.config[nengo.Connection].synapse = 0.0
    with net:
        net.dim = config.N_JOINTS
        net.arm_node = arm_sim.create_nengo_node()
        net.error = nengo.Ensemble(500, 2)
        net.xy = nengo.Node(size_in=2)

        # create an M1 model ------------------------------------------------------
        net.M1 = M1.generate(arm_sim, kp=kp,
                             operational_space=True,
                             inertia_compensation=True,
                             # direct_mode=True,
                             means=[np.pi/2, np.pi/2, 0, 0, 0],
                             scales=[np.pi/2, np.pi/2, np.pi, 1, 1],
                             # means=[np.pi/2, np.pi/2, 0, 0],
                             # scales=[np.pi/2, np.pi/2, 1, 1],
                             )

        # create an S1 model ------------------------------------------------------
        net.S1 = S1.generate(arm_sim,
                             means=[np.pi, np.pi, 0, 0, 0, 0, 0, 1.5],
                             scales=[np.pi, np.pi, np.pi, 1.5, 1.5, 1.5, 1, 1.5],
                             # direct_mode=True,
                             )
        # subtract out current position to get desired task space direction
        nengo.Connection(net.S1.output[net.dim*2:], net.error, transform=-1)

        # create a PMC substitute -------------------------------------------------
        net.PMC = nengo.Network('PMC')
        with net.PMC:
            def PMC_func(t):
                """ every 2 seconds alternate targets """
                if t % 2 < 1:
                    return end_target
                return start_target
            net.PMC.output = nengo.Node(output=PMC_func, label='PMC')
        # send target for calculating control signal
        nengo.Connection(net.PMC.output, net.error)
        # store target (x,y) for plotting
        nengo.Connection(net.PMC.output, net.xy)

        # create  a CB model ------------------------------------------------------
        net.CB = CB.generate(arm_sim, kv=kv,
                             # means=[0.6, 2.2, 0, 0],
                             # scales=[.125, .25, 1, 1.5])
                             means=[np.pi/2, np.pi/2, 0, 0, 0, 0],
                             scales=[np.pi/2, np.pi/2, np.pi, 2, 2, 4],
                             # means=[np.pi, np.pi, 0, 0],
                             # scales=[np.pi, np.pi, 1, 1.5],
                             # direct_mode=True,
                             )

        net.probe_xy = nengo.Probe(net.arm_node[net.dim*2:])

    model = framework.generate(net=net, probes_on=False)
    return model

# Check to see if it's open in the GUI
from nengo.simulator import Simulator as NengoSimulator
if nengo.Simulator is not NengoSimulator or __name__ == '__main__':
    # connect up the models we've defined, set up the functions, probes, etc
    model = generate()
    sim = nengo.Simulator(model)
    sim.run(10)

    xy = sim.data[model.probe_xy]
    import matplotlib.pyplot as plt
    plt.plot(sim.trange(), xy)
    plt.show()
