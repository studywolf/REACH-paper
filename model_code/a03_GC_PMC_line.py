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
generalized coordinates (GC) control trace a joint trajectory generated by the
pre-motor cortex (PMC) that moves the hand in a straight line to the target.
The model has a spiking implementation of M1, CB, PMC, and a simple S1 that
just relays joint angle and velocity information.

The M1, CB, and PMC sub-networks are defined in the M1.py, CB.py, and PMC.py
files in the REACH folder, and should be read for further details on their
implementation.

In this model the PMC trajectory generation subsystem outputs a series of
target locations for the joints to follow that trace out a line with the hand.

The framework.py file is called to connect the model components defined here
together. This architecture file was defined separately to reduce overlap
between examples.

To run this model, open in Nengo GUI.

-------------------------------------------------------------------------------
"""

import importlib
import numpy as np

import nengo

from REACH import arm; importlib.reload(arm)
from REACH import M1; importlib.reload(M1)
from REACH import CB; importlib.reload(CB)
from REACH import PMC; importlib.reload(PMC)
from REACH import framework; importlib.reload(framework)


def generate():
    kp = 100
    kv = np.sqrt(kp) * 1.5

    start_target = np.array([0, 1.0])
    end_target = [0.0, 1.5]

    mouse_arm = arm.Arm2Link(dt=1e-3)
    # set the initial position of the arm
    mouse_arm.init_q = mouse_arm.inv_kinematics(start_target)
    mouse_arm.reset()


    net = nengo.Network(seed=0)
    with net:
        net.dim = mouse_arm.DOF
        net.arm_node = mouse_arm.create_nengo_node()
        net.error = nengo.Ensemble(1000, 2)
        net.xy = nengo.Node(size_in=2)

        # create an M1 model --------------------------------------------------
        net.M1 = M1.generate(mouse_arm, kp=kp,
                             operational_space=False,
                             inertia_compensation=True,
                             means=[0.6, 2.2, 0, 0],
                             scales=[.25, .25, .25, .25])

        # create an S1 model --------------------------------------------------
        net.S1 = nengo.Ensemble(n_neurons=3000, dimensions=net.dim*2+2,
                                radius=3, label='S1')
        # subtract out current position to get desired task space direction
        nengo.Connection(net.S1[:net.dim], net.error, transform=-1)

        # load in a joint trajectory to follow --------------------------------
        n_steps = 20
        xy_traj = np.vstack([
            np.linspace(start_target[0], end_target[0], n_steps),
            np.linspace(start_target[1], end_target[1], n_steps)]).T
        joint_traj = np.array([mouse_arm.inv_kinematics(xy) for xy in xy_traj]).T
        # create / connect up PMC ---------------------------------------------
        net.PMC = PMC.generate(joint_traj, speed=2)
        nengo.Connection(net.PMC.output, net.error)
        # create node to get target (x,y) for plotting
        net.transform_to_xy = nengo.Node(
            lambda t, x: mouse_arm.position(q=x, ee_only=True),
            size_in=2)
        nengo.Connection(net.PMC.output, net.transform_to_xy)
        nengo.Connection(net.transform_to_xy, net.xy)

        net.CB = CB.generate(mouse_arm, kv=kv,
                             means=[0.6, 2.2, 0, 0],
                             scales=[.125, .25, 1, 1.5])

    model = framework.generate(net=net, probes_on=False)
    return model

# Check to see if it's open in the GUI
from nengo.simulator import Simulator as NengoSimulator
if nengo.Simulator is not NengoSimulator or __name__ == '__main__':
    # connect up the models we've defined, set up the functions, probes, etc
    model = generate()
