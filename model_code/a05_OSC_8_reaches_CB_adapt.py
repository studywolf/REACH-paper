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
An implementation of the REACH model (DeWolf et al, 2016) using operational
space control (OSC) to perform the centre-out reaching task, starting from a
centre position and reaching to 8 targets around a circle. The model has a
spiking implementation of M1, CB, and a simple S1 that just relays joint angle
and velocity information.

The M1, CB, and PMC sub-networks are defined in the M1.py and CB.py files in
the REACH folder, and should be read for further details on their
implementation.

In this model the PMC trajectory generation subsystem is replaced with a Nengo
Node that outputs the target positions for the hand around a circle,
changing targets every 2 seconds, and returning to centre between each reach.

Additionally, a perturbation based on the forcefield described in (Shadmehr,
1994) has been added, and the dynamics adaptation function of the cerebellum
(CB) has been activated. Initially, the model will be affected by the
unexpected perturbations of the forcefield, but after approximately 150
seconds of training the system will have learned to compensate and the reaches
will be straight again.

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
from REACH import S1; importlib.reload(S1)
from REACH import framework; importlib.reload(framework)

force_matrix = np.array([[-10.1, -11.1],
                         [-11.2, 11.1]])
def perturbation_func(t, x):
    """ x = [q0, q1, dq0, dq1]
    This adds a forcefield as described in (Shadmehr, 1994)
    """
    return np.dot(force_matrix, x[2:]) * .4

def generate():
    kp = 50
    kv = np.sqrt(kp) * 1.5

    n_reaches = 8
    dist = .25
    center = [0, 1.25]
    end_points = [[dist * np.cos(theta) + center[0],
                dist * np.sin(theta) + center[1]]
                for theta in np.linspace(0, 2*np.pi, n_reaches+1)][:-1]
    targets = []
    for ep in end_points:
        targets.append(center)
        targets.append(ep)
    targets = np.array(targets)

    arm_sim = arm.Arm2Link(dt=1e-3)
    # set the initial position of the arm
    arm_sim.init_q = arm_sim.inv_kinematics(center)
    arm_sim.reset()

    net = nengo.Network(seed=0)
    with net:
        net.dim = arm_sim.DOF
        net.arm_node = arm_sim.create_nengo_node()
        net.error = nengo.Ensemble(1000, 2)
        net.xy = nengo.Node(size_in=2)

        # create an M1 model ------------------------------------------------------
        net.M1 = M1.generate(arm_sim, kp=kp,
                             operational_space=True,
                             inertia_compensation=True,
                             means=[0.6, 2.2, 0, 0],
                             scales=[.25, .3, .25, .25])

        # create an S1 model ------------------------------------------------------
        net.S1 = S1.generate(arm_sim,
                             means=[.6, 2.2, -.5, 0, 0, 1.25],
                             scales=[.5, .5, 1.7, 1.5, .75, .75])

        # subtract out current position to get desired task space direction
        nengo.Connection(net.S1.output[net.dim*2:], net.error, transform=-1)

        # create a PMC substitute -------------------------------------------------
        net.PMC = nengo.Network('PMC')
        with net.PMC:
            def PMC_func(t):
                """ every 2 seconds change target """
                return targets[int(t / 2) % len(targets)]
            net.PMC.output = nengo.Node(output=PMC_func, label='PMC')
        # send target for calculating control signal
        nengo.Connection(net.PMC.output, net.error)
        # send target (x,y) for plotting
        nengo.Connection(net.PMC.output, net.xy)

        # create  a CB model ------------------------------------------------------
        net.CB = CB.generate(arm_sim, kv=kv,
                             learning_rate=1e-4,
                             means=[0.65, 2.2, 0, 0],
                             scales=[.25, .3, .75, 1.25])

        perturbation_node = nengo.Node(perturbation_func, size_in=4)
        nengo.Connection(net.arm_node[:arm_sim.DOF*2], perturbation_node)
        nengo.Connection(perturbation_node, net.arm_node[:arm_sim.DOF])

    model = framework.generate(net=net, probes_on=False)
    return model

# Check to see if it's open in the GUI
from nengo.simulator import Simulator as NengoSimulator
if nengo.Simulator is not NengoSimulator or __name__ == '__main__':
    # connect up the models we've defined, set up the functions, probes, etc
    model = generate()
