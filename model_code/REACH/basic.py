import importlib
import numpy as np

import nengo

from .. import arm; importlib.reload(arm)
from .. import M1; importlib.reload(M1)
from .. import CB; importlib.reload(CB)

import framework; importlib.reload(framework)


kp = 100
kv = np.sqrt(kp)
seed = 0

start_target = np.array([0, 1.25]) #+ np.random.random(2) - 0.5
end_target = [0.0, 1.5]

mouse_arm = arm.Arm2Link(dt=1e-3)
# set the initial position of the arm
mouse_arm.init_q = mouse_arm.inv_kinematics(start_target)
mouse_arm.reset()


net = nengo.Network(seed=seed)
with net:

    # create an M1 model ------------------------------------------------------
    net.M1 = M1.generate(mouse_arm, kp=kp,
                        # if the PMC is being modeled, then M1
                        # is not transforming from operational space
                        operational_space=(not model_PMC),
                        inertia_compensation=True)

    # create an S1 model ------------------------------------------------------
    net.S1 = nengo.Ensemble(n_neurons=3000, dimensions=dim*2+2,
                            radius=3, label='S1')

    # create a PMC substitute -------------------------------------------------
    def PMC_func(t):
        """ every 1 seconds change target """
        if t % 2 < 1:
            return end_target
        return start_target
    net.PMC = nengo.Node(output=PMC_func, label='PMC')

    # create  a CB model ------------------------------------------------------
    net.CB = CB.generate(mouse_arm, kv=kv,
                         dynamics_adaptation=True,
                         learned_weights=CB_weights,
                         learned_encoders=CB_encoders)

model = framework.generate(net=net, probes_on=False)
