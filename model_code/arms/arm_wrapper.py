import numpy as np

import nengo


class ArmWrapper:
    """A wrapper for abr_control arm configs """

    def __init__(self, config, sim,
                 scale=20, offset=[50, 75],
                 rotation=0, reflect=[1, 1]):

        self.config = config
        self.DOF = config.N_JOINTS
        self.sim = sim
        # visualization parameters
        self.scale = scale
        self.offset = np.array(offset)
        self.rotation = rotation
        self.reflect = np.array(reflect)

        self.l = [np.sum(self.config.L[(ii+1)*2:(ii+1)*2+2])
                  for ii in range(self.config.N_JOINTS)]

        self.sim.connect()


    def apply_torque(self, u, dt=None):
        self.sim.send_forces(u, dt=dt)
        return self.sim.q, self.sim.dq


    def reset(self, q=None, dq=None):
        self.sim.init_state[::2] = q
        self.sim.reset()


    def J(self, q=None):
        """Clipping to [[dx/dq], [dy/dq]]"""
        return self.config.J('EE', q=q)[:2]


    def M(self, q=None):
        return self.config.M(q=q)


    def Mx(self, q):
        '''Calculate the task space inertia matrix'''
        M = self.M(q)
        J = self.J(q)
        # calculate the inertia matrix in task space
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        if abs(np.linalg.det(Mx_inv)) > 1e-3:
            Mx = np.linalg.inv(Mx_inv)
        else:
            # using the rcond to set singular values < thresh to 0
            # is slightly faster than doing it manually with svd
            # singular values < (rcond * max(singular_values)) set to 0
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-4)

        return Mx


    def position(self, q=None):
        """ Compute x,y position of the hand
        """
        return self.sim._position()


    def inv_kinematics(self, xy):
        """Calculate the joint angles for a given (x,y)
        hand position"""
        import scipy.optimize
        # function to optimize TODO: also return jacobian to minimize func
        def distance_to_target(q, xy, L):
            x = np.sum([L[ii] * np.cos(np.sum(q[:ii+1])) for ii in range(self.DOF)])
            y = np.sum([L[ii] * np.sin(np.sum(q[:ii+1])) for ii in range(self.DOF)])
            return np.sqrt((x - xy[0])**2 + (y - xy[1])**2)

        return scipy.optimize.minimize(
            fun=distance_to_target, x0=self.sim.q,
            args=([xy[0], xy[1]], [self.l[ii] for ii in range(self.DOF)]))['x']


    def create_nengo_node(self):
        def arm_func(t, vals):
            u = vals[:self.config.N_JOINTS]
            self.q, self.dq = self.apply_torque(u)
            # self.q = self.q % (2 * np.pi)

            # visualization code ----------------------------------------------
            offset = self.offset
            scale = self.scale
            reflect = self.reflect
            theta = self.rotation

            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            # rotate target and reflect target
            target = np.dot(R, vals[-2:]) * reflect
            # scale and offset target
            target = target * np.array([scale, scale]) + offset

            # calculate and rotate joint (x, y) positions
            xy = [self.config.Tx('joint%i' % ii, q=self.q)[:2]
                  for ii in range(self.config.N_JOINTS)]
            ee_xy = self.config.Tx('EE', q=self.q)[:2]
            xy = np.vstack([xy, ee_xy])
            xy = np.array([np.dot(R, row) * reflect * scale + offset for row in xy])

            html_code = '<svg width="100%" height="100%" viewbox="0 0 100 100">'
            for ii in range(self.config.N_JOINTS):
                html_code += '<line x1="%f" y1="%f" x2="%f" y2="%f" \
                    style="stroke:black"/>' % (xy[ii, 0], xy[ii, 1], xy[ii+1, 0], xy[ii+1, 1])
            # for ii in range(self.config.N_JOINTS):
            html_code += '<circle cx="%f" cy="%f" r="1.5" stroke="black" \
                stroke-width="1" fill="black" />' % (xy[0, 0], xy[0, 1])
            html_code += '<circle cx="%f" cy="%f" r="1.5" stroke="red" \
                stroke-width="1" fill="black" /></svg>' % (target[0], target[1])
            arm_func._nengo_html_ = html_code

            # end of visualization code ---------------------------------------

            return np.hstack([self.q, self.dq, ee_xy])

        return nengo.Node(output=arm_func, size_in=self.config.N_JOINTS + 2, label='arm')
