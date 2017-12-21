import numpy as np

import nengo


class Arm2Link:
    """A base class for arm simulators"""

    def __init__(self, init_q=[.75613, 1.8553], init_dq=[0., 0.], dt=1e-4):

        self.init_q = np.copy(init_q)
        self.init_dq = np.copy(init_dq)
        self.singularity_thresh = 0.00025

        self.DOF = 2
        self.dt = dt

        # length of arm links
        self.l = [1.5, 1.3]
        # mass of links
        m = [1.98, 1.32]
        # moments of inertia
        izz = [.4, .2]
        # arm limits (radians)
        self.q_upper_limits = np.array([3.0, 3.0])
        self.q_lower_limits = np.array([0.1, 0.1])

        # create mass matrices at COM for each link
        self.M1 = np.zeros((6, 6))
        self.M2 = np.zeros((6, 6))
        self.M1[0:3, 0:3] = np.eye(3)*m[0]
        self.M1[3:, 3:] = np.eye(3)*izz[0]
        self.M2[0:3, 0:3] = np.eye(3)*m[1]
        self.M2[3:, 3:] = np.eye(3)*izz[1]

        # compute non changing constants
        self.K1 = ((1.0/3.0 * m[0] + m[1]) * self.l[0]**2. +
                   1.0/3.0 * m[1] * self.l[1]**2.0)
        self.K2 = m[1] * self.l[0] * self.l[1]
        self.K3 = 1.0/3. * m[1] * self.l[1]**2.
        self.K4 = 1.0/2.0 * m[1] * self.l[0] * self.l[1]

        self.reset()

    def apply_torque(self, u, dt=None):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly.

        u np.array: the control signal to apply
        dt float: the timestep
        """
        dt = self.dt if dt is None else dt
        q0, q1 = self.q
        dq0, dq1 = self.dq

        # equations solved for angles
        C2 = np.cos(q1)
        S2 = np.sin(q1)
        M11 = (self.K1 + self.K2*C2)
        M12 = (self.K3 + self.K4*C2)
        M21 = M12
        M22 = self.K3
        H1 = -self.K2*S2*dq0*dq1 - 1/2.*self.K2*S2*dq1**2.
        H2 = 1/2.*self.K2*S2*dq0**2.

        # update joint accelerations
        ddq1 = (H2*M11 - H1*M21 - M11*u[1] + M21*u[0]) / (M12**2. - M11*M22)
        ddq0 = (-H2 + u[1] - M22*ddq1) / M21

        # update joint velocities
        dq1 = dq1 + ddq1*dt
        dq0 = dq0 + ddq0*dt

        # update joint angles
        q1 = q1 + dq1*dt
        q0 = q0 + dq0*dt

        # enforce joint angle limits
        def check_limits(q, dq, lower_limit, upper_limit):
            if q < lower_limit:
                q = lower_limit
                dq = 0.0
            if q > upper_limit:
                q = upper_limit
                dq = 0.0
            return q, dq
        q0, dq0 = check_limits(q0, dq0, self.q_lower_limits[0],
                               self.q_upper_limits[0])
        q1, dq1 = check_limits(q1, dq1, self.q_lower_limits[1],
                               self.q_upper_limits[1])

        return [q0, q1], [dq0, dq1]

    def reset(self, q=None, dq=None):
        """Resets the state of the arm
        q list: a list of the joint angles
        dq list: a list of the joint velocities
        """
        self.q = self.init_q if q is None else q
        self.dq = self.init_dq if dq is None else dq

    def Jcom1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q

        Jcom1 = np.zeros((6, 2))
        Jcom1[0, 0] = self.l[0] / 2. * -np.sin(q[0])
        Jcom1[1, 0] = self.l[0] / 2. * np.cos(q[0])
        Jcom1[5, 0] = 1.0

        return Jcom1

    def Jcom2(self, q=None):
        """Generates the Jacobian from the COM of the second
        link to the origin frame"""
        q = self.q if q is None else q

        Jcom2 = np.zeros((6, 2))
        # define column entries right to left
        Jcom2[0, 1] = self.l[1] / 2. * -np.sin(q[0]+q[1])
        Jcom2[1, 1] = self.l[1] / 2. * np.cos(q[0]+q[1])
        Jcom2[5, 1] = 1.0

        Jcom2[0, 0] = self.l[0] * -np.sin(q[0]) + Jcom2[0, 1]
        Jcom2[1, 0] = self.l[0] * np.cos(q[0]) + Jcom2[1, 1]
        Jcom2[5, 0] = 1.0

        return Jcom2

    def J(self, q=None):
        """Generates the Jacobian from end-effector to
        the origin frame"""
        q = self.q if q is None else q

        J = np.zeros((2, 2))
        # define column entries right to left
        J[0, 1] = self.l[1] * -np.sin(q[0]+q[1])
        J[1, 1] = self.l[1] * np.cos(q[0]+q[1])

        J[0, 0] = self.l[0] * -np.sin(q[0]) + J[0, 1]
        J[1, 0] = self.l[0] * np.cos(q[0]) + J[1, 1]

        return J

    def M(self, q=None):
        """Generates the mass matrix for the arm in joint space"""
        # get the instantaneous Jacobians
        Jcom1 = self.Jcom1(q=q)
        Jcom2 = self.Jcom2(q=q)
        # generate the mass matrix in joint space
        M = (np.dot(Jcom1.T, np.dot(self.M1, Jcom1)) +
             np.dot(Jcom2.T, np.dot(self.M2, Jcom2)))

        return M

    def Mx(self, q=None, **kwargs):
        """Generate the mass matrix in operational space"""
        q = self.q if q is None else q

        M = self.M(q=q, **kwargs)

        J = self.J(q=q)
        Mx_inv = np.dot(J, np.dot(np.linalg.inv(M), J.T))
        u, s, v = np.linalg.svd(Mx_inv)
        # cut off any singular values that could cause control problems
        for i in range(len(s)):
            s[i] = 0 if s[i] < self.singularity_thresh else 1./float(s[i])
        # numpy returns U,S,V.T, so have to transpose both here
        Mx = np.dot(v.T, np.dot(np.diag(s), u.T))

        return Mx

    def position(self, q=None, ee_only=False):
        """Compute x,y position of the hand

        q list: a list of the joint angles,
                if None use current system state
        ee_only boolean: if true only return the
                         position of the end-effector
        """
        q0, q1 = self.q if q is None else q

        x = np.cumsum([0,
                       self.l[0] * np.cos(q0),
                       self.l[1] * np.cos(q0+q1)])
        y = np.cumsum([0,
                       self.l[0] * np.sin(q0),
                       self.l[1] * np.sin(q0+q1)])

        if ee_only:
            return np.array([x[-1], y[-1]])

        return (x, y)

    def inv_kinematics(self, xy):
        """Calculate the joint angles for a given (x,y)
        hand position"""
        import scipy.optimize
        # function to optimize TODO: also return jacobian to minimize func
        def distance_to_target(q, xy, L):
            x = L[0] * np.cos(q[0]) + L[1] * np.cos(q[0] + q[1])
            y = L[0] * np.sin(q[0]) + L[1] * np.sin(q[0] + q[1])
            return np.sqrt((x - xy[0])**2 + (y - xy[1])**2)

        return scipy.optimize.minimize(
            fun=distance_to_target, x0=self.q,
            args=([xy[0], xy[1]], [self.l[0], self.l[1]]))['x']

    def create_nengo_node(self):

        def arm_func(t, x):
            x1 = 50
            y1 = 75
            scale = 30

            u = x[:2]
            target_x = x[2] * scale + x1
            target_y = x[3] * -scale + y1
            self.q, self.dq = self.apply_torque(u)
            # data returned from node to model
            data = np.hstack([self.q, self.dq, self.position(ee_only=True)])

            # visualization code ----------------------------------------------
            len0 = self.l[0] * scale
            len1 = self.l[1] * scale

            angles = data[:3]
            angle_offset = np.pi/2
            x2 = x1 + len0 * np.sin(angle_offset-angles[0])
            y2 = y1 - len0 * np.cos(angle_offset-angles[0])
            x3 = x2 + len1 * np.sin(angle_offset-angles[0] - angles[1])
            y3 = y2 - len1 * np.cos(angle_offset-angles[0] - angles[1])

            arm_func._nengo_html_ = '''
            <svg width="100%" height="100%" viewbox="0 0 100 100">
                <line x1="{x1}" y1="{y1}"
                 x2="{x2}" y2="{y2}" style="stroke:black"/>
                <line x1="{x2}" y1="{y2}"
                 x2="{x3}" y2="{y3}" style="stroke:black"/>
                <circle cx="{x3}" cy="{y3}"
                 r="1.5" stroke="black" stroke-width="1" fill="black" />
                <circle cx="{target_x}" cy="{target_y}"
                 r="1.5" stroke="red" stroke-width="1" fill="black" />
            </svg>
            '''.format(**locals())

            # TODO: add boxes around start area and end area

            # end of visualization code ---------------------------------------

            return data

        return nengo.Node(output=arm_func, size_in=self.DOF + 2, label='arm')
