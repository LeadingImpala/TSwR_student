import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.1
        self.r3 = 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.d1 = self.l1/2
        self.d2 = self.l2/2

        self.alpha = self.m1 * self.d1**2 + self.I_1 + self.m2*(self.l1**2+self.d2**2) +self.I_2 + self.m3*(self.l1**2 + self.l2**2) +self.I_3
        self.beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 *self.l2
        self.gamma = self.m2*self.d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3
        
    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        M = np.array([
            [self.alpha+2*self.beta*np.cos(q2), self.gamma+self.beta*np.cos(q2)],
            [self.gamma+self.beta*np.cos(q2), self.gamma]
            ])
        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        C = np.array([
            [-1*self.beta*np.sin(q2)*q2_dot, -1*self.beta*np.sin(q2)*(q1_dot+q2_dot)],
            [self.beta*np.sin(q2)*q1_dot, 0]
            ])
        return C
    
    def predict(self, x):
        """
        Predicts the next state x_mi given the current state x using the model dynamics.
        """
        M = self.M(x)  # Mass matrix
        C = self.C(x)  # Coriolis matrix
        
        q_dot = x[2:]  # Extract velocity (q1_dot, q2_dot)
        
        # Compute acceleration (assuming no external forces for prediction)
        q_ddot = np.linalg.inv(M) @ (-C @ q_dot[:, np.newaxis])
        
        # Predict next state using Euler integration
        x_pred = x + np.hstack((q_dot, q_ddot.flatten())) * self.Tp
        
        return x_pred