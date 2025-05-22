from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.max_state = 1e3
        self.states = []
        #macierz C mówi nam co chcemy obserwować
        #W to macierz C
    def set_B(self, B):
        self.B = np.array(B)

    def update(self, q, u):
        self.curr_state = self.state.copy()  # current estimated state
        y_hat = np.dot(self.W, self.curr_state)  # observer output
        u = np.array(u).reshape(-1, 1)  # force column vector
        error = np.array(q - y_hat).reshape(-1, 1)
        dx = self.A @ self.curr_state.reshape(-1, 1) + self.B @ u + self.L @ error
        self.state = (self.curr_state.reshape(-1, 1) + self.Tp * dx).flatten()
        self.state = np.clip(self.state, -self.max_state, self.max_state)



        self.states.append(copy(self.state))

    def get_state(self):
        return self.state.flatten()
