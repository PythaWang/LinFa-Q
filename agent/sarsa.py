import numpy as np
import warnings

np.random.seed(1)

class SARSA:

    def __init__(self, n_action, tile_coder, alpha=0.1, gamma=1, epsilon=0.1):
        self.tile_coder = tile_coder
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.epsilon = epsilon

        self.w_0 = np.ones(self.tile_coder.n_feature) * -10
        self.w = self.w_0

    def choose_action(self, s):
        qs = self.tile_coder.q_for_all_action(s, self.w)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_action)
        else:
            a = np.argmax(qs)
        return a

    def learn(self, s, a, r, s_, a_, done):
        fa = self.tile_coder.feature_for_given_action(s, a)
        u = r + (1 - done) * self.gamma * self.tile_coder.q_for_given_action(s_, a_, self.w)
        td_error = u - self.tile_coder.q_for_given_action(s, a, self.w)
        with warnings.catch_warnings(record=True) as w:
            self.w += self.alpha * td_error * fa
        if len(w) > 0:
            print(111)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


