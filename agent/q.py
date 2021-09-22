import numpy as np


class Q:

    def __init__(self, n_action, tile_coder, gamma=0.99, alpha=0.1, epsilon=1, trial=1):
        self.tile_coder = tile_coder
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.epsilon = epsilon

        self.buffer = []
        self.buffer_size = 1000
        self.bath_size = 32

        self.w = np.ones(self.tile_coder.n_feature) * 0
        self.fa_record = {}

        np.random.seed(trial)

    def choose_action(self, s):
        qs = self.tile_coder.q_for_all_action(s, self.w)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_action)
        else:
            a = np.argmax(qs)
        return a

    def learn(self, s, a, r, s_, done):
        fa = self.tile_coder.feature_for_given_action(s, a)
        u = r + (1 - done) * self.gamma * max(self.tile_coder.q_for_all_action(s_, self.w))
        td_error = u - self.tile_coder.q_for_given_action(s, a, self.w)
        w = self.w + self.alpha * td_error * fa

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return w

    def store(self, s, a, r, s_, done):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([s, a, r, s_, done])

        if len(self.buffer) >= self.bath_size:
            # learn
            batch = np.random.randint(len(self.buffer), size=self.bath_size)
            w = np.zeros(shape=self.w.shape)
            for i in batch:
                w += self.learn(self.buffer[i][0], self.buffer[i][1], self.buffer[i][2], self.buffer[i][3],
                                self.buffer[i][4])
            self.w = w / self.bath_size

