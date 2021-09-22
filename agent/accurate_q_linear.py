import time
import numpy as np
# import copy
# from functools import wraps
from line_profiler import LineProfiler


# def func_line_time(f):
#     @wraps(f)
#     def decorator(*args, **kwargs):
#         func_return = f(*args, **kwargs)
#         lp = LineProfiler()
#         lp_wrap = lp(f)
#         lp_wrap(*args, **kwargs)
#         lp.print_stats()
#         return func_return
#     return decorator

class AccurateQLinear:

    def __init__(self, env, tile_coder, alpha=0.01,
                 accurate=True, horizon=99999999999999999,
                 gamma=1, epsilon=1, decay_replace=False, trial=1):
        self.n_action = env.action_space.n
        self.tile_coder = tile_coder
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.epsilon = epsilon

        # accurate on or off
        self.accurate = accurate

        # memory replay parameter
        self.buffer = []
        self.buffer_size = 1
        self.bath_size = 1

        self.w_0 = np.ones(tile_coder.n_feature) * 0  # initialize w with the same value
        self.w = self.w_0

        # some variates for updating w
        self.term_w = None  # value of the term with the initial parameter w_0 after each update
        self.term_r = None  # value of the term with rewards R after each update
        self.term_m = None  # value of the term with max_q(f_) m after each update
        self.M = None  # array [m_0, m_1, ...], store values of different terms with m_i = max_q(f_i)
        self.f_list = []  # record, store the feature of the next state at each step (only for the unseen state)
        self.fa_index_M = {}
        self.m_list = []  # record, store max_q(f_) after each update
        self.replace_frequency = 1
        self.decay_replace = decay_replace
        self.horizon = horizon
        self.memory = 0

        np.random.seed(trial)

    # @func_line_time
    def learn(self, s, a, r, s_, done):
        # w_{t+1} = E(w_0) + E(R) + E(M)

        # 0. prepare
        # construct the feature of the state-action pair
        fa = self.tile_coder.feature_for_given_action(s, a)
        # get the maximum value of the next state
        qs_ = self.tile_coder.q_for_all_action(s_, self.w)
        q_ = max(qs_)

        # 1. compute the term with w_0
        if self.term_w is None:
            self.term_w = self.w
        self.term_w = self.term_w - self.alpha * np.dot(fa, self.term_w) * fa

        # 2. compute the term with R
        if self.term_r is None:
            self.term_r = self.alpha * r * fa
        else:
            self.term_r = self.term_r - self.alpha * np.dot(fa, self.term_r) * fa + self.alpha * r * fa

        # 3. compute the term with M
        fa_index = tuple(self.tile_coder.get_feature(s_, 0, True))

        # For clarifying that the new forms of update is equivalent to vanilla q-learning,
        # here we provide two update processes with respect to two forms of update, respectively.
        # Uncomment the process (1) or (2), and comment the part of accurate update (the process (3)),
        # we will achieve the same result as vanilla q-learning.

        # (1) update with the value of the term with max_q(f_) [compute term_m directly]
        # ------------------ process (1) ------------------
        # if self.term_m is None:
        #     self.term_m = self.alpha * self.gamma * q_ * fa
        # else:
        #     # update term_m: E_{t+1}(M) = E_{t}(M) - alpha * fa * E_{t}(M) * fa + alpha * gamma * m_t * fa
        #     self.term_m = self.term_m - self.alpha * np.dot(fa, self.term_m) * fa + self.alpha * self.gamma * q_ * fa
        # ------------------ process (1) ------------------

        # (2) update with each value of terms with m_i = max_q(f_i) [compute M and then compute term_m]
        # ------------------ process (2) ------------------
        # if self.M is None:
        #     self.M = self.alpha * self.gamma * q_ * fa
        #     self.M = np.expand_dims(self.M, axis=0)
        #     self.f_list.append(fa_index)
        # else:
        #     index_f_ = -1
        #     for i in range(0, len(self.f_list)):
        #         if self.f_list[i] == fa_index:
        #             index_f_ = i
        #             break
        #
        #     fam = np.matmul(fa, self.M.T)  # (n_feature,) * (n, n_feature).T = (n,) (M stores n terms)
        #     fam = np.expand_dims(fam, axis=1)  # (n,1)
        #     fa_ex = np.expand_dims(fa, axis=0)  # (1, n_feature)
        #     self.M = self.M - self.alpha * np.matmul(fam, fa_ex)
        #
        #     if index_f_ == -1:
        #         self.M = np.vstack((self.M, self.alpha * self.gamma * q_ * fa))
        #         self.f_list.append(fa_index)
        #     else:
        #         self.M[index_f_] += self.alpha * self.gamma * q_ * fa
        # self.term_m = self.M.sum(axis=0)  # (n_feature,) <- (n, n_feature)
        # ------------------ process (2) ------------------

        # (3) the part of accurate update
        # ------------------ process (3) ------------------
        if self.M is None:
            # store the new term with the new m, f_, and append the relevant max_q(f_)
            self.M = self.alpha * self.gamma * q_ * fa
            self.M = np.expand_dims(self.M, axis=0)
            self.term_m = self.alpha * self.gamma * q_ * fa  # do not ues (term_m = M) to avoid the probable shallow copy
            self.f_list.append(fa_index)
            self.fa_index_M[fa_index] = len(self.M) - 1
            self.m_list.append(q_)
        else:
            # Before replacing the value of m_i = max_q(f_i) (refer to q_old)
            # and updating the value of the term with max_q(f_) (refer to term_m),
            # update values of different terms with m_i = max_q(f_i) (refer to M).
            fam = np.matmul(fa, self.M.T)  # shape: (n_feature,) * (n, n_feature).T = (n,) (M stores n terms)
            fam = np.expand_dims(fam, axis=1)  # (n,1)
            fa_ex = np.expand_dims(fa, axis=0)  # (1, n_feature)
            self.M = self.M - self.alpha * np.matmul(fam, fa_ex)

            if not done:  # if done, skip the replacement process
                # If f_ has been in the trajectory, which means the part with respect to f_ of w has updated,
                # we need update the max_q(f_).
                # index_f_ = -1
                # for i in range(0, len(self.f_list)):
                #     if self.f_list[i] == fa_index:
                #         index_f_ = i
                #         break

                if fa_index not in self.fa_index_M.keys():
                    # store the new term with the new m, f_, and append the relevant max_q(f_)
                    self.M = np.vstack((self.M, self.alpha * self.gamma * q_ * fa))
                    self.f_list.append(fa_index)
                    self.fa_index_M[fa_index] = len(self.M) - 1
                    self.m_list.append(q_)
                else:
                    index_f_ = self.fa_index_M[fa_index]
                    # Replace the value of m_i = max_q(f_i) and the value of the term with m_i,
                    # and update M: M_{t+1} = M_{t} - alpha * fa * M_{t} * fa
                    q_old = self.m_list[index_f_]
                    self.m_list[index_f_] = q_
                    if self.accurate:
                        if np.random.uniform() < self.replace_frequency:
                            if q_old == 0:
                                self.M[index_f_] = self.M[index_f_] * q_
                            else:
                                self.M[index_f_] = self.M[index_f_] * q_ / q_old
                            if self.decay_replace:
                                self.replace_frequency = max(self.replace_frequency*0.9999, 0.3)
                    # add the new term into M, where they share the same max_q(f_)
                    self.M[index_f_] += self.alpha * self.gamma * q_ * fa

            # compute the term with
            self.term_m = self.M.sum(axis=0)  # (n_feature,) <- (n, n_feature)
        # ------------------ process (3) ------------------

        # 4. return updated w
        return self.term_w + self.term_r + self.term_m

    def store(self, s, a, r, s_, done):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([s, a, r, s_, done])

        if len(self.buffer) >= self.bath_size:
            # learn
            batch = np.random.randint(len(self.buffer), size=self.bath_size)
            w = np.zeros(shape=self.w.shape)
            for i in batch:
                w += self.learn(self.buffer[i][0], self.buffer[i][1], self.buffer[i][2], self.buffer[i][3], self.buffer[i][4])
            self.w = w / self.bath_size

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if self.accurate and done:
                self.memory += 1
                if self.memory == self.horizon:
                    self.clear()
                    self.memory = 0

    def clear(self):
        self.term_w = None
        self.term_r = None
        self.term_m = None
        self.M = None
        self.f_list = []
        self.m_list = []
        self.fa_index_M = {}

    def choose_action(self, s):
        qs = self.tile_coder.q_for_all_action(s, self.w)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_action)
        else:
            a = np.argmax(qs)
        return a
