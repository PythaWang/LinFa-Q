from helper.tile_coding import *
import numpy as np

class TileCoder:

    def __init__(self, env, n_layer=2, n_feature=42):
        self.env = env
        self.n_layer = n_layer
        self.n_feature = n_feature
        self.iht = IHT(n_feature)
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.scale = self.high - self.low

    def get_feature(self, state, action, need_code):
        hash_table = self.iht
        num_tilings = self.n_layer
        measure_for_each_dim = num_tilings / self.scale

        # position_scale = num_tilings / (max_position - min_position)
        # velocity_scale = num_tilings / (max_velocity - min_velocity)
        # position, velocity = state

        indexs = tiles(
            hash_table,
            num_tilings,
            measure_for_each_dim * state,
            [action]
        )
        if need_code:
            return indexs
        feature = np.zeros(self.n_feature)
        for index in indexs:
            feature[index] = 1
        return feature


    def feature_for_all_action(self, s):
        fas = []
        for i in range(self.env.action_space.n):
            fas.append(self.feature_for_given_action(s, i))
        return fas


    def feature_for_given_action(self, s, a):
        return self.get_feature(s, a, False)



    def q_for_given_action(self, s, a, w):
        fa = self.feature_for_given_action(s, a)
        q = np.dot(fa, w)
        return q


    def q_for_all_action(self, s, w):
        fas = self.feature_for_all_action(s)
        return [np.dot(fa, w) for fa in fas]


if __name__ == '__main__':
    import gym
    env = gym.make('MountainCar-v0')
    tile = TileCoder(env=env, n_layer=10)
    print(tile.get_feature([0.03750000149011612, 0.077], 0))
    print(tile.get_feature([0.0376, 0.078], 0))
    print(tile.get_feature([0.03750000149011612, 0.077], 1))
    print(tile.get_feature([0.03750000149011612, 0.077], 2))