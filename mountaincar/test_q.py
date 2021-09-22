import gym
from helper.sutton_tile_coder import TileCoder
import numpy as np
import copy
from agent.q import Q

for trial in range(0, 1):
    np.random.seed(trial)
    random_reward = True
    pro = 'random_reward_alpha_decay_gamma_1_epsilonMin_0.01_b'

    env = gym.make('MountainCar-v0')
    env.seed(trial)
    agent = Q(n_action=env.action_space.n, tile_coder=TileCoder(env, n_layer=5, n_feature=2048),
              alpha=0.01,
              gamma=1,
              epsilon=0.1,
              trial=trial)

    ws = []
    steps = []
    rs = []
    fa_record = {}

    MAX_episode = 2000
    MAX_step = 200
    for i in range(MAX_episode):
        r_sum = 0
        s = env.reset()

        step = 0
        while True:
            #     env.render()
            a = agent.choose_action(s)
            step += 1
            s_, r, done, _ = env.step(a)

            fa = agent.tile_coder.get_feature(s, a, True)
            if str(fa) in fa_record.keys():
                fa_record[str(fa)] += 1
            else:
                fa_record[str(fa)] = 1

            if random_reward:
                if np.random.uniform() < 0.5:
                    r = -8
                else:
                    r = 6
            if step == MAX_step:
                # r = 5
                done = True

            agent.alpha = 1 / (200 + fa_record[str(fa)])
            agent.store(s, a, r, s_, done)
            s = s_
            r_sum += r

            if done:
                ws.append(copy.deepcopy(agent.w))
                steps.append(step)
                rs.append(r_sum)
                print(i, step, agent.alpha, agent.epsilon)
                break
    np.savez('layer_5/q_trial_{}_{}'.format(trial, pro), ws=ws, steps=steps, rs=rs)
