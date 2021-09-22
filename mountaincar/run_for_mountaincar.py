import sys
sys.path.append('/usr/wzc/test_for_client/code/Accurate_Q_Liner_Function_Approximation/')
import gym
import numpy as np
from agent.accurate_q_linear import AccurateQLinear
from helper.sutton_tile_coder import TileCoder


random_reward = False
decay_replace = False
pro = 'alpha_decay_gamma_1_epsilonMin_0.01'

for trial in range(1, 5):
    np.random.seed(trial)
    env = gym.make('MountainCar-v0')
    env.seed(trial)
    agent1 = AccurateQLinear(env=env, tile_coder=TileCoder(env, n_layer=5, n_feature=2048),
                             alpha=0.01,
                             gamma=1,
                             epsilon=0.1,
                             accurate=False)
    agent2 = AccurateQLinear(env=env, tile_coder=TileCoder(env, n_layer=5, n_feature=2048),
                             alpha=0.01,
                             gamma=1,
                             epsilon=0.1,
                             accurate=True,
                             decay_replace=decay_replace,
                             horizon=2000,
                             trial=trial)
    # agents = [('vanilla', agent1), ('accurate', agent2)]
    agents = [('accurate', agent2)]
    # agents = [('vanilla', agent1)]

    # s = env.reset()
    # f = agent2.tile_coder.feature_for_all_action(s)
    # np.save('f_focus_4096', f)

    for k, agent in agents:
        MAX_episode = 2000
        MAX_step = 200
        step_record = []
        w_record = []
        r_record = []
        fa_record = {}

        for i in range(MAX_episode):
            s = env.reset()

            step = 0
            r_sum = 0

            while True:
                step += 1
                # env.render()
                a = agent.choose_action(s)
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
                    done = True

                agent.alpha = 1/(100+fa_record[str(fa)])
                agent.store(s, a, r, s_, done)
                s = s_
                r_sum += r

                if done:
                    print('E', i, step, agent.epsilon, agent.alpha, len(agent.f_list))
                    step_record.append(step)
                    w_record.append(agent.w)
                    r_record.append(r_sum)
                    break
                if i % 100 == 0:
                    np.savez('layer_5/{}_random_reward^{}_decay_replace^{}_trial_{}_'
                             .format(k, int(random_reward), int(decay_replace), trial) + pro,
                             step=step_record, w=w_record, r=r_record)
        np.savez('layer_5/{}_random_reward^{}_decay_replace^{}_trial_{}_'
                 .format(k, int(random_reward), int(decay_replace), trial) + pro,
                 step=step_record, w=w_record, r=r_record)
        print(agent.tile_coder.iht)