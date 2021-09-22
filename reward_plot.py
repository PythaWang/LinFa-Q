import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
fontsize = 15
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
linestyles = ['-', '-.', ':', '--', '-']
markers = ['o', '>', '+', 'D']


als = ['accurate', 'q']
al_legend = ['linFa-Q', 'Q']
len_plot = 2000
num_trial = 5
n_feature = 4096
env = 'cartpole'
title = 'Cart Pole'
f_focus = np.load(env + '/f_focus_{}.npy'.format(n_feature))

rewards = [0, 1]
legends = ['', ' (RR)']
steps = [[],[]]
ws = [[],[]]

for r in rewards:
    acc_steps = np.zeros(len_plot)
    acc_ws = np.zeros((len_plot, n_feature))
    ql_steps = np.zeros(len_plot)
    ql_ws = np.zeros((len_plot, n_feature))
    for trial in range(0, num_trial):
        acc = np.load(env + '/layer_8/accurate_random_reward^{}_decay_replace^0_trial_'
                      '{}_alpha_decay_gamma_1_epsilonMin_0.01.npz'.format(r, trial))
        ql = np.load(env + '/layer_8/q_trial_{}_{}alpha_decay_gamma_1_epsilonMin_0.01.npz'
                     .format(trial, 'random_reward_' if r else ''))
        acc_steps += acc['step'][:len_plot]
        acc_ws += acc['w'][:len_plot]
        ql_steps += ql['steps'][:len_plot]
        ql_ws += ql['ws'][:len_plot]
    steps[0].append(acc_steps/num_trial)
    ws[0].append(acc_ws/num_trial)
    steps[1].append(ql_steps / num_trial)
    ws[1].append(ql_ws / num_trial)


fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(121)
fig.suptitle(title, fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plot step
plt.xlabel('Episode', fontsize=fontsize)
plt.ylabel('Steps', fontsize=fontsize)
for k, al in enumerate(als):
    for i, step in enumerate(steps[k]):
        # ax1.plot(steps[algorithm], label='accurate')
        ax1.plot(range(0, len(step), 25),
                 [np.mean(step[i: i + 25]) for i in range(0, len(step), 25)], color=colors[k],
                 linestyle=linestyles[i], marker=markers[i], markevery=10, label='{}{}'.format(al_legend[k], legends[i]))
        # ax1.plot([np.mean(step[0: i]) for i in range(1, len(step))], color=colors[k], linestyle=linestyles[i],
        #          marker=markers[i], markevery=250, label='{}_{}'.format(al_legend[k], legends[i]))
# plt.legend(fontsize=fontsize)
plt.grid()
# plt.show()

# plot q values for features we focus
ax2 = fig.add_subplot(122)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Episode', fontsize=fontsize)
plt.ylabel('Initial maximum Q value', fontsize=fontsize)
for k, al in enumerate(als):
    for i, w in enumerate(ws[k]):
        q = [max(np.dot(f_focus, e)) for e in w]
        ax2.plot(range(0, len(q), 25),
                [np.mean(q[i: i + 25]) for i in range(0, len(q), 25)],
                 color=colors[k], linestyle=linestyles[i], marker=markers[i], markevery=10,
                 label='{}{}'.format(al_legend[k], legends[i]))
        # ax2.plot(q, color=colors[k], linestyle=linestyles[i], marker=markers[i], markevery=250,
        #          label='{}_{}'.format(al, legends[i]))
ax2.legend(fontsize=fontsize)
# plt.subplots_adjust(right=0.8)
plt.grid()
plt.savefig(env+'/results/reward')
plt.show()




