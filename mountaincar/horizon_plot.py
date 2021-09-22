import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

len_plot = 2000
num_trial = 1
n_feature = 2048
f_focus = np.load('f_focus_{}.npy'.format(n_feature))

fontsize = 15

horizons = [50, 100, 200, 500, 2000]
steps = []
ws = []

for h in horizons:
    acc_steps = np.zeros(len_plot)
    acc_w = np.zeros((len_plot, n_feature))
    for trial in range(0, num_trial):
        acc = np.load('layer_5/accurate_random_reward^1_decay_replace^0_trial_'
                      '{}_alpha_decay_gamma_1_horizon_{}_epsilonMin_0.01.npz'.format(trial, h))
        acc_steps += acc['step']
        acc_w += acc['w']
    acc_steps = acc_steps / num_trial
    acc_w = acc_w / num_trial
    steps.append(acc_steps)
    ws.append(acc_w)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(121)
fig.suptitle('Mountain Car', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plot step
plt.xlabel('Episode', fontsize=fontsize)
plt.ylabel('Steps', fontsize=fontsize)
for i, step in enumerate(steps):
    # ax1.plot(steps[algorithm], label='accurate')
    # ax1.plot(range(0, len(steps[algorithm]), 10),
    #          [np.mean(steps[algorithm][i: i + 10]) for i in range(0, len(steps[algorithm]), 10)])
    ax1.plot([np.mean(step[0: i]) for i in range(1, len(step))], label='{}'.format(horizons[i]))
# plt.legend(fontsize=fontsize)
plt.grid()
# plt.show()

# plot q values for features we focus
ax2 = fig.add_subplot(122)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Episode', fontsize=fontsize)
plt.ylabel('Tnitial maximum Q value', fontsize=fontsize)
for i, w in enumerate(ws):
    q = [max(np.dot(f_focus, e)) for e in w]
    # ax2.plot(q, label='{}'.format(horizons[i]))
    ax2.plot([np.mean(q[0: i]) for i in range(1, len(q))], label='{}'.format(horizons[i]))
ax2.legend(fontsize=fontsize)
plt.grid()
plt.savefig('results/horizon_M')
plt.show()




