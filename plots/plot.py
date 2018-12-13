from matplotlib import pyplot as plt
import numpy as np

eps_reward = np.genfromtxt('eps_reward2.csv', delimiter=',')
d_reward = np.genfromtxt('d_reward2.csv', delimiter=',')
sl_loss = np.genfromtxt('sl_loss2.csv', delimiter=',')


w = 10
h = 5


plt.figure(figsize=[w, h])
plt.plot(d_reward[:, 1], d_reward[:, 2])
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Avg. Disc. Reward', fontsize=20)
plt.grid()
plt.show()

plt.figure(figsize=[w, h])
plt.plot(sl_loss[:, 1], sl_loss[:, 2])
plt.xlabel('Steps', fontsize=20)
plt.ylabel('SL Loss', fontsize=20)
plt.xlim([0, 3500])
plt.grid()
plt.show()


window = 10
avg_mask = np.ones(window) / window
y_avg = np.convolve(eps_reward[:,2], avg_mask, 'same')

plt.figure(figsize=[w, h])
plt.plot(eps_reward[:, 1], y_avg)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Avg. Episode Reward', fontsize=20)
plt.grid()
plt.xlim([0, 3000])
plt.show()
