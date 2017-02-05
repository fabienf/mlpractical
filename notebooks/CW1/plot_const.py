import cPickle
import matplotlib.pyplot as plt
import numpy as np

tag_type = 'Const_rule'


results = cPickle.load(open(tag_type+'.p', 'rb'))

print results

for run in results:
    print '   rate = '+str(run[0])+ ' acc(valid )=' + str(run[1][-1, run[2]['error(train)']])


fig_1 = plt.figure(figsize=(8, 4))
fig_1.suptitle('error(train)', fontsize=20)
ax_1 = fig_1.add_subplot(111)
for run in results:
    ax_1.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['error(train)']], label='rate = '+str(run[0]))
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')

fig_2 = plt.figure(figsize=(8, 4))
fig_2.suptitle('error(valid)', fontsize=20)
ax_2 = fig_2.add_subplot(111)
for run in results:
    ax_2.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['error(valid)']], label='rate = '+str(run[0]))
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')

fig_3 = plt.figure(figsize=(8, 4))
fig_3.suptitle('acc(train)', fontsize=20)
ax_3 = fig_3.add_subplot(111)
for run in results:
    ax_3.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['acc(train)']], label='rate = '+str(run[0]))
ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')

fig_4 = plt.figure(figsize=(8, 4))
fig_4.suptitle('acc(valid)', fontsize=20)
ax_4 = fig_4.add_subplot(111)
for run in results:
    ax_4.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['acc(valid)']], label='rate = '+str(run[0]))
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')



plt.show()
