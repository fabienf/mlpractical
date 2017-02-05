import cPickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

tag_type = 'Ada'


results = cPickle.load(open(tag_type+'.p', 'rb'))

print results

for run in results:
    #print '   rate = '+str(run[0])+ ' acc(valid)=' + str(run[1][-1, run[2]['acc(valid)']])
    print '\hline'
    print str(run[0][0])+ ' & ' + str(run[0][1])+ ' & ' + str(run[1][-1, run[2]['acc(valid)']]) + ' \\\\'
color=iter(cm.rainbow(np.linspace(0,1,len(results))))

fig_1 = plt.figure(figsize=(16, 8))
fig_1.suptitle('error(train)', fontsize=20)
ax_1 = fig_1.add_subplot(111)
for run in results:
    ax_1.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['error(train)']], label='rate='+str(run[0][0])+' eps='+str(run[0][1]),c=next(color))
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
fig_1.savefig('figures/'+tag_type+'_error(train).pdf')

color=iter(cm.rainbow(np.linspace(0,1,len(results))))
fig_2 = plt.figure(figsize=(16, 8))
fig_2.suptitle('error(valid)', fontsize=20)
ax_2 = fig_2.add_subplot(111)
for run in results:
    ax_2.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['error(valid)']], label='rate='+str(run[0][0])+' eps='+str(run[0][1]),c=next(color))
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
fig_2.savefig('figures/'+tag_type+'_error(valid).pdf')

color=iter(cm.rainbow(np.linspace(0,1,len(results))))
fig_3 = plt.figure(figsize=(16, 8))
fig_3.suptitle('acc(train)', fontsize=20)
ax_3 = fig_3.add_subplot(111)
for run in results:
    ax_3.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['acc(train)']], label='rate='+str(run[0][0])+' eps='+str(run[0][1]),c=next(color))
ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
fig_3.savefig('figures/'+tag_type+'_acc(train).pdf')

color=iter(cm.rainbow(np.linspace(0,1,len(results))))
fig_4 = plt.figure(figsize=(16, 8))
fig_4.suptitle('acc(valid)', fontsize=20)
ax_4 = fig_4.add_subplot(111)
for run in results:
    ax_4.plot(np.arange(1, run[1].shape[0]) * 5,
              run[1][1:, run[2]['acc(valid)']], label='rate='+str(run[0][0])+' eps='+str(run[0][1]),c=next(color))
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
fig_4.savefig('figures/'+tag_type+'_acc(valid).pdf')

#plt.show()
