import cPickle
import matplotlib.pyplot as plt
import numpy as np

tag_type = 'hidden_units_layers_2'


results = cPickle.load(open('results/'+tag_type+'.p', 'rb'))

for meta,res_data in results:
    print  meta
    print('valid_acc = {0:.2f}%'.format(float(res_data[max(res_data.iterkeys())]['valid_acc'])*100))

for res_data in results[1]:
    print res_data

fig_1 = plt.figure(figsize=(8, 4))
fig_1.suptitle('error(train)', fontsize=20)
ax_1 = fig_1.add_subplot(111)
for meta, res_data in results:
    x = [n for n in sorted(res_data)]
    y = [res_data[n]['train_err'] for n in sorted(res_data)]
    ax_1.plot(x, y, label=meta)
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
fig_1.savefig('figures/'+tag_type+'_error(train).pdf')

fig_1 = plt.figure(figsize=(8, 4))
fig_1.suptitle('error(valid)', fontsize=20)
ax_1 = fig_1.add_subplot(111)
for meta, res_data in results:
    x = [n for n in sorted(res_data)]
    y = [res_data[n]['valid_err'] for n in sorted(res_data)]
    ax_1.plot(x, y, label=meta)
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
fig_1.savefig('figures/'+tag_type+'_error(valid).pdf')

fig_1 = plt.figure(figsize=(8, 4))
fig_1.suptitle('accuracy(train)', fontsize=20)
ax_1 = fig_1.add_subplot(111)
for meta, res_data in results:
    x = [n for n in sorted(res_data)]
    y = [res_data[n]['train_acc'] for n in sorted(res_data)]
    ax_1.plot(x, y, label=meta)
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
fig_1.savefig('figures/'+tag_type+'accuracy(train).pdf')

fig_1 = plt.figure(figsize=(8, 4))
fig_1.suptitle('accuracy(valid)', fontsize=20)
ax_1 = fig_1.add_subplot(111)
for meta, res_data in results:
    x = [n for n in sorted(res_data)]
    y = [res_data[n]['valid_acc'] for n in sorted(res_data)]
    ax_1.plot(x, y, label=meta)
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
fig_1.savefig('figures/'+tag_type+'accuracy(valid).pdf')

#plt.show()