import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import os








def get_files_in_dir():
	files = [f for f in glob.glob(os.path.join(os.getcwd(), '*.pickle'))]
	return files


files = get_files_in_dir()
model_data = []
for file_path in files:
	with open(file_path, 'rb') as f:
		data = pickle.load(f)
		model_data.append(data)
print model_data

def plot(model_data, target):
	fig = plt.figure(1)
	fig.suptitle(target, fontsize=15)
	ax_1 = fig.add_subplot(111)
	
	for idx,data in enumerate(model_data):
		if (idx==0 or idx==5 or idx==7):
			epoch_data = data['epoch_data']
			info = data['info']
		
			sorted_keys = sorted(map(int, epoch_data.keys()))
			y = [epoch_data[k][target] for k in sorted_keys]
			print info
			ax_1.plot(sorted_keys, y, label=info['description']['label'])
	ax_1.legend(loc=0)
	ax_1.set_xlabel('Epoch number')
	fig.savefig(target+'.pdf')
	# plt.show()

plot(model_data,'acc(valid)')










# tag_type = 'hidden_units_layers_2_reg'


# results = cPickle.load(open('results/'+tag_type+'.p', 'rb'))

# for meta,res_data in results:
#     print  meta
#     print('valid_acc = {0:.2f}%'.format(float(res_data[max(res_data.iterkeys())]['valid_acc'])*100))

# for res_data in results[1]:
#     print res_data

# fig_1 = plt.figure(figsize=(8, 4))
# fig_1.suptitle('error(train)', fontsize=20)
# ax_1 = fig_1.add_subplot(111)
# for meta, res_data in results:
#     x = [n for n in sorted(res_data)]
#     y = [res_data[n]['train_err'] for n in sorted(res_data)]
#     ax_1.plot(x, y, label=meta)
# ax_1.legend(loc=0)
# ax_1.set_xlabel('Epoch number')
# fig_1.savefig('figures/'+tag_type+'_error(train).pdf')

# fig_1 = plt.figure(figsize=(8, 4))
# fig_1.suptitle('error(valid)', fontsize=20)
# ax_1 = fig_1.add_subplot(111)
# for meta, res_data in results:
#     x = [n for n in sorted(res_data)]
#     y = [res_data[n]['valid_err'] for n in sorted(res_data)]
#     ax_1.plot(x, y, label=meta)
# ax_1.legend(loc=0)
# ax_1.set_xlabel('Epoch number')
# fig_1.savefig('figures/'+tag_type+'_error(valid).pdf')

# fig_1 = plt.figure(figsize=(8, 4))
# fig_1.suptitle('accuracy(train)', fontsize=20)
# ax_1 = fig_1.add_subplot(111)
# for meta, res_data in results:
#     x = [n for n in sorted(res_data)]
#     y = [res_data[n]['train_acc'] for n in sorted(res_data)]
#     ax_1.plot(x, y, label=meta)
# ax_1.legend(loc=0)
# ax_1.set_xlabel('Epoch number')
# fig_1.savefig('figures/'+tag_type+'_accuracy(train).pdf')

# fig_1 = plt.figure(figsize=(8, 4))
# fig_1.suptitle('accuracy(valid)', fontsize=20)
# ax_1 = fig_1.add_subplot(111)
# for meta, res_data in results:
#     x = [n for n in sorted(res_data)]
#     y = [res_data[n]['valid_acc'] for n in sorted(res_data)]
#     ax_1.plot(x, y, label=meta)
# ax_1.legend(loc=0)
# ax_1.set_xlabel('Epoch number')
# fig_1.savefig('figures/'+tag_type+'_accuracy(valid).pdf')

# #plt.show()