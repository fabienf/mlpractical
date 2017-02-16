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
		epoch_data = data['epoch_data']
		info = data['info']
	
		sorted_keys = sorted(map(int, epoch_data.keys()))
		y = [epoch_data[k][target] for k in sorted_keys]
		print info
		if (info['description']['label']=='random_bright'):
			info['description']['label']='random_contrast'
		ax_1.plot(sorted_keys, y, label=info['description']['label'])
	ax_1.legend(loc=0)
	ax_1.set_xlabel('Epoch number')
	fig.savefig(target+'.pdf')
	# plt.show()

plot(model_data,'acc(valid)')

