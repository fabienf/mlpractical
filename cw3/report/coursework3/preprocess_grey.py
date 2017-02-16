import os
import numpy as np

which_set = 'train'

data_path = os.path.join(
    os.environ['MLP_DATA_DIR'], 'cifar-10-{0}.npz'.format(which_set+'-original'))
assert os.path.isfile(data_path), (
    'Data file does not exist at expected path: ' + data_path
)
# load data from compressed numpy file
loaded = np.load(data_path)
inputs, targets = loaded['inputs'], loaded['targets']
label_map = loaded['label_map']


grey_array_inputs = [None]*len(inputs)
for idx, image in enumerate(inputs):
	grey_image = np.empty(shape=(1024))
	for x in range(1024):
		grey_image[x] = (image[x]+image[x+1024]+image[x+2048])/3.0
	grey_array_inputs[idx] = grey_image
	if (idx%1000==0):
		print idx


data_path_to_save = os.path.join(
    os.environ['MLP_DATA_DIR'], 'cifar-10-{0}.npz'.format(which_set))
np.savez_compressed(data_path_to_save, inputs=grey_array_inputs, targets=targets, label_map=label_map)

print len(grey_array_inputs)
print len(grey_array_inputs[0])

print len(inputs)
print len(inputs[0])
print len(targets)
print targets[0]

