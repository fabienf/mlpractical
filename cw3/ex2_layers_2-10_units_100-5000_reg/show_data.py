import os
import numpy as np
import matplotlib.pyplot as plt

which_set = 'valid'


def load_data(file_name):
    data_path = os.path.join(os.environ['MLP_DATA_DIR'], file_name)
    loaded = np.load(data_path)
    inputs, targets, label_map = loaded['inputs'], loaded['targets'], loaded['label_map']

    inputs = inputs.reshape((inputs.shape[0], -1)).astype('float32')
    
    return inputs, targets, label_map

def show_images(data):
    inputs, targets, label_map = data

    X = inputs
    X = X.reshape(X.shape[0], 1, 32, 32).astype("float")

    #Visualizing CIFAR 10
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            print X[i:i+1][0].shape
            axes1[j][k].imshow(X[i:i+1][0][0], cmap='gray')
    fig.show()
    raw_input()


'''
data_path = os.path.join(
    os.environ['MLP_DATA_DIR'], 'cifar-10-{0}.npz'.format(which_set))
assert os.path.isfile(data_path), (
    'Data file does not exist at expected path: ' + data_path
)
'''

data = load_data('cifar-10-{0}.npz'.format(which_set))

show_images(data)