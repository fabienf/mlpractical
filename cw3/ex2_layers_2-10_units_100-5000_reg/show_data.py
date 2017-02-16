import os
import numpy as np
import matplotlib.pyplot as plt
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider, AugmentedCIFAR10DataProvider


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

def show_batch_with_augmentation(transformer=None):
    train_data = CIFAR10DataProvider('train-original', batch_size=25)
    inputs = train_data.next()
    X = inputs.reshape(inputs.shape[0], 3, 32, 32).astype("float")
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    count = 0
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            print X[count][0].shape
            axes1[j][k].imshow(X[count][0][0])
            count = count + 1

    fig.show()
    raw_input()

'''
data = load_data('cifar-10-{0}.npz'.format(which_set))

show_images(data)
'''

show_batch_with_augmentation()