import os
import numpy as np
import tensorflow as tf
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



def random_flip(inputs, rng):

    orig_ims = inputs.reshape((-1, 32, 32, 3))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    #angles = rng.uniform(-1., 1., size=indices.shape[0]) * 1.
    for idx, item in enumerate(indices):
        new_ims[item] = tf.image.random_flip_left_right(image, seed=None)
    return new_ims.reshape((-1, 3072))



def show_batch_with_augmentation(rng=None):
    train_data = CIFAR10DataProvider('train-original', batch_size=25)
    inputs = train_data.next()[0]
    print inputs.shape
    X = inputs.reshape(inputs.shape[0], 3, 32, 32)
    X = X.transpose(0,2,3,1).astype("float")

    print X.shape
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    count = 0
    for j in range(5):
        for k in range(5):
            axes1[j][k].set_axis_off()
            print X[count].shape
            axes1[j][k].imshow(X[count:count+1][0])
            count = count + 1


    aug_data = AugmentedCIFAR10DataProvider('train-original', batch_size=25, transformer=random_flip, rng=rng )

    inputs = train_data.next()[0]
    print inputs.shape
    X = inputs.reshape(inputs.shape[0], 3, 32, 32)
    X = X.transpose(0,2,3,1).astype("float")

    axes2 = fig.add_subplot(111)
    count = 0
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes2[j][k].set_axis_off()
            print X[count].shape
            axes2[j][k].imshow(X[count:count+1][0])
            count = count + 1




    fig.show()
    raw_input()




'''
data = load_data('cifar-10-{0}.npz'.format(which_set))

show_images(data)
'''

seed = 7474747 
rng = np.random.RandomState(seed)

show_batch_with_augmentation()