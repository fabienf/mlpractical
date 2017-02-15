import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider, AugmentedCIFAR10DataProvider
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import cPickle




def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs, weights



def transform_grey(inputs):
    new_inputs = []
    for image in inputs:
        grey = np.zeros(shape=(1024))
        for x in range(1024):
            grey[x] = (image[x]+image[x+1024]+image[x+2048])/3.0
        new_inputs.append(grey)
    return new_inputs


def runNetwork(meta=None):

    train_data = CIFAR10DataProvider('train', batch_size=50)
    valid_data = CIFAR10DataProvider('valid', batch_size=50)

    #The results to be returned
    epoch_res_dict = {}
    if meta==None:
        meta = {
            'num_hidden': 200,
            'num_epochs': 10,
            'layers': 2
        }

    inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
    targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')

    with tf.name_scope('fc-layer-1'):
        hidden_1, weights_1 = fully_connected_layer(inputs, train_data.inputs.shape[1], meta['num_hidden'])
    '''
    with tf.name_scope('fc-layer-2'):
        hidden_2 = fully_connected_layer(hidden_1, meta['num_hidden'], meta['num_hidden'])
    '''
    with tf.name_scope('output-layer'):
        outputs,weights_2 = fully_connected_layer(hidden_1, meta['num_hidden'], train_data.num_classes, tf.identity)

    with tf.name_scope('error'):
        error = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
                tf.float32))

    beta = 0.01
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(error + beta * regularizers)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer().minimize(loss)
         
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        for e in xrange(meta['num_epochs']):
            running_error = 0.
            running_accuracy = 0.
            for input_batch, target_batch in train_data:
                _, batch_error, batch_acc = sess.run(
                    [train_step, error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            running_error /= train_data.num_batches
            running_accuracy /= train_data.num_batches
            print('End of epoch {0:02d} for {3}: err(train)={1:.2f} acc(train)={2:.2f}'
                  .format(e + 1, running_error, running_accuracy,meta))
            
            if (e + 1) % 5 == 0:
                valid_error = 0.
                valid_accuracy = 0.
                for input_batch, target_batch in valid_data:
                    batch_error, batch_acc = sess.run(
                        [error, accuracy], 
                        feed_dict={inputs: input_batch, targets: target_batch})
                    valid_error += batch_error
                    valid_accuracy += batch_acc
                valid_error /= valid_data.num_batches
                valid_accuracy /= valid_data.num_batches
                print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                       .format(valid_error, valid_accuracy))
                
                epoch_res_dict[e+1] = {
                    'train_err':running_error,
                    'train_acc':running_accuracy,
                    'valid_err':valid_error,
                    'valid_acc':valid_accuracy
                }

    return((meta,epoch_res_dict))


TAG = 'hidden_units_layers_2_reg'

tested_params = [{
            'num_hidden': 50,
            'num_epochs': 50,
            'layers': 2
            },{
            'num_hidden': 100,
            'num_epochs': 50,
            'layers': 2
            },{
            'num_hidden': 500,
            'num_epochs': 50,
            'layers': 2
            },{
            'num_hidden': 1000,
            'num_epochs': 50,
            'layers': 2
            },{
            'num_hidden': 200,
            'num_epochs': 50,
            'layers': 2
            }
]
'''
tested_params = [{
            'num_hidden': 400,
            'num_epochs': 80,
            'layers': 2
            }
]

'''
pool = ThreadPool(len(tested_params))
results = pool.map(runNetwork, tested_params)
print (results)
cPickle.dump(results, open('results/'+TAG+'.p', "wb"))


'''
train_data = AugmentedCIFAR10DataProvider('train', batch_size=50, transformer = transform_grey)

get_images = train_data.next()[0]

image = get_images[0]

import matplotlib.cm as cm 

plt.imshow(image.reshape((32,32)), cmap = cm.Greys_r)
plt.show()


results = runNetwork()
print (results)
'''
