import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider, AugmentedCIFAR10DataProvider
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import cPickle
from scipy.ndimage.interpolation import shift



seed = 7474747 
rng = np.random.RandomState(seed)

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu,dropout = 1.,doDrop=False):
	weights = tf.Variable(
		tf.truncated_normal(
			[input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
		'weights')
	biases = tf.Variable(tf.zeros([output_dim]), 'biases')
	outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
	if (doDrop):
		if (dropout!=1.):
			outputs = tf.nn.dropout(outputs,dropout)
	return outputs, weights



def transform_grey(inputs):
	new_inputs = []
	for image in inputs:
		grey = np.zeros(shape=(1024))
		for x in range(1024):
			grey[x] = (image[x]+image[x+1024]+image[x+2048])/3.0
		new_inputs.append(grey)
	return new_inputs


def random_flip(inputs, rng):
	orig_ims = inputs.reshape((-1, 3, 32, 32)).transpose(0,1,3,2)
	indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 2, False)
	for idx, item in enumerate(indices):
		orig_ims[item] = np.fliplr(orig_ims[item])
	return orig_ims.transpose(0,1,3,2).reshape((-1, 3072))

def random_bright(inputs, rng):
	orig_ims = inputs.reshape((-1, 3, 32, 32)).transpose(0,1,3,2)
	indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
	bright = rng.uniform(-0.1,0.1)
	for idx, item in enumerate(indices):
		orig_ims[item] = orig_ims[item]+bright
	orig_ims[orig_ims>=1] = 1.
	orig_ims[orig_ims<=0] = 0.
	return orig_ims.transpose(0,1,3,2).reshape((-1, 3072))

def random_contrast(inputs, rng):
	orig_ims = inputs.reshape((-1, 3, 32, 32)).transpose(0,1,3,2)
	indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
	contrast = rng.uniform(0.9,1.1)
	for idx, item in enumerate(indices):
		orig_ims[item] = orig_ims[item]*contrast
	orig_ims[orig_ims>=1] = 1.
	return orig_ims.transpose(0,1,3,2).reshape((-1, 3072))



def random_noise(inputs, rng):
	orig_ims = inputs.reshape((-1, 3, 32, 32))
	indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
	noise = rng.uniform(0,0.05,size=(3,32,32))
	for idx, item in enumerate(indices):
		orig_ims[item] = orig_ims[item]+noise
	orig_ims[orig_ims>=1] = 1.
	orig_ims[orig_ims<=0] = 0.
	return orig_ims.reshape((-1, 3072))


def random_mix(inputs, rng):
	choice = rng.randint(2,size=1)[0]
	if (choice==0):
		return random_contrast(inputs, rng)
	if (choice==1):
		return random_flip(inputs, rng)


def runNetwork(meta=None):

	train_data = AugmentedCIFAR10DataProvider('train-original', batch_size=50, transformer=random_flip,rng=rng)
	valid_data = CIFAR10DataProvider('test', batch_size=50)

	#The results to be returned
	epoch_res_dict = {}

	inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
	targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
	dropout = tf.placeholder(tf.float32, name='dropout')


	with tf.name_scope('fc-layer-1'):
		hidden_1, weights_1 = fully_connected_layer(inputs, train_data.inputs.shape[1], meta['num_hidden'], dropout=dropout,doDrop=meta['drop'])
	
	with tf.name_scope('fc-layer-2'):
		hidden_2, weights_2 = fully_connected_layer(hidden_1, meta['num_hidden'], meta['num_hidden'], dropout=dropout,doDrop=meta['drop'])
	
	with tf.name_scope('fc-layer-3'):
		hidden_3, weights_3 = fully_connected_layer(hidden_2, meta['num_hidden'], meta['num_hidden'], dropout=dropout,doDrop=meta['drop'])
	
	with tf.name_scope('output-layer'):
		outputs, weights_4 = fully_connected_layer(hidden_3, meta['num_hidden'], train_data.num_classes, tf.identity,doDrop=False)

	with tf.name_scope('error'):
		if (meta['L2']):
			beta = 0.01
		else:
			beta = 0.
		regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3)
		error = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(outputs, targets) + beta * regularizers)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(
				tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
				tf.float32))

	with tf.name_scope('train'):
		train_step = tf.train.AdagradOptimizer(0.01).minimize(error)

		 
	init = tf.global_variables_initializer()


	with tf.Session() as sess:
		sess.run(init)
		for e in xrange(meta['num_epochs']):
			running_error = 0.
			running_accuracy = 0.
			for input_batch, target_batch in train_data:
				_, batch_error, batch_acc = sess.run(
					[train_step, error, accuracy], 
					feed_dict={inputs: input_batch, targets: target_batch, dropout: 0.8})
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
						feed_dict={inputs: input_batch, targets: target_batch, dropout: 1.0})
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


TAG = 'hidden_units_layers_4_flip_test'

tested_params = [{
			'num_hidden': 200,
			'num_epochs': 50,
			'layers': 4,
			'L2': False,
			'drop': False,
			'name': 'L2 and drop-out'
			}
]



pool = ThreadPool(len(tested_params))
results = pool.map(runNetwork, tested_params)
print results
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
