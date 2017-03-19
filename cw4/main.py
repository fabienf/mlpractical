import os
import tensorflow as tf
import numpy as np
import cifar10_input
from IPython import embed



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
						   """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
							"""Train the model using fp16.""")


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
	images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	labels: Labels. 1D tensor of [batch_size] size.

  Raises:
	ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
	raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
												  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
	images = tf.cast(images, tf.float16)
	labels = tf.cast(labels, tf.float16)
  return images, labels


def original_inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
	eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
	images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	labels: Labels. 1D tensor of [batch_size] size.

  Raises:
	ValueError: If no data_dir
  """
  print "get inputs"
  if not FLAGS.data_dir:
	raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
										data_dir=data_dir,
										batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
	images = tf.cast(images, tf.float16)
	labels = tf.cast(labels, tf.float16)
  return images, labels


def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
	weights = tf.Variable(
		tf.truncated_normal(
			[input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
		'weights')
	biases = tf.Variable(tf.zeros([output_dim]), 'biases')
	outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
	return outputs, weights

def runGraph(meta=None):

	print ("run graph")
	#inputs = tf.placeholder(tf.float32, [None, 1728], 'inputs')
	#targets = tf.placeholder(tf.float32, [None, 10], 'targets')
	#is_train = tf.placeholder(tf.bool)
	is_train =  tf.Variable(True,  name='training')
	#inputs, targets = tf.cond(is_train, lambda: original_inputs(eval_data=False), lambda: original_inputs(eval_data=True))
	inputs, targets = original_inputs(eval_data=False)
	embed(header="acc")
	reshaped_in = tf.reshape(inputs, [FLAGS.batch_size, -1])
	#targets = tf.reshape(targets,[FLAGS.batch_size, 1])
	tf.Print(targets,[targets])
	#The results to be returned
	epoch_res_dict = {}
	print ("data loaded")



	with tf.name_scope('fc-layer-1'):
		print "layer 1"
		hidden_1, weights_1 = fully_connected_layer(reshaped_in, 1728, 200)

	with tf.name_scope('fc-layer-1'):
		print "layer 1"
		hidden_2, weights_2 = fully_connected_layer(hidden_1, 200, 200)
	with tf.name_scope('fc-layer-1'):
		print "layer 1"
		hidden_3, weights_3 = fully_connected_layer(hidden_2, 200, 200)
	
	with tf.name_scope('output-layer'):
		print "layer 2"
		outputs, weights_4 = fully_connected_layer(hidden_3, 200, 10, tf.identity)

	with tf.name_scope('error'):
		print "er"
		error = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, targets))

	with tf.name_scope('accuracy'):

		accuracy = tf.reduce_mean(tf.cast(
				tf.equal(tf.argmax(outputs, 1), tf.cast(targets[0],tf.int64)), 
				tf.float32))
		

	with tf.name_scope('train'):
		train_step = tf.train.AdagradOptimizer(0.01).minimize(error)

		 
	init = tf.global_variables_initializer()


	with tf.Session() as sess:
		sess.run(init)
			# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

			for e in xrange(10):
				running_error = 0.
				running_accuracy = 0.
				for i in xrange(40000/128):
					if coord.should_stop():
						break
					_, batch_error, batch_acc = sess.run(
						[train_step, error, accuracy], 
						feed_dict={is_train:True})
					running_error += batch_error
					running_accuracy += batch_acc
				running_error /= 40000/128
				running_accuracy /= 40000/128
				print('End of epoch {0:02d} for {3}: err(train)={1:.2f} acc(train)={2:.2f}'
					  .format(e + 1, running_error, running_accuracy,meta))
				
				if (e + 1) % 1 == 0:
					if coord.should_stop():
						break
					valid_error = 0.
					valid_accuracy = 0.
					for i in xrange(40000/128):
						batch_error, batch_acc = sess.run(
							[error, accuracy], 
							feed_dict={is_train:False})
						valid_error += batch_error
						valid_accuracy += batch_acc
					valid_error /= 40000/128
					valid_accuracy /= 40000/128
					print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
						   .format(valid_error, valid_accuracy))
					
					epoch_res_dict[e+1] = {
						'train_err':running_error,
						'train_acc':running_accuracy,
						'valid_err':valid_error,
						'valid_acc':valid_accuracy
					}
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

	return((meta,epoch_res_dict))

runGraph()

