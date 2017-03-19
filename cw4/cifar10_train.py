# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10
from IPython import embed
import math



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('name', 'test',
                           """Name of current model """)
tf.app.flags.DEFINE_string('train_dir', 'models/'+FLAGS.name+'/train',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_boolean('log_device_placement', False,
							"""Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_train_examples', 40000,
							"""Number of training data examples""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 10,
							"""Number of epochs to run""")
import cifar10_eval


def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default() as g:
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Get images and labels for CIFAR-10.
		images, labels = cifar10.distorted_inputs()

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = cifar10.inference(images)

		# Calculate loss.
		loss = cifar10.loss(logits, labels)


		accuracy = cifar10.accuracy(logits,labels)


		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = cifar10.train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""
			'''
			def after_create_session(session, coord):
				self._sess = session
			'''
			def begin(self):
				print ('begin')
				self._step = 0
				self._epoch = 1
				self._running_acc = 0.
				self._running_loss = 0.
				self._epoch_time = time.time()
				self._summary_writer = tf.summary.FileWriter(FLAGS.train_dir,g)


			def before_run(self, run_context):
				self._step += 1
				self._start_time = time.time()
				return tf.train.SessionRunArgs([loss, accuracy])  # Asks for loss value.
								
			def after_run(self, run_context, run_values):
				loss_value = run_values.results[0]
				accuracy_value = run_values.results[1]
				self._running_acc +=accuracy_value
				self._running_loss += loss_value
				if self._step % 10 == 0:
					duration = time.time() - self._start_time
					format_str = ('%s: step %d, loss = %.2f, accuracy = %.2f, time/batch = %.3f')
					print (format_str % (datetime.now(), self._step, loss_value, accuracy_value, duration))
				#per each elapsed epoch
				num_train_batches_per_epoch = math.ceil(FLAGS.num_train_examples / FLAGS.batch_size)
				if ( self._step % num_train_batches_per_epoch == 0):
					#get loss and acc per epoch
					self._running_acc/=num_train_batches_per_epoch
					self._running_loss/=num_train_batches_per_epoch
					#get time of running
					duration = time.time() - self._epoch_time
					#print epoch results
					format_str = ('%s: EPOCH %d, error = %.2f, accuracy = %.2f, time per epoch %.3f secs)')
					print (format_str % (datetime.now(), self._epoch, self._running_loss,self._running_acc, duration))
					#write values to summary
					summary = tf.Summary()
					summary.value.add(tag='accuracy', simple_value=self._running_acc)
					summary.value.add(tag='error', simple_value=self._running_loss)
					self._summary_writer.add_summary(summary, self._epoch)
					#reset values
					self._running_acc = 0.
					self._running_loss = 0.
					self._epoch_time = time.time()
					self._epoch+=1
					#eval
					cifar10_eval.evaluate()
				
		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=FLAGS.train_dir,
				hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_epochs * math.ceil(FLAGS.num_train_examples / FLAGS.batch_size)),
							 tf.train.NanTensorHook(loss),
							 _LoggerHook()],
				#save_checkpoint_secs=None,
				save_checkpoint_secs=50,
				save_summaries_steps=100,
				config=tf.ConfigProto(
						log_device_placement=FLAGS.log_device_placement)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	cifar10_eval.evalDelFolders()
	train()


if __name__ == '__main__':
	tf.app.run()
