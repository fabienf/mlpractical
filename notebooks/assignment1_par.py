# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
import matplotlib.pyplot as plt
from mlp.optimisers import Optimiser
from mlp.schedulers import ConstantLearningRateScheduler,ExpLearningRateScheduler
from mlp.learning_rules import GradientDescentLearningRule
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from multiprocessing.dummy import Pool as ThreadPool
#import multiprocessing
import json


tag_task = 'Task_1'
tag_type = 'Exp_rule'

num_epochs = 100  # number of training epochs to perform
stats_interval = 5  # epoch interval between recording and printing stats
input_dim, output_dim, hidden_dim = 784, 10, 100

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, learning_rate,n=1,r=1):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
    
    #schedulers = [ConstantLearningRateScheduler(learning_rate)]
    schedulers = [ExpLearningRateScheduler(n,r)]
    
    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, schedulers)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2


def setup_run_with_param(param):
	#learning_rate = param
	learning_rate=0.2
	n = param[0]
	r = param[1]
	# Seed a random number generator
	seed = 10102016 
	rng = np.random.RandomState(seed)

	# Set up a logger object to print info about the training run to stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.handlers = [logging.StreamHandler()]

	# Create data provider objects for the MNIST data set
	train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
	valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

	print 'Created Providers'


	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)

	model = MultipleLayerModel([
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    SigmoidLayer(),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    SigmoidLayer(),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
	])

	error = CrossEntropySoftmaxError()

	# Use a basic gradient descent learning rule
	#learning_rule = GradientDescentLearningRule(learning_rate=param)	
	learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

	stats, keys, run_time, fig_1, ax_1, fig_2, ax_2 = train_model_and_plot_stats(
	    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, learning_rate,n,r)

	# Save figure to current directory in PDF format
	#plt.show()

	#fig_1.savefig('figures/'+tag_task+'/'+tag_type+'_learn_'+str(learning_rate)+'_fig_1.pdf')
	#fig_2.savefig('figures/'+tag_task+'/'+tag_type+'_learn_'+str(learning_rate)+'_fig_2.pdf')
	fig_1.savefig('figures/'+tag_task+'/'+tag_type+'_n_'+str(n)+'_r_'+str(r)+'_fig_1.pdf')
	fig_2.savefig('figures/'+tag_task+'/'+tag_type+'_n_'+str(n)+'_r_'+str(r)+'_fig_2.pdf')


	print('	   param '+ str(n)+' '+str(r) )
	print('    final error(train) = {0:.2e}').format(stats[-1, keys['error(train)']])
	print('    final error(valid) = {0:.2e}').format(stats[-1, keys['error(valid)']])
	print('    final acc(train)   = {0:.2e}').format(stats[-1, keys['acc(train)']])
	print('    final acc(valid)   = {0:.2e}').format(stats[-1, keys['acc(valid)']])
	print('    run time per epoch = {0:.2f}s').format(run_time * 1. / num_epochs)

	return [n,r, stats[-1, keys['error(train)']],stats[-1, keys['error(valid)']],stats[-1, keys['acc(train)']],stats[-1, keys['acc(valid)']],run_time * 1. / num_epochs]
	#return [param, stats[-1, keys['error(train)']],stats[-1, keys['error(valid)']],stats[-1, keys['acc(train)']],stats[-1, keys['acc(valid)']],run_time * 1. / num_epochs]


pool = ThreadPool(16) 
my_array = [[1,1],[1,5],[1,10],[1,20],[1,0.5],[1,0.1],[0.5,1],[0.5,5],[0.5,10],[0.5,20],[0.5,0.5],[0.5,0.1],[2,1],[2,5],[2,10],[2,20],[2,0.5],[2,0.1]]
#my_array = [[1,1],[1,5],[1,10],[1,20]]
#my_array = [0.5,0.2,0.1,0.01]

results = pool.map(setup_run_with_param, my_array)
print results
with open('figures/'+tag_task+'/'+tag_type+'.txt', "w+") as f:
	json.dump(results,f)
'''
#result_queue = multiprocessing.Queue()
jobs = [multiprocessing.Process(target=setup_run_with_param,args=(arg)) for arg in my_array]
for job in jobs: job.start()
for job in jobs: job.join()
'''

#setup_run_with_param([1,1])







