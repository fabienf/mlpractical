import os
os.environ["MLP_DATA_DIR"] = "/afs/inf.ed.ac.uk/user/s12/s1247438/MLP/mlpractical/data/"
import sys
sys.path.append("/afs/inf.ed.ac.uk/user/s12/s1247438/MLP/mlpractical/notebooks/CW1/")

# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
from mlp.optimisers import Optimiser
from mlp.learning_rules import AdaGradLearningRule
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from multiprocessing.dummy import Pool as ThreadPool
import cPickle



tag_type = 'Ada'

num_epochs = 100  # number of training epochs to perform
stats_interval = 5  # epoch interval between recording and printing stats
input_dim, output_dim, hidden_dim = 784, 10, 100




def setup_run_with_param(param):
    #learning_rate = param
    # Seed a random number generator
    learning_rate=param[0]
    eps=param[1]
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
    learning_rule = AdaGradLearningRule(learning_rate,eps)

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}


    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    print('       param '+ str(learning_rate)+' '+str(eps) )
    print('    final error(train) = {0:.2e}').format(stats[-1, keys['error(train)']])
    print('    final error(valid) = {0:.2e}').format(stats[-1, keys['error(valid)']])
    print('    final acc(train)   = {0:.2e}').format(stats[-1, keys['acc(train)']])
    print('    final acc(valid)   = {0:.2e}').format(stats[-1, keys['acc(valid)']])
    print('    run time per epoch = {0:.2f}s').format(run_time * 1. / num_epochs)

    return param, stats, keys, run_time


#Create workers

tested_params = [[1.5,1e-8],[1.2,1e-8],[0.9,1e-8],[0.5,1e-8],[0.03,1e-8],[0.2,1e-8],[0.1,1e-8],[0.01,1e-8],[1.5,1e-6],[1.2,1e-6],[0.9,1e-6],[0.5,1e-6],[0.03,1e-6],[0.2,1e-6],[0.1,1e-6],[0.01,1e-6]]

pool = ThreadPool(len(tested_params))
results = pool.map(setup_run_with_param, tested_params)
print results
cPickle.dump(results, open(tag_type+'.p', "wb"))


#setup_run_with_param([0.9,1e-8])
