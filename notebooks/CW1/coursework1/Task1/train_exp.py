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
from mlp.schedulers import ExpLearningRateScheduler
from mlp.learning_rules import GradientDescentLearningRule
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from multiprocessing.dummy import Pool as ThreadPool
import cPickle



tag_task = 'Task_1'
tag_type = 'Exp_rule'

num_epochs = 100  # number of training epochs to perform
stats_interval = 5  # epoch interval between recording and printing stats
input_dim, output_dim, hidden_dim = 784, 10, 100




def setup_run_with_param(param):
    #learning_rate = param
    # Seed a random number generator
    n=param[0]
    r=param[1]
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
    learning_rule = GradientDescentLearningRule()

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    schedulers = [ExpLearningRateScheduler(n,r)]

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, schedulers)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    print('       param '+ str(n)+' '+str(r) )
    print('    final error(train) = {0:.2e}').format(stats[-1, keys['error(train)']])
    print('    final error(valid) = {0:.2e}').format(stats[-1, keys['error(valid)']])
    print('    final acc(train)   = {0:.2e}').format(stats[-1, keys['acc(train)']])
    print('    final acc(valid)   = {0:.2e}').format(stats[-1, keys['acc(valid)']])
    print('    run time per epoch = {0:.2f}s').format(run_time * 1. / num_epochs)

    return param, stats, keys, run_time


#Create workers

tested_params = [[2,50000],[1,50000],[0.5,50000],[0.1,50000],[0.01,50000],[2,5000],[1,5000],[0.5,5000],[0.1,5000],[0.01,5000],[2,50],[1,50],[0.5,50],[0.1,50],[0.01,50],[2,50],[1,5],[0.5,5],[0.1,5],[0.01,5]]

pool = ThreadPool(len(tested_params))
results = pool.map(setup_run_with_param, tested_params)
print results
cPickle.dump(results, open(tag_type+'.p', "wb"))


#obj = cPickle.load(open('save.p', 'rb'))
