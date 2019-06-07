'''
Test the optimizer class
'''
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises
from brian2tools import Optimizer, NevergradOptimizer, SkoptOptimizer

# tests should be split by the class?

def test_init():
    # Optimizer(bounds=[[0, 1]], parameter_names=['a'], method='DE')
    Optimizer(bounds=[[0, 1]], parameter_names={'a'}, method='DE')

def test_nevergrad_init():
    # different inputs
    NevergradOptimizer(bounds=[[0, 1]], parameter_names={'a'})
    assert_raises(AssertionError, NevergradOptimizer, bounds=[[0, 1]], parameter_names={'a', 'b'})



def test_skopt_init():
    SkoptOptimizer(bounds=[[0, 1]], parameter_names={'a'})

def test_ask():
    # loop through those and check the sizes
    param_names = {'a', 'b'}
    n_params = 10

    n_opt = NevergradOptimizer(bounds=[[0, 1]], parameter_names=['a'])
    param = n_opt.ask(10)
    assert_equal(np.shape(param), (10,1))

    n_opt = NevergradOptimizer(bounds=[[0, 1], [0, 2]], parameter_names=['a', 'b'])
    param = n_opt.ask(10)
    assert_equal(np.shape(param), (10,2))


def test_tell():
    # param = np.random.random((10,2))

    n_opt = NevergradOptimizer(bounds=[[0, 1], [0, 1]], parameter_names=['a', 'b'])
    param = n_opt.ask(10)
    error = np.random.random((10))
    n_opt.tell(param, error)

    # parameters and errors should be same size
    # parameters should be the same as candidates / candidates will be overwritten every round
    # opt.tell(parameters, errors)
    pass

def test_recommend():

    n_opt = NevergradOptimizer(bounds=[[0, 1], [0, 1]], parameter_names=['a', 'b'])
    param = n_opt.ask(10)
    error = np.random.random((10))
    n_opt.tell(param, error)
    n_opt.recommend()

    pass
    # number of best results
