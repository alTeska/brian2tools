'''
Test the optimizer class
'''
import numpy as np

from random import choice
from string import digits, ascii_lowercase
from numpy.testing.utils import assert_equal, assert_raises
from brian2tools import Optimizer, NevergradOptimizer, SkoptOptimizer


# tests should be split by the class?

def test_init():
    Optimizer(bounds=[[0, 1]], parameter_names={'a'}, method='DE')

def test_nevergrad_init():
    # different inputs
    NevergradOptimizer(bounds=[[0, 1]], parameter_names={'a'})
    assert_raises(AssertionError, NevergradOptimizer, bounds=[[0, 1]], parameter_names={'a', 'b'})

def test_skopt_init():
    SkoptOptimizer(bounds=[[0, 1]], parameter_names={'a'})
    assert_raises(AssertionError, SkoptOptimizer, bounds=[[0, 1]], parameter_names={'a', 'b'})

def test_nevergrad_optimizer():
    labels = ["".join([choice(digits + ascii_lowercase) for i in range(2)]) for j in range(10)]

    bounds = np.zeros((10,2))
    bounds[:, 1] = np.arange(1,11,1.)

    for n in np.arange(1,11):
        par_names, bound = labels[:n], bounds[:n]
        n_samples = np.random.randint(1, 30)

        n_opt = NevergradOptimizer(bounds=bound, parameter_names=par_names)
        params = n_opt.ask(n_samples)

        assert_equal(np.shape(params), (n_samples, n))

        for i in np.arange(1, n):
            assert all(np.array(params)[:,i-1] <= i), 'Values in params are bigger than required'
            assert all(np.array(params)[:,i-1] >  0), 'Values in params are smaller than required'

        errors = np.random.rand(n_samples)
        n_opt.tell(params, errors)

        ans = n_opt.recommend()
        er_min = (errors).argmin()
        assert params[er_min] == list(ans.args), "Optimizer didn't return the minimal value"


def test_tell():
    pass

def test_recommend():
    pass
