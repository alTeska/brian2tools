'''
Test the metric class
'''
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises, assert_array_less
from brian2 import ms
from brian2tools import Metric, MSEMetric, GammaFactor


def test_firing_rate():
    pass


def test_init():
    Metric()
    MSEMetric()
    GammaFactor(10*ms, 0.1*ms)

    assert_raises(AssertionError, GammaFactor, dt=0.1*ms)


def test_get_gamma_factor():
    # something when it's supposed to be the same
    # something where it's zero
    pass


def test_calc():
    mse = MSEMetric()
    out_mse = np.random.rand(2, 20)
    inp_mse = np.random.rand(10, 20)

    errors = mse.calc(inp_mse, out_mse, 2)
    assert_equal(np.shape(errors), (5,))
    assert_equal(mse.calc(out_mse, out_mse, 2), [0.])
    assert(np.all(mse.calc(inp_mse, out_mse, 2) > 0))

    # calc GammaFactor


def test_get_features_mse():
    mse = MSEMetric()
    out_mse = np.random.rand(2, 20)
    inp_mse = np.random.rand(6, 20)

    mse.get_features(inp_mse, out_mse, 2)
    assert_equal(np.shape(mse.features), (6,))
    assert(np.all(np.array(mse.features) > 0))

    mse.get_features(out_mse, out_mse, 2)
    assert_equal(np.shape(mse.features), (2,))
    assert_equal(mse.features, [0., 0.])


def test_get_errors_mse():
    mse = MSEMetric()

    mse.get_errors(np.random.rand(10, 1), 2)
    assert_equal(np.shape(mse.errors), (5,))
    assert(np.all(np.array(mse.errors) > 0))

    mse.get_errors(np.zeros(10, 1), 5)
    assert_equal(np.shape(mse.errors), (2,))
    assert_equal(mse.errors, [0., 0.])


def test_get_features_gamma():
    pass


def test_get_errors_gamma():
    pass
