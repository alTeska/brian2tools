'''
Test the metric class
'''
import numpy as np
from brian2tools import Metric, MSEMetric


def test_init():
    Metric()
    MSEMetric()

def test_traces_to_features():
    inp = np.random.rand(10,2)
    metric = MSEMetric()


    out = metric.traces_to_features(np.random.rand(2,10))
    print(np.shape(out))

    pass

def test_features_to_errors():
    pass

def test_calc():
    pass
