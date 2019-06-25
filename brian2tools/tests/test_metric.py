'''
Test the metric class
'''
from brian2tools import Metric, MSEMetric


def test_init():
    Metric()
    MSEMetric()
