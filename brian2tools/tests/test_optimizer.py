'''
Test the optimizer class
'''
from brian2tools import Optimizer, NevergradOptimizer

opt = Optimizer()
nevergrad_opt = NevergradOptimizer()

def test_import():
    Optimizer()
    NevergradOptimizer()

def test_ask():
    opt.ask(n_samples=10)

def test_tell():
    opt.tell()

def test_recommend():
    opt.recommend(n_best=2)
