'''
Test the optimizer class
'''
from brian2tools import Optimizer, NevergradOptimizer, SkoptOptimizer

opt = Optimizer(bounds=[[0,1]], parameter_names=['a'], method='DE')
nevergrad_opt = NevergradOptimizer(bounds=[[0,1]], parameter_names=['a'])
skopt_opt = SkoptOptimizer(bounds=[[0,1]], parameter_names=['a'])

def test_import():
    Optimizer(bounds=[[0,1]], parameter_names=['a'], method='DE')
    NevergradOptimizer(bounds=[[0,1]], parameter_names={'a'})

def test_ask():
    opt.ask(n_samples=10)

def test_tell():
    opt.tell()

def test_recommend():
    opt.recommend(n_best=2)
