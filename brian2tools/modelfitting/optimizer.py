import abc
import numpy as np

from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst
# from skopt import Optimizer
# from skopt.space import Real


class Optimizer(object):
    """
        Optimizer class created as a base for optimization initialization and
        performance with different libraries. To be used with modelfitting
        fit_traces.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, method='DE'):
        """Initialize the given optimization method and bounded arguments"""
        pass

    @abc.abstractmethod
    def ask(self, n_samples):
        """Returns the requested amount of parameter sets"""
        pass

    @abc.abstractmethod
    def tell(self):
        """Provides the evaluated errors from parameter sets"""
        pass

    @abc.abstractmethod
    def recommend(self, n_best):
        """Returns best recomentation provided by the method"""
        pass


class NevergradOptimizer(Optimizer):
    # def __init__(self, method='DE', input_var, bounds, **kwargs):
    def __init__(self, method='DE', **kwargs):
        super(Optimizer, self).__init__(**kwargs)
        # check if method in registry of nevergrad
        # TODO: bounds/input_var as input from fit_traces_ask_tell
        # check if input_var is str/list
        # check for bounds size and act accordingly
        arg1 = inst.var.Array(1).bounded(-5, 5).asscalar()
        arg2 = inst.var.Array(1).bounded(0, 10).asscalar()
        self.instrum = inst.Instrumentation(arg1, arg2)
        self.optim = optimizerlib.registry[method](instrumentation=self.instrum, budget=10000)

    def ask(self, n_samples):
        candidates, parameters = [], []

        for _ in range(n_samples):
            cand = self.optim.ask()
            candidates.append(cand)
            parameters.append(list(cand.args))

        return candidates, parameters

    def tell(self, candidates, errors):
        for i, candidate in enumerate(candidates):
            self.optim.tell(candidate, errors[i])

    def recommend(self):
        ans = self.optim.provide_recommendation()
        return ans
