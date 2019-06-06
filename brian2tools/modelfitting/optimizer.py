import abc
# import numpy as np

from nevergrad.optimization import optimizerlib, registry
from nevergrad import instrumentation as inst
from skopt import Optimizer as skoptOptimizer
from skopt.space import Real


class Optimizer(object):
    """
    Optimizer class created as a base for optimization initialization and
    performance with different libraries. To be used with modelfitting
    fit_traces.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, parameter_names, bounds, method='DE'):
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
    """
    NevergradOptimizer instance creates all the tools necessary for the user
    to use it with Nevergrad library.

    Parameters
    ----------
    parameter_names : (list, dict)
        List/Dict of strings with parameters to be used as instruments.
    bounds : (list)
        List with appropiate bounds for each parameter.
    method : (str), optional
        The optimization method. By default differential evolution, can be
        chosen from any method in Nevergrad registry
    """

    def __init__(self,  parameter_names, bounds, method='DE'):
        super(Optimizer, self).__init__()

        if method not in list(registry.keys()):
            raise AssertionError('Unknown to Nevergrad optimizatino method: '+  method)

        # TODO: bounds/input_var as input from fit_traces_ask_tell
        # check if input var and bounds appropiate size/type

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = inst.var.Array(1).bounded(*bounds[i]).asscalar()
            instruments.append(vars()[name])

        self.instrum = inst.Instrumentation(*instruments)
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
        # TODO: check on possible parametrs
        return self.optim.provide_recommendation()
