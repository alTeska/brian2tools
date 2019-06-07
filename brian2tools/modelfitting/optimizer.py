import abc
from numpy import array, shape
from functools import reduce

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
    def __init__(self, parameter_names, bounds, method, **kwds):
        """Initialize the given optimization method and bounded arguments"""
        pass

    @abc.abstractmethod
    def ask(self, n_samples):
        """Returns the requested number of samples of parameter sets"""
        pass

    @abc.abstractmethod
    def tell(self, parameters, errors):
        """Provides the evaluated errors from parameter sets to optimizer"""
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
            raise AssertionError("Unknown to Nevergrad optimization method:"+ method)

        # check if input var and bounds appropiate size/type
        print(len(parameter_names), shape(bounds)[1])
        if not (len(parameter_names) == shape(bounds)[0]):
            raise AssertionError("You need to specify bounds for each of the parameters")

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = inst.var.Array(1).bounded(*bounds[i]).asscalar()
            instruments.append(vars()[name])

        self.instrum = inst.Instrumentation(*instruments)
        self.optim = optimizerlib.registry[method](instrumentation=self.instrum,
                                                   budget=10000)

    def ask(self, n_samples):
        self.candidates, parameters = [], []

        for _ in range(n_samples):
            cand = self.optim.ask()
            self.candidates.append(cand)
            parameters.append(list(cand.args))

        return parameters

    def tell(self, parameters, errors):
        if not(parameters == [list(v.args) for v in self.candidates]):
            raise AssertionError("Parameters and Candidates don't have identical values")

        for i, candidate in enumerate(self.candidates):
            self.optim.tell(candidate, errors[i])

    def recommend(self):
        # TODO: check on possible parametrs
        return self.optim.provide_recommendation()



class SkoptOptimizer(Optimizer):
    """
    SkoptOptimizer instance creates all the tools necessary for the user
    to use it with scikit-optimize library.

    Parameters
    ----------
    parameter_names : (list, dict)
        List/Dict of strings with parameters to be used as instruments.
    bounds : (list)
        List with appropiate bounds for each parameter.
    method : (str), optional
        The optimization method. POssibilities: "GP", "RF", "ET", "GBRT" or
        sklearn regressor, default="GP"
    """
    def __init__(self,  parameter_names, bounds, method='GP'):
        super(Optimizer, self).__init__()

        # TODO: make this more robust
        if method.upper() not in ["GP", "RF", "ET", "GBRT"]:
            raise Warning('Unknown to skopt optimization method:'+ method)

        instruments = []
        for i, name in enumerate(parameter_names):
            vars()[name] = Real(*bounds[i])
            instruments.append(vars()[name])

        self.optimizer = skoptOptimizer(
            dimensions=instruments,
            random_state=1,
            base_estimator=method
        )

    def ask(self, n_samples):
        return self.optimizer.ask(n_points=n_samples)

    def tell(self, parameters, errors):
        self.optimizer.tell(parameters, errors.tolist());

    def recommend(self):
        xi = self.optimizer.Xi
        yii = array(self.optimizer.yi)
        return xi[yii.argmin()]
