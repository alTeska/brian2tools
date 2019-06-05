import numpy as np
import abc
from abc import abstractmethod, abstractproperty

class Optimizer(object):
    """Optimizer class """
    __metaclass__ = abc.ABCMeta

    def __init__(self, method):
        """Initialize the given optimization method and bounded arguments"""
        pass

    @abc.abstractmethod
    def ask(self):
        """Returns the requested amount of parameter sets"""
        pass

    @abc.abstractmethod
    def tell(self):
        """Provides the evaluated errors from parameter sets"""
        pass

    @abc.abstractmethod
    def recommendation(self):
        """Returns best"""
        pass
