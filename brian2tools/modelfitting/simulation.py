import abc
from brian2 import (NeuronGroup, run, defaultclock, second,
                    device, store, restore, get_device, StateMonitor)


class Simulation(object):
    """
    Simluation class
    """
    __metaclass__= abc.ABCMeta
    def __init__(self):
        pass

    @abc.abstractmethod
    def initialize_simulation(self):
        """Prepares the simulation for running"""
        pass

    @abc.abstractmethod
    def run_simulation(self):
        """
        Restores the network, sets neurons to required parameters and runs
        the simulation
        """
        pass


class RuntimeSimulation(Simulation):
    def initialize_simulation(self):
        store()

    def run_simulation(self):
        pass

class CPPStandaloneSimulation(Simulation):
    def initialize_simulation(self):
        pass

    def run_simulation(self):
        pass
