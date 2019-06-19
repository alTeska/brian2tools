import os
import abc
from numpy import atleast_1d
from brian2 import (run, device, store, restore)


def initialize_parameter(variableview, value):
    variable = variableview.variable
    array_name = device.get_array_name(variable)
    static_array_name = device.static_array(array_name, value)
    device.main_queue.append(('set_by_array', (array_name,
                                               static_array_name,
                                               False)))
    return static_array_name


def initialize_neurons(params_names, neurons, d):
    params_init = dict()

    for name in params_names:
        params_init[name] = initialize_parameter(neurons.__getattr__(name),
                                                 d[name])
    return params_init


def set_parameter_value(identifier, value):
    atleast_1d(value).tofile(os.path.join(device.project_dir,
                                          'static_arrays',
                                          identifier))


def run_again():
    device.run(device.project_dir, with_output=False, run_args=[])


def set_states(init_dict, values):
    # TODO: add a param checker
    for obj_name, obj_values in values.items():
        set_parameter_value(init_dict[obj_name], obj_values)



class Simulation(object):
    """
    Simluation class
    """
    __metaclass__= abc.ABCMeta
    def __init__(self):
        pass

    @abc.abstractmethod
    def initialize_simulation(self, neurons):
        """
        Prepares the simulation for running

        Parameters
        ----------
        neurons: NeuronGroup initialized instance
            neurons to be simulatied
        """
        pass

    @abc.abstractmethod
    def run_simulation(self, neurons, duration, params):
        """
        Restores the network, sets neurons to required parameters and runs
        the simulation

        Parameters
        ----------
        neurons: NeuronGroup initialized instance
            neurons to be simulatied
        duration: simulation duration [ms]

        params: dict
            parameters to be set
        """
        pass


class RuntimeSimulation(Simulation):
    """Simulation class created for use with RuntimeDevice"""
    def initialize_simulation(self, neurons):
        store()

    def run_simulation(self, neurons, duration, params, params_names):
        restore()
        neurons.set_states(params, units=False)
        run(duration, namespace={})


class CPPStandaloneSimulation(Simulation):
    """Simulation class created for use with CPPStandaloneDevice"""
    def initialize_simulation(self, neurons):
        pass

    def run_simulation(self, neurons, duration, params, params_names):
        """
        simulation has to be run in two stages in order to initalize the
        code generaion
        """
        if not device.has_been_run:
            self.params_init = initialize_neurons(params_names, neurons,
                                                  params)
            run(duration)

        else:
            set_states(self.params_init, params)
            run_again()
