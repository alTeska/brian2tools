import os
from numpy import mean, ones, array, where, atleast_1d, shape
from brian2 import (NeuronGroup, run, defaultclock, second,
                    device, store, restore, get_device, StateMonitor)
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice

from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import Simulation, RuntimeSimulation, CPPStandaloneSimulation

from brian2 import *   # Temporary


__all__ = ['fit_traces_standalone']


def make_dic(names, res):
    resdict = dict()
    for name, value in zip(names, res):
        resdict[name] = value

    return resdict




def fit_traces_standalone(model=None,
                          input_var=None,
                          input=None,
                          output_var=None,
                          output=None,
                          dt=None,
                          maxiter=None,
                          t_start=0*second,
                          method=('linear', 'exponential_euler', 'euler'),
                          optimizer=None,
                          n_samples=10,
                          n_rounds=1,
                          **params):
    '''
    Creates an interface for evaluation of parameters drawn by evolutionary
    algorithms (throough ask/tell interfaces).


    Parameters
    ----------
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    input_var : string
        Input variable name.
    output_var : string
        Output variable name.
    input : input data as a 2D array
    output : output data as a 2D array
    dt : time step
    maxiter : int, optional
        Maximum number of iterations.
    method: string, optional
        Integration method
    t_start: starting time of error measurement.

    optimizer: ~brian2tools.modelfitting.Optimizer children
        Child of Optimizer class, specific for each library.
    method_opt: string
        Optimization method, to be chosen within each library.
    n_samples: int
        Number of parameter samples to be optimized over.
    n_rounds: int
        Number of rounds to optimize over. (feedback provided over each round)

    TODO:
        -tolerance

    Returns
    -------
    resdict : dict
        dictionary with best parameter set
    error: float
        error value for best parameter set
    '''

    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }

    simulator = simulators[get_device().__class__.__name__]
    parameter_names = model.parameter_names

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    defaultclock.dt = dt

    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    # Check output variable
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
    if output.shape != input.shape:
        raise Exception("Input and output must have the same size")

    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt

    # Replace input variable by TimedArray
    input_traces = TimedArray(input.transpose(), dt=dt)
    # input_traces = TimedArray(input, dt=dt)

    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              '+ "% s" % repr(input_unit))


    # Add criterion with TimedArray
    output_traces = TimedArray(output.transpose(), dt=dt)
    # output_traces = TimedArray(output, dt=dt)
    output_unit = output.dim
    error_unit = output.dim**2

    model = model + Equations('total_error : %s' % repr(error_unit))

    # Population size for differential evolution
    neurons = NeuronGroup(Ntraces * n_samples, model, method=method)
    neurons.namespace['input_var'] = input_traces
    neurons.namespace['output_var'] = output_traces
    neurons.namespace['t_start'] = t_start
    neurons.namespace['Nsteps'] = Nsteps
    neurons.namespace['Ntraces'] = Ntraces

    # Record error
    neurons.run_regularly('total_error +=  (' + output_var + '-output_var(t,i % Ntraces))**2 * int(t>=t_start)',
                          when='end')

    simulator.initialize_simulation(neurons)

    # Initialize the values
    def get_param_dic(parameters):
        parameters = array(parameters)

        d = dict()

        for name, value in zip(parameter_names, parameters.T):
            d[name] = (ones((Ntraces, n_samples)) * value).T.flatten()
        return d

    def calc_error():
        err = neurons.total_error/int((duration-t_start)/defaultclock.dt)
        err = mean(err.reshape((n_samples, Ntraces)), axis=1)

        return array(err)

    # Set up the Optimizer
    optimizer.initialize(parameter_names, **params)

    ot = []
    # Run Optimization Loop
    for k in range(n_rounds):
        parameters = optimizer.ask(n_samples=n_samples)
        d = get_param_dic(parameters)

        simulator.run_simulation(neurons, duration, d, parameter_names)

        # run_neurons(neurons, duration, d, [output_var, input_var, 'total_error'])
        # monitor = run_neurons(duration, d, [output_var, input_var, 'total_error'])

        # output_traces = monitor.get_states(output_var)
        # tot_err = getattr(monitor, 'total_error' + '_')
        # inp = getattr(monitor, input_var + '_')
        # out = getattr(monitor, output_var + '_')

        # fig, ax = plt.subplots(nrows=3)
        # ax[0].plot(out.transpose())
        # ax[1].plot(inp.transpose())
        # ax[2].plot(tot_err.transpose())
        # plt.show()

        errors = calc_error()

        optimizer.tell(parameters, errors)
        res = optimizer.recommend()

        # create output variables
        resdict = make_dic(parameter_names, res)

        index_param = where(array(parameters) == array(res))
        ii = index_param[0]
        error = errors[ii]  # TODO: re-check

    return resdict, error
