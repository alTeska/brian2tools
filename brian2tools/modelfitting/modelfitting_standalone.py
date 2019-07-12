from numpy import mean, ones, array, arange
from brian2 import (NeuronGroup,  defaultclock, get_device, Network,
                    StateMonitor, SpikeMonitor, ms, second)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric

__all__ = ['fit_traces_standalone', 'fit_spikes']


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = dict()
    for name, value in zip(names, values):
        result_dict[name] = value

    return result_dict


def get_param_dic(params, param_names, n_traces, n_samples):
    """Transform parameters into a dictionary of appropiate size"""
    params = array(params)

    d = dict()

    for name, value in zip(param_names, params.T):
        d[name] = (ones((n_traces, n_samples)) * value).T.flatten()
    return d


def get_spikes(monitor):
    """
    Get spikes from spike monitor for each neuron, change from dict to a list,
    remove units
    """
    spike_trains = monitor.spike_trains()

    spikes = []
    for i in arange(len(spike_trains)):
        spike_list = spike_trains[i] / ms
        spikes.append(spike_list)

    return spikes


def fit_setup(model=None, dt=None, param_init=None, input_var=None,
              metric=None):
    """
    Function sets up simulator and checks the variables for fit_traces/
    fit_spikes.

    Returns:
        simulator
    """
    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }

    simulator = simulators[get_device().__class__.__name__]

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')

    defaultclock.dt = dt

    # Check initialization of params
    if param_init:
        for param, val in param_init.items():
            if not (param in model.identifiers or param in model.names):
                raise Exception("%s is not a model variable or an identifier \
                                in the model")

    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    # Check Metric
    if not (isinstance(metric, Metric) or metric is None):
        raise Exception("metric has to be a child of class Metric or None")

    return simulator


def setup_neuron_group(model, n_neurons, method, threshold, reset, refractory,
                       param_init, **namespace):
    """
    Setup neuron group, initialize required number of neurons, create namespace
    and initite the parameters
    """
    neurons = NeuronGroup(n_neurons, model, method=method, threshold=threshold,
                          reset=reset, refractory=refractory, name='neurons')
    for name in namespace:
        neurons.namespace[name] = namespace[name]

    if param_init:
        neurons.set_states(param_init)

    return neurons


def fit_traces_standalone(model=None,
                          input_var=None,
                          input=None,
                          output_var=None,
                          output=None,
                          dt=None,
                          method=('linear', 'exponential_euler', 'euler'),
                          optimizer=None,
                          metric=None,
                          n_samples=10,
                          n_rounds=1,
                          verbose=True,
                          param_init=None,
                          reset=None, refractory=False, threshold=None,
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
    input : input data as a 2D array
    output_var : string
        Output variable name.
    output : output data as a 2D array
    dt : time step
    method: string, optional
        Integration method
    optimizer: ~brian2tools.modelfitting.Optimizer children
        Child of Optimizer class, specific for each library.
    metric: ~brian2tools.modelfitting.Metric children
        Child of Metric class, specifies optimization metric
    n_samples: int
        Number of parameter samples to be optimized over.
    n_rounds: int
        Number of rounds to optimize over. (feedback provided over each round)
    verbose: bool
        Provide error feedback at each round
    param_init: dict
        Dictionary of variables to be initialized with the value
    **params:
        bounds for each parameter

    Returns
    -------
    result_dict : dict
        dictionary with best parameter set
    error: float
        error value for best parameter set

    TODO:
     - resolve t_start
     - tolerance

    '''
    # Check output variable
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
        if output.shape != input.shape:
            raise Exception("Input and output must have the same size")

    simulator = fit_setup(model,dt, param_init, input_var, metric)

    parameter_names = model.parameter_names
    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt
    n_neurons = Ntraces * n_samples

    if metric is None:
        model = model + Equations('total_error : %s' % repr(output.dim**2))

    # Replace input variable by TimedArray
    output_traces = TimedArray(output.transpose(), dt=dt)
    input_traces = TimedArray(input.transpose(), dt=dt)
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              ' + "% s" % repr(input.dim))

    # Setup NeuronGroup
    neurons = setup_neuron_group(model, n_neurons, method, threshold, reset,
                                 refractory, param_init,
                                 input_var=input_traces,
                                 output_var=output_traces,
                                 Ntraces=Ntraces)

    # Online metric calc
    if metric is None:
        t_start = 0*second
        neurons.namespace['t_start'] = t_start
        neurons.run_regularly('total_error +=  (' + output_var + '-output_var\
                            (t,i % Ntraces))**2 * int(t>=t_start)', when='end')

        def calc_error():
            """calculate online error"""
            err = neurons.total_error/int((duration-t_start)/defaultclock.dt)
            err = mean(err.reshape((n_samples, Ntraces)), axis=1)

            return array(err)

    # Set up Simulator and Optimizer
    monitor = StateMonitor(neurons, output_var, record=True, name='monitor')
    network = Network(neurons, monitor)
    simulator.initialize(network)
    optimizer.initialize(parameter_names, **params)

    # Run Optimization Loop
    for k in range(n_rounds):
        parameters = optimizer.ask(n_samples=n_samples)
        d_param = get_param_dic(parameters, parameter_names, Ntraces,
                                n_samples)
        simulator.run(duration, d_param, parameter_names)

        if isinstance(metric, Metric):
            traces = getattr(simulator.network['monitor'], output_var)
            errors = metric.calc(traces, output, Ntraces)
        elif metric is None:
            errors = calc_error()

        optimizer.tell(parameters, errors)
        res = optimizer.recommend()

        # create output variables
        result_dict = make_dic(parameter_names, res)
        error = min(errors)

        if verbose:
            print('round {} with error {}'.format(k, error))
            print("resulting parameters:", result_dict)

    return result_dict, error


def fit_spikes(model=None,
               input_var=None,
               input=None,
               output=None,
               dt=None,
               method=('linear', 'exponential_euler', 'euler'),
               optimizer=None,
               metric=None,
               n_samples=10,
               n_rounds=1,
               verbose=True,
               param_init=None,
               reset=None, refractory=False, threshold=None,
               **params):
    """Fit spikes"""
    simulator = fit_setup(model=model, dt=dt, param_init=param_init,
                          input_var=input_var, metric=metric)

    parameter_names = model.parameter_names
    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt

    n_neurons = Ntraces * n_samples

    # Replace input variable by TimedArray
    input_traces = TimedArray(input.transpose(), dt=dt)
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                               ' + "% s" % repr(input.dim))

    # Population size for differential evolution
    neurons = setup_neuron_group(model, n_neurons, method, threshold, reset,
                                 refractory, param_init,
                                 input_var=input_traces,
                                 Ntraces=Ntraces)

    # Set up Simulator and Optimizer
    monitor = SpikeMonitor(neurons, record=True, name='monitor')
    network = Network(neurons, monitor)
    simulator.initialize(network)
    optimizer.initialize(parameter_names, **params)

    # Run Optimization Loop
    for k in range(n_rounds):
        parameters = optimizer.ask(n_samples=n_samples)

        d_param = get_param_dic(parameters, parameter_names, Ntraces,
                                n_samples)

        simulator.run(duration, d_param, parameter_names)

        spikes = get_spikes(monitor)

        errors = metric.calc(spikes, output, Ntraces)
        optimizer.tell(parameters, errors)
        res = optimizer.recommend()

        # create output variables
        result_dict = make_dic(parameter_names, res)
        error = min(errors)

        if verbose:
            print('round {} with error {}'.format(k, error))
            print("resulting parameters:", result_dict)

    return result_dict, error
